"""
train_feature_fusion.py
=======================
Row-2 intermediate (feature-level) fusion, run inside the SAME harness as
``train.py`` — identical data splits, training loop, checkpoint format, and
``val.py`` evaluation.

SELF-CONTAINED: the encoders, fusion blocks, and FeatureFusionNet are inlined
below (copied verbatim from the Row-2 design, kernels unchanged). This file does
not depend on row2_feature_fusion.py at all. Its only project dependency is train.py
(and what it already imports: data.py / _preprocessors.py).

Run
---
    python data.py                  # build ./splits   (unchanged)
    python train_feature_fusion.py  # -> ./checkpoints/<run>/best.pt
    # then evaluate with val.py after the one-line import swap (see note at bottom)

Training note
-------------
The original Row-2 estimator trains with an uncertainty-weighted multi-task loss
over {joint, eeg-aux, emg-aux}. train.py's harness uses a single CrossEntropyLoss
on the joint head, so ONLY the joint head is supervised here; the aux heads exist
but receive no gradient. Modality dropout inside FeatureFusionNet stays active
during training. Architecture and kernels are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
from datetime import datetime

# train.py resolves CHECKPOINT_PATH at import time; ensure the dir exists first.
BASE_DIR = Path(__file__).resolve().parent
(BASE_DIR / "checkpoints").mkdir(exist_ok=True)

import train as base

SPLITS_DIR = BASE_DIR / "splits"
ARCH = "feature"

# --- architecture choices (MUST match between training here and val.py load) ---
FUSION_MODE = "gate"        # "gate" | "attention"  (gate = the Row-2 default for closed-vocab)
USE_BILSTM = False
USE_HUDGINS = True


# =============================================================================
#  Inlined Row-2 model code (verbatim; kernel sizes unchanged)
# =============================================================================
SEED = 42
D_MODEL = 64                 # shared token / embedding width
EEG_F_TEMPORAL = 40
EEG_KERNEL_TIME = 25
EEG_POOL = 75
EEG_POOL_STRIDE = 30         # larger stride -> fewer, coarser tokens
EMG_CONV_CHANNELS = (32, 64)
EMG_KERNELS = (11, 7)
EMG_STRIDES = (4, 4)
ATTN_HEADS = 4
ATTN_LAYERS = 2
ATTN_FF = 128
HUDGINS_ZC_THRESH = 1e-5
EPS = 1e-6
# modality dropout: bias toward dropping the EASY modality (EMG) more often, so
# the joint head is forced to extract usable EEG features instead of riding EMG.
P_DROP_EMG = 0.30
P_DROP_EEG = 0.10


def _device(dev=None):
    return dev or ("cuda" if torch.cuda.is_available() else "cpu")


def hudgins_td(x: torch.Tensor, zc_thresh: float = HUDGINS_ZC_THRESH) -> torch.Tensor:
    """[MAV, WL, ZC, SSC, RMS] per channel.  x:(B,C,T) -> (B,C*5).  Optional EMG
    side-features; with good data their marginal value is small (see docstring)."""
    mav = x.abs().mean(dim=2)
    wl = x.diff(dim=2).abs().mean(dim=2)
    s = x[..., 1:] * x[..., :-1]
    big = (x[..., 1:] - x[..., :-1]).abs() > zc_thresh
    zc = ((s < 0) & big).float().mean(dim=2)
    d = x.diff(dim=2)
    ssc = ((d[..., 1:] * d[..., :-1] < 0) & (d[..., 1:].abs() > zc_thresh)).float().mean(dim=2)
    rms = x.pow(2).mean(dim=2).clamp_min(EPS).sqrt()
    return torch.cat([mav, wl, zc, ssc, rms], dim=1)


def sinusoidal_pe(n, d, device):
    """Standard sinusoidal positional encoding, (n, d). Added per modality so the
    attention path retains temporal order without a fixed-size learned table."""
    pos = torch.arange(n, device=device).unsqueeze(1).float()
    i = torch.arange(d, device=device).unsqueeze(0).float()
    angle = pos / torch.pow(10000, (2 * (i // 2)) / d)
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


# =============================================================================
#  EEG encoder — ShallowConv tokenizer (-> token sequence), optional BiLSTM
# =============================================================================
class ShallowConvTokenizer(nn.Module):
    """ShallowConvNet front-end, but the time axis is KEPT: output is a sequence
    of (B, N, D_MODEL) tokens, one per pooled time window. A pooled vector is the
    mean over tokens (used by the gate path). Optional BiLSTM over the tokens
    adds a temporal model before fusion."""

    def __init__(self, n_channels, n_times, d_model=D_MODEL, use_bilstm=False,
                 f_temporal=EEG_F_TEMPORAL, k_time=EEG_KERNEL_TIME,
                 pool=EEG_POOL, pool_stride=EEG_POOL_STRIDE, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, f_temporal, (1, k_time), padding=(0, k_time // 2),
                                  bias=False)
        self.spatial = nn.Conv2d(f_temporal, f_temporal, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(f_temporal)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, pool_stride))
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(f_temporal, d_model)
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)

    def forward(self, x):                          # x:(B,C,T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.bn(self.spatial(self.temporal(x)))
        z = self.pool(z.pow(2)).clamp_min(EPS).log()      # (B,F,1,N)
        z = self.drop(z).squeeze(2).transpose(1, 2)       # (B,N,F)
        z = self.proj(z)                                  # (B,N,D)
        if self.use_bilstm:
            z, _ = self.lstm(z)
        return z


# =============================================================================
#  EMG encoder — 1D conv tokenizer (-> token sequence), optional Hudgins token
# =============================================================================
class EMGConvTokenizer(nn.Module):
    def __init__(self, n_channels, n_times, d_model=D_MODEL, use_hudgins=True,
                 conv_channels=EMG_CONV_CHANNELS, kernels=EMG_KERNELS,
                 strides=EMG_STRIDES, dropout=0.5):
        super().__init__()
        self.use_hudgins = use_hudgins
        layers, c_in = [], n_channels
        for c_out, k, s in zip(conv_channels, kernels, strides):
            layers += [nn.Conv1d(c_in, c_out, k, stride=s, padding=k // 2, bias=False),
                       nn.BatchNorm1d(c_out), nn.ELU(), nn.Dropout(dropout)]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(c_in, d_model)
        if use_hudgins:
            self.hud_proj = nn.Linear(n_channels * 5, d_model)

    def forward(self, x):                          # x:(B,C,T)
        z = self.conv(x).transpose(1, 2)           # (B,M,C_out)
        z = self.proj(z)                           # (B,M,D)
        if self.use_hudgins:
            h = self.hud_proj(hudgins_td(x)).unsqueeze(1)   # (B,1,D) side-token
            z = torch.cat([h, z], dim=1)                    # prepend
        return z


# =============================================================================
#  Fusion blocks
# =============================================================================
class GatedFusion(nn.Module):
    """Pool each modality to one vector, then a learned (input-dependent) gate
    mixes them. Returns (fused_vec, v_eeg, v_emg) — the latter two feed the aux
    heads. Modality dropout is applied by the parent net by zeroing a pooled
    vector before this call."""

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())

    def forward(self, v_eeg, v_emg):
        g = self.gate(torch.cat([v_eeg, v_emg], dim=1))
        return g * v_eeg + (1 - g) * v_emg, g


class AttentionFusion(nn.Module):
    """Masked JOINT self-attention over [CLS] + EEG tokens + EMG tokens, with
    learned modality-type embeddings and sinusoidal positional encodings per
    modality. Modality dropout = masking a modality's tokens via key_padding_mask
    (CLS and the survivor still attend). CLS readout is the fused vector."""

    def __init__(self, d_model=D_MODEL, heads=ATTN_HEADS, layers=ATTN_LAYERS,
                 ff=ATTN_FF, dropout=0.5):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.type_emb = nn.Embedding(3, d_model)   # 0=CLS, 1=EEG, 2=EMG
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.d_model = d_model

    def forward(self, eeg_tok, emg_tok, keep_eeg, keep_emg):
        B, dev = eeg_tok.shape[0], eeg_tok.device
        Ne, Nm = eeg_tok.shape[1], emg_tok.shape[1]
        # add positional + type embeddings
        eeg = eeg_tok + sinusoidal_pe(Ne, self.d_model, dev) + self.type_emb.weight[1]
        emg = emg_tok + sinusoidal_pe(Nm, self.d_model, dev) + self.type_emb.weight[2]
        cls = self.cls.expand(B, 1, -1) + self.type_emb.weight[0]
        seq = torch.cat([cls, eeg, emg], dim=1)              # (B, 1+Ne+Nm, D)
        # key_padding_mask: True == ignore. CLS never masked; mask a modality's
        # tokens for the samples where that modality was dropped.
        mask = torch.zeros(B, 1 + Ne + Nm, dtype=torch.bool, device=dev)
        mask[:, 1:1 + Ne] = ~keep_eeg.view(B, 1)
        mask[:, 1 + Ne:] = ~keep_emg.view(B, 1)
        out = self.encoder(seq, src_key_padding_mask=mask)
        return out[:, 0]                                     # CLS token (fused)


# =============================================================================
#  Full feature-fusion network
# =============================================================================
class FeatureFusionNet(nn.Module):
    """Encoders + swappable fusion + joint head + two per-modality aux heads.
    forward returns a dict of logits plus the modality-keep masks so the loss can
    skip a dropped modality's aux term for those samples."""

    def __init__(self, n_ch_eeg, n_times_eeg, n_ch_emg, n_times_emg, n_classes,
                 fusion="gate", d_model=D_MODEL, use_bilstm=False, use_hudgins=True,
                 p_drop_eeg=P_DROP_EEG, p_drop_emg=P_DROP_EMG):
        super().__init__()
        self.fusion_mode = fusion
        self.p_drop_eeg, self.p_drop_emg = p_drop_eeg, p_drop_emg
        self.eeg_enc = ShallowConvTokenizer(n_ch_eeg, n_times_eeg, d_model, use_bilstm)
        self.emg_enc = EMGConvTokenizer(n_ch_emg, n_times_emg, d_model, use_hudgins)
        if fusion == "gate":
            self.fuse = GatedFusion(d_model)
        elif fusion == "attention":
            self.fuse = AttentionFusion(d_model)
        else:
            raise ValueError("fusion must be 'gate' or 'attention'")
        self.head_joint = nn.Linear(d_model, n_classes)
        self.head_eeg = nn.Linear(d_model, n_classes)   # per-modality readout
        self.head_emg = nn.Linear(d_model, n_classes)

    def _moddrop(self, B, dev, training):
        """Per-sample keep masks. Never drop both; bias toward dropping EMG."""
        if not training:
            return (torch.ones(B, dtype=torch.bool, device=dev),
                    torch.ones(B, dtype=torch.bool, device=dev))
        ke = torch.rand(B, device=dev) >= self.p_drop_eeg
        km = torch.rand(B, device=dev) >= self.p_drop_emg
        both_gone = ~ke & ~km                         # restore EEG if both dropped
        ke = ke | both_gone
        return ke, km

    def forward(self, x_eeg, x_emg):
        eeg_tok = self.eeg_enc(x_eeg)                 # (B,Ne,D)
        emg_tok = self.emg_enc(x_emg)                 # (B,Nm,D)
        B, dev = eeg_tok.shape[0], eeg_tok.device
        ke, km = self._moddrop(B, dev, self.training)
        # per-modality pooled vectors (mean over tokens) for the aux heads
        v_eeg = eeg_tok.mean(dim=1)
        v_emg = emg_tok.mean(dim=1)

        if self.fusion_mode == "gate":
            fused, _ = self.fuse(v_eeg * ke.view(B, 1), v_emg * km.view(B, 1))
        else:
            fused = self.fuse(eeg_tok, emg_tok, ke, km)

        return {
            "joint": self.head_joint(fused),
            "eeg": self.head_eeg(v_eeg),
            "emg": self.head_emg(v_emg),
            "keep_eeg": ke, "keep_emg": km,
        }


# =============================================================================
#  val.py-compatible wrapper + train.py-style harness
# =============================================================================
class FeatureFusionEEGEMGNet(nn.Module):
    """Constructor ``(n_eeg, n_emg, n_classes, T)`` and ``forward(eeg, emg)`` match
    ``IntermediateFusionEEGNet`` so val.py can rebuild and load the checkpoint after
    a one-line import swap. eeg/emg arrive (B, 1, C, T) from FusionDataset; the
    encoders want (B, C, T), so the singleton axis is squeezed. Returns JOINT-head
    logits, shape (B, n_classes)."""

    def __init__(self, n_eeg: int, n_emg: int, n_classes: int, T: int):
        super().__init__()
        self.net = FeatureFusionNet(
            n_ch_eeg=n_eeg, n_times_eeg=T,
            n_ch_emg=n_emg, n_times_emg=T,
            n_classes=n_classes,
            fusion=FUSION_MODE, use_bilstm=USE_BILSTM, use_hudgins=USE_HUDGINS,
        )

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        return self.net(eeg.squeeze(1), emg.squeeze(1))["joint"]


def construct_model(splits):
    """Mirror ``train.construct_model`` but build the feature-fusion wrapper."""
    label_to_idx = base.build_label_map(splits.dataset, splits.train.indices)
    n_classes = len(label_to_idx)
    eeg0, emg0, _ = base.FusionDataset(splits.dataset, splits.train.indices, label_to_idx)[0]
    n_eeg, T = eeg0.shape[1], eeg0.shape[2]
    n_emg = emg0.shape[1]
    model = FeatureFusionEEGEMGNet(n_eeg, n_emg, n_classes, T).to(base.device)
    return model, label_to_idx, {"n_eeg": n_eeg, "n_emg": n_emg, "n_classes": n_classes, "T": T}


if __name__ == "__main__":
    base.seed_everything(base.SEED)
    splits = base.load_dataset_splits(SPLITS_DIR)
    print(f"device: {base.device} | fusion={FUSION_MODE}")
    model, label_to_idx, model_config = construct_model(splits)
    print(f"window T={model_config['T']} samples, classes={model_config['n_classes']}")
    model_config["arch"] = ARCH
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base.CHECKPOINT_DIR / f"{stamp}_{ARCH}_run_{uuid.uuid4().hex[:8]}"
    model, history, run_dir = base.train(model, splits, label_to_idx, model_config)
    print(f"artifacts saved under {run_dir}")

# To evaluate with val.py, replace its `from train import (...)` block with:
#     from train import (FusionDataset, default_checkpoint_path, get_device, seed_everything)
#     from train_feature_fusion import FeatureFusionEEGEMGNet as IntermediateFusionEEGNet
