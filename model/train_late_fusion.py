"""
train_late_fusion.py
====================
Row-3 decision-level (late) fusion, run inside the SAME harness as ``train.py`` —
identical data splits, training loop, checkpoint format, and ``val.py`` evaluation.

SELF-CONTAINED: the two branch nets (ShallowConvNet for EEG, EMGHybridNet for EMG)
and the Hudgins features are inlined below (copied verbatim from the Row-3 design,
kernels unchanged). This file does not depend on row3_late_fusion.py. Its only project
dependency is train.py (and what it already imports).

Run
---
    python data.py               # build ./splits   (unchanged)
    python train_late_fusion.py  # -> ./checkpoints/<run>/best.pt
    # then evaluate with val.py after the one-line import swap (see note at bottom)

Training note
-------------
The original Row-3 estimator trains the branches INDEPENDENTLY, temperature-
calibrates each, and stacks their OUT-OF-FOLD probabilities with a sklearn MLP —
which has no single torch state_dict and cannot be loaded by val.py. Here the two
branch architectures are kept EXACTLY and the decision-level constraint is kept
(only the branches' LOGITS cross the boundary, never embeddings), but they are
fused with a small linear head trained END-TO-END under train.py's CrossEntropyLoss.
Dropped: branch independence, temperature calibration, out-of-fold stacking.
Preserved: both branch architectures/kernels and the logits-only fusion boundary.
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
ARCH = "late"

# --- architecture choice (MUST match between training here and val.py load) ---
EMG_USE_HUDGINS = True


# =============================================================================
#  Inlined Row-3 branch code (verbatim; kernel sizes unchanged)
# =============================================================================
SEED = 42
EEG_F_TEMPORAL = 40          # ShallowConvNet temporal filters
EEG_KERNEL_TIME = 25         # temporal kernel (~half a 50 Hz cycle at 1 kHz->decim)
EEG_POOL = 75                # mean-pool width (Schirrmeister uses 75)
EEG_POOL_STRIDE = 15
EMG_CONV_CHANNELS = (32, 64) # 1D CNN width per stage on the EMG branch
EMG_KERNELS = (11, 7)
EMG_STRIDES = (4, 2)
HUDGINS_ZC_THRESH = 1e-5     # deadzone so baseline noise doesn't inflate ZC/SSC
STACK_HIDDEN = (32,)         # the "shallow MLP" meta-learner
DEFAULT_OOF_FOLDS = 5
EPS = 1e-6


def _device(dev=None):
    return dev or ("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
#  Hudgins time-domain features  (also useful standalone in your data pipeline)
# =============================================================================
def hudgins_td(x: torch.Tensor, zc_thresh: float = HUDGINS_ZC_THRESH) -> torch.Tensor:
    """Classic 4 Hudgins features + RMS, per channel.

    x : (B, C, T)  ->  (B, C*5)  in the order [MAV, WL, ZC, SSC, RMS] per channel.

    ZC and SSC use a deadzone threshold so that low-amplitude baseline noise does
    not manufacture spurious crossings (the standard fix vs naive sign-change
    counts). These are *features*, not a layer to learn through — sign() has zero
    a.e. gradient, which is fine; they act as a fixed, deterministic transform.
    """
    mav = x.abs().mean(dim=2)                                   # (B,C)
    wl = x.diff(dim=2).abs().mean(dim=2)                        # waveform length
    # zero crossings: sign flips between adjacent samples, gated by amplitude
    s = x[..., 1:] * x[..., :-1]
    big = (x[..., 1:] - x[..., :-1]).abs() > zc_thresh
    zc = ((s < 0) & big).float().mean(dim=2)
    # slope sign changes: sign flips of the first difference
    d = x.diff(dim=2)
    ssc = ((d[..., 1:] * d[..., :-1] < 0) &
           (d[..., 1:].abs() > zc_thresh)).float().mean(dim=2)
    rms = x.pow(2).mean(dim=2).clamp_min(EPS).sqrt()
    return torch.cat([mav, wl, zc, ssc, rms], dim=1)            # (B, C*5)


# =============================================================================
#  EEG branch — ShallowConvNet (logits)
# =============================================================================
class ShallowConvNet(nn.Module):
    """Schirrmeister shallow net: temporal conv -> spatial conv -> square ->
    mean-pool -> log -> dropout -> linear. The square/log is the band-power
    prior. Flatten size is computed by a dummy forward (no hard-coded T//k bug)."""

    def __init__(self, n_channels, n_times, n_classes,
                 f_temporal=EEG_F_TEMPORAL, k_time=EEG_KERNEL_TIME,
                 pool=EEG_POOL, pool_stride=EEG_POOL_STRIDE, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, f_temporal, (1, k_time), padding=(0, k_time // 2),
                                  bias=False)
        self.spatial = nn.Conv2d(f_temporal, f_temporal, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(f_temporal)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, pool_stride))
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            flat = self._features(torch.zeros(1, 1, n_channels, n_times)).flatten(1).shape[1]
        self.fc = nn.Linear(flat, n_classes)

    def _features(self, x):                       # x: (B,1,C,T)
        x = self.bn(self.spatial(self.temporal(x)))
        x = x.pow(2)                              # square  -> variance
        x = self.pool(x)                          # mean-pool over time
        x = x.clamp_min(EPS).log()                # log band-power
        return self.drop(x)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)                     # (B,1,C,T)
        return self.fc(self._features(x).flatten(1))


# =============================================================================
#  EMG branch — hybrid 1D temporal CNN  ++  Hudgins features (logits)
# =============================================================================
class EMGHybridNet(nn.Module):
    """1D temporal CNN over raw EMG, optionally concatenated with the Hudgins
    time-domain feature vector before the classifier head."""

    def __init__(self, n_channels, n_times, n_classes,
                 conv_channels=EMG_CONV_CHANNELS, kernels=EMG_KERNELS,
                 strides=EMG_STRIDES, dropout=0.5, use_hudgins=True):
        super().__init__()
        self.use_hudgins = use_hudgins
        layers, c_in = [], n_channels
        for c_out, k, s in zip(conv_channels, kernels, strides):
            layers += [nn.Conv1d(c_in, c_out, k, stride=s, padding=k // 2, bias=False),
                       nn.BatchNorm1d(c_out), nn.ELU(), nn.Dropout(dropout)]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            conv_flat = self.conv(torch.zeros(1, n_channels, n_times)).mean(dim=2).shape[1]
        feat_dim = conv_flat + (n_channels * 5 if use_hudgins else 0)
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):                          # x: (B,C,T)
        z = self.conv(x).mean(dim=2)               # global average pool over time
        if self.use_hudgins:
            z = torch.cat([z, hudgins_td(x)], dim=1)
        return self.fc(z)


# =============================================================================
#  val.py-compatible wrapper + train.py-style harness
# =============================================================================
class LateFusionEEGEMGNet(nn.Module):
    """Decision-level fusion of the two branches. Constructor
    ``(n_eeg, n_emg, n_classes, T)`` and ``forward(eeg, emg)`` match
    ``IntermediateFusionEEGNet``. Each branch emits class logits; ONLY those logits
    (never embeddings) are concatenated and passed through a linear fusion head, so
    the Row-3 probabilities-only boundary is respected. eeg/emg arrive (B, 1, C, T)
    from FusionDataset; the branches want (B, C, T), so the singleton is squeezed."""

    def __init__(self, n_eeg: int, n_emg: int, n_classes: int, T: int):
        super().__init__()
        self.eeg = ShallowConvNet(n_eeg, T, n_classes)
        self.emg = EMGHybridNet(n_emg, T, n_classes, use_hudgins=EMG_USE_HUDGINS)
        self.head = nn.Linear(2 * n_classes, n_classes)   # decision-level: logits in, logits out

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        logits_eeg = self.eeg(eeg.squeeze(1))   # (B, n_classes)
        logits_emg = self.emg(emg.squeeze(1))   # (B, n_classes)
        return self.head(torch.cat([logits_eeg, logits_emg], dim=1))


def construct_model(splits):
    """Mirror ``train.construct_model`` but build the late-fusion wrapper."""
    label_to_idx = base.build_label_map(splits.dataset, splits.train.indices)
    n_classes = len(label_to_idx)
    eeg0, emg0, _ = base.FusionDataset(splits.dataset, splits.train.indices, label_to_idx)[0]
    n_eeg, T = eeg0.shape[1], eeg0.shape[2]
    n_emg = emg0.shape[1]
    model = LateFusionEEGEMGNet(n_eeg, n_emg, n_classes, T).to(base.device)
    return model, label_to_idx, {"n_eeg": n_eeg, "n_emg": n_emg, "n_classes": n_classes, "T": T}


if __name__ == "__main__":
    base.seed_everything(base.SEED)
    splits = base.load_dataset_splits(SPLITS_DIR)
    print(f"device: {base.device}")
    model, label_to_idx, model_config = construct_model(splits)
    print(f"window T={model_config['T']} samples, classes={model_config['n_classes']}")
    model_config["arch"] = ARCH
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base.CHECKPOINT_DIR / f"{stamp}_{ARCH}_run_{uuid.uuid4().hex[:8]}"
    model, history, run_dir = base.train(model, splits, label_to_idx, model_config)
    print(f"artifacts saved under {run_dir}")

# To evaluate with val.py, replace its `from train import (...)` block with:
#     from train import (FusionDataset, default_checkpoint_path, get_device, seed_everything)
#     from train_late_fusion import LateFusionEEGEMGNet as IntermediateFusionEEGNet
