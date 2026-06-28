"""
row2_feature_fusion.py
======================
Row 2 of the EEG+EMG fusion taxonomy: **intermediate (feature-level) fusion**,
trained end-to-end. Dedicated per-modality encoders extract within-modality
features, then a fusion block lets the modalities interact, then a joint head
predicts. Two per-modality readout heads sit on the encoder outputs so each
branch stays individually discriminative and you can watch for EMG domination.

The fusion block is swappable behind one flag, because the right choice is a
function of the task, not of taste:

    fusion="gate"       pooled embedding + gated fusion.
        Each encoder is mean-pooled over time to ONE vector; a learned gate
        mixes them: fused = g*v_eeg + (1-g)*v_emg. Time is gone before fusion,
        so this CANNOT model cross-modal *timing* (EEG motor-prep leading EMG
        execution). It is cheap, data-efficient, robust to misalignment, and the
        gate value is a clean audit of each modality's contribution. Right
        default for closed-vocab classification.

    fusion="attention" token sequence + masked JOINT self-attention.
        Each encoder emits a SEQUENCE of tokens (time axis preserved). A CLS
        token plus both token streams (tagged with modality-type + positional
        encodings) go through joint self-attention; the CLS readout is the
        fused representation. This can learn time-resolved cross-modal structure
        — the corticomuscular lead-lag — but is hungrier and needs good temporal
        sync (you have it: one 1 kHz clock, sample_index axis). Right substrate
        for open-vocab / continuous decoding.

Two correctness points this file bakes in (both easy to get wrong):
  * Cross-attention over a SINGLE pooled vector per modality is degenerate
    (softmax over one key = identity). So the attention path keeps the token
    sequences; only the gate path pools. You cannot pool *and* meaningfully
    attend — hence one flag, two real code paths.
  * Modality dropout breaks *pure two-tower* cross-attention (drop a tower and
    its keys/values vanish). We therefore use JOINT self-attention with a
    key_padding_mask: dropping a modality masks its tokens while the CLS and the
    surviving modality still attend among themselves. A modality can always be
    decoded alone, which is the entire point of modality dropout.

Loss balancing: UncertaintyWeightedLoss (Kendall 2018) over the three CE terms
{joint, eeg-aux, emg-aux}. With the high-SNR EMG branch this stops the EMG loss
from swamping the shared gradient, and it is far more stable than GradNorm
(which needs a target-rate hyperparameter and per-step gradient-norm bookkeeping
at a shared layer). GradNorm is a documented drop-in alternative; uncertainty
weighting is the default for exactly the reasons in our design review. Note that
loss balancing is only meaningful BECAUSE the per-modality aux heads exist —
with the joint loss alone there is nothing to balance.

RA / alignment (EEG only, default OFF): same demoted status as Row 3 — a
label-free Euclidean-whitening *ablation* on the EEG inputs, not a load-bearing
assumption. The real cross-session medicine here is re-donned data diversity
(and, if you want a learned mechanism, a session-adversarial term — not RA).

Interface
---------
    clf = FeatureFusionClassifier(n_classes=K, fusion="attention")
    clf.fit(Xe, Xm, y, groups=ids)        # Xe:(n,Ce,T) Xm:(n,Cm,T)
    clf.predict(Xe, Xm) / clf.predict_proba(Xe, Xm)
    clf.aux_val_acc_      # {'eeg':..,'emg':..} per-modality readout on val — the audit
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it=None, **k):
        return it if it is not None else iter([])

# =============================================================================
#  CONSTANTS
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
                 p_drop_eeg=P_DROP_EEG, p_drop_emg=P_DROP_EMG,
                 eeg_encoder=None, emg_encoder=None):
        super().__init__()
        self.fusion_mode = fusion
        self.p_drop_eeg, self.p_drop_emg = p_drop_eeg, p_drop_emg
        # custom encoders (e.g. an SSL-pretrained channel-as-token encoder) may be
        # injected; they must emit (B, N, d_model) token sequences.
        self.eeg_enc = eeg_encoder or ShallowConvTokenizer(n_ch_eeg, n_times_eeg, d_model, use_bilstm)
        self.emg_enc = emg_encoder or EMGConvTokenizer(n_ch_emg, n_times_emg, d_model, use_hudgins)
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
#  Uncertainty-weighted multitask loss (Kendall 2018)
# =============================================================================
class UncertaintyWeightedLoss(nn.Module):
    """L = sum_t [ exp(-s_t) * CE_t + s_t ],  s_t = log variance (learned).
    Aux CE terms are computed only over samples where that modality was kept."""

    def __init__(self, class_weight=None):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(3))    # [joint, eeg, emg]
        self.register_buffer("cw", class_weight if class_weight is not None
                             else torch.tensor(0.0))
        self.has_cw = class_weight is not None

    def _ce(self, logits, y, keep=None):
        w = self.cw if self.has_cw else None
        if keep is None:
            return F.cross_entropy(logits, y, weight=w)
        if keep.sum() == 0:
            return logits.sum() * 0.0                   # nothing kept this batch
        return F.cross_entropy(logits[keep], y[keep], weight=w)

    def forward(self, out, y):
        ce = [self._ce(out["joint"], y),
              self._ce(out["eeg"], y, out["keep_eeg"]),
              self._ce(out["emg"], y, out["keep_emg"])]
        loss = sum(torch.exp(-self.log_var[t]) * ce[t] + self.log_var[t]
                   for t in range(3))
        return loss


# =============================================================================
#  Euclidean alignment toggle (EEG only, default off) — same demoted role
# =============================================================================
class EuclideanAlignment:
    def fit(self, X):
        covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[2]
        w, V = np.linalg.eigh(covs.mean(axis=0))
        self.W_ = V @ np.diag(1.0 / np.sqrt(np.maximum(w, EPS))) @ V.T
        return self

    def transform(self, X):
        return np.einsum("cd,ndt->nct", self.W_, X).astype(np.float32)


# =============================================================================
#  sklearn-style classifier
# =============================================================================
class FeatureFusionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes, *, fusion="gate", use_bilstm=False, use_hudgins=True,
                 align_eeg="none", p_drop_eeg=P_DROP_EEG, p_drop_emg=P_DROP_EMG,
                 lr=1e-3, weight_decay=1e-3, batch_size=32, max_epochs=200,
                 patience=40, val_frac=0.2, class_weight=True, seed=SEED, device=None,
                 verbose=True, eeg_pretrained=None, emg_pretrained=None,
                 freeze_encoders=False, eeg_encoder=None, emg_encoder=None):
        self.n_classes = n_classes
        self._k = n_classes
        self.fusion = fusion
        self.use_bilstm = use_bilstm
        self.use_hudgins = use_hudgins
        self.align_eeg = align_eeg
        self.p_drop_eeg = p_drop_eeg; self.p_drop_emg = p_drop_emg
        self.lr = lr; self.weight_decay = weight_decay
        self.batch_size = batch_size; self.max_epochs = max_epochs
        self.patience = patience; self.val_frac = val_frac
        self.class_weight = class_weight; self.seed = seed; self.device = device
        self.verbose = verbose
        self.eeg_pretrained = eeg_pretrained
        self.emg_pretrained = emg_pretrained
        self.freeze_encoders = freeze_encoders
        self.eeg_encoder = eeg_encoder
        self.emg_encoder = emg_encoder

    def _maybe_align(self, Xe, fit_on=None):
        if self.align_eeg == "none":
            return Xe
        if fit_on is not None:
            self._ea_ = EuclideanAlignment().fit(fit_on)
        return self._ea_.transform(Xe)

    def fit(self, Xe, Xm, y, groups=None, classes=None):
        dev = _device(self.device)
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        Xe = np.asarray(Xe, np.float32); Xm = np.asarray(Xm, np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y) if classes is None else np.asarray(classes)
        self._k = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)

        if groups is not None and len(np.unique(groups)) >= 2:
            tr, va = next(GroupShuffleSplit(1, test_size=self.val_frac,
                          random_state=self.seed).split(Xe, y_idx, groups))
        else:
            tr, va = next(StratifiedShuffleSplit(1, test_size=self.val_frac,
                          random_state=self.seed).split(Xe, y_idx))

        Xe_tr = self._maybe_align(Xe[tr], fit_on=Xe[tr])
        Xe_va = self._maybe_align(Xe[va])

        import copy
        # deep-copy injected encoders so each fit starts from the pristine
        # pretrained weights (CV folds must not share fine-tuned state)
        eeg_enc = copy.deepcopy(self.eeg_encoder) if self.eeg_encoder is not None else None
        emg_enc = copy.deepcopy(self.emg_encoder) if self.emg_encoder is not None else None
        self.net_ = FeatureFusionNet(
            Xe.shape[1], Xe.shape[2], Xm.shape[1], Xm.shape[2], self._k,
            fusion=self.fusion, use_bilstm=self.use_bilstm, use_hudgins=self.use_hudgins,
            p_drop_eeg=self.p_drop_eeg, p_drop_emg=self.p_drop_emg,
            eeg_encoder=eeg_enc, emg_encoder=emg_enc).to(dev)

        # ---- transfer pretrained weights into the DEFAULT encoders ---------
        if self.eeg_pretrained or self.emg_pretrained:
            from pretrain import transfer_load, freeze_module
            if self.eeg_pretrained:
                transfer_load(self.net_.eeg_enc, self.eeg_pretrained,
                              verbose=self.verbose, tag="EEG")
                if self.freeze_encoders:
                    freeze_module(self.net_.eeg_enc, True)
            if self.emg_pretrained:
                transfer_load(self.net_.emg_enc, self.emg_pretrained,
                              verbose=self.verbose, tag="EMG")
                if self.freeze_encoders:
                    freeze_module(self.net_.emg_enc, True)
        # ---- freeze INJECTED (already-loaded) encoders if requested --------
        if self.freeze_encoders:
            if self.eeg_encoder is not None:
                for p in self.net_.eeg_enc.parameters():
                    p.requires_grad = False
            if self.emg_encoder is not None:
                for p in self.net_.emg_enc.parameters():
                    p.requires_grad = False

        cw = None
        if self.class_weight:
            counts = np.bincount(y_idx, minlength=self._k).astype(np.float64)
            cw = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)),
                              dtype=torch.float32, device=dev)
        self.loss_ = UncertaintyWeightedLoss(class_weight=cw).to(dev)
        trainable = [p for p in self.net_.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable + list(self.loss_.parameters()),
                                lr=self.lr, weight_decay=self.weight_decay)

        def to_t(a): return torch.tensor(np.asarray(a, np.float32), device=dev)
        Xe_tr_t, Xm_tr_t = to_t(Xe_tr), to_t(Xm[tr])
        y_tr_t = torch.tensor(y_idx[tr], dtype=torch.long, device=dev)
        Xe_va_t, Xm_va_t = to_t(Xe_va), to_t(Xm[va])
        y_va_t = torch.tensor(y_idx[va], dtype=torch.long, device=dev)

        idx = np.arange(len(tr))
        best, best_state, bad = np.inf, None, 0
        pbar = tqdm(range(self.max_epochs), desc=f"R2 {self.fusion}",
                    disable=not self.verbose, leave=False)
        for _ in pbar:
            self.net_.train()
            np.random.shuffle(idx)
            for s in range(0, len(idx), self.batch_size):
                b = idx[s:s + self.batch_size]
                opt.zero_grad()
                out = self.net_(Xe_tr_t[b], Xm_tr_t[b])
                self.loss_(out, y_tr_t[b]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                out = self.net_(Xe_va_t, Xm_va_t)
                vloss = F.cross_entropy(out["joint"], y_va_t).item()
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose:
                pbar.set_postfix(val=f"{best:.3f}", patience=f"{bad}/{self.patience}")
        if best_state is not None:
            self.net_.load_state_dict(best_state)

        # per-modality audit on the val split (modality dropout off in eval)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(Xe_va_t, Xm_va_t)
        self.aux_val_acc_ = {
            "eeg": float((out["eeg"].argmax(1).cpu().numpy() == y_idx[va]).mean()),
            "emg": float((out["emg"].argmax(1).cpu().numpy() == y_idx[va]).mean()),
            "joint": float((out["joint"].argmax(1).cpu().numpy() == y_idx[va]).mean()),
        }
        self._dev = dev
        return self

    def predict_proba(self, Xe, Xm):
        Xe = self._maybe_align(np.asarray(Xe, np.float32))
        Xm = np.asarray(Xm, np.float32)
        self.net_.eval()
        with torch.no_grad():
            z = self.net_(torch.tensor(Xe, device=self._dev),
                          torch.tensor(Xm, device=self._dev))["joint"]
        return torch.softmax(z, dim=1).cpu().numpy()

    def predict(self, Xe, Xm):
        return self.classes_[self.predict_proba(Xe, Xm).argmax(1)]


# =============================================================================
#  Smoke test — exercises BOTH fusion modes, modality dropout, and the aux heads
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, Ce, Cm, T, K = 80, 16, 16, 512, 4
    g = np.repeat(np.arange(8), 10)
    y = rng.integers(0, K, n)
    Xe = rng.standard_normal((n, Ce, T)).astype(np.float32)
    Xm = rng.standard_normal((n, Cm, T)).astype(np.float32)
    for k in range(K):
        Xe[y == k, k % Ce] += 0.6 * np.sin(np.linspace(0, 6, T))
        Xm[y == k, k % Cm] += 0.6

    for fusion in ("gate", "attention"):
        for bilstm in (False, True) if fusion == "attention" else (False,):
            clf = FeatureFusionClassifier(
                n_classes=K, fusion=fusion, use_bilstm=bilstm,
                max_epochs=6, patience=4, batch_size=16)
            clf.fit(Xe, Xm, y, groups=g)
            p = clf.predict(Xe, Xm)
            tag = f"{fusion}{'+bilstm' if bilstm else ''}"
            print(f"[{tag:18s}] train acc={float((p==y).mean()):.3f} "
                  f"| aux val: eeg={clf.aux_val_acc_['eeg']:.2f} "
                  f"emg={clf.aux_val_acc_['emg']:.2f} joint={clf.aux_val_acc_['joint']:.2f} "
                  f"| log_var={clf.loss_.log_var.detach().cpu().numpy().round(2)}")
    print("SMOKE OK")
