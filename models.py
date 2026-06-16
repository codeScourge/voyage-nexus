"""
models.py
=========
A common, scikit-learn-style interface over every classifier in the project so
that hpo.py and observability.py treat them identically:

    est = make_model(name, n_channels, n_times, n_classes, **hp)
    est.fit(X, y, groups=...)          # X: (n, C, T) float32
    est.predict(X) / predict_proba(X)

Models
------
  "eegnet"      -> EEGNetFixed (Lawhern 2018) with the THREE fixes vs your file:
                     (1) max_norm on the depthwise spatial conv (=1.0) and the
                         dense layer (=0.25)  -- the regularizer your version
                         dropped, which matters most in the small-data regime;
                     (2) dynamic flatten size from a dummy forward (no T//32 bug);
                     (3) channels taken from the data, not hard-coded 64.
  "deepconvnet" -> braindecode Deep4Net (Schirrmeister 2017), the data-hungry net.
  "dda"         -> DDAClassifier from dda.py (no Torch).
  "cspdnn"      -> Panachakel CSP+DWT+DNN; see csp_patch.py for the leakage fix.

Torch is imported lazily; "dda"/"cspdnn" work without it.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit


# =============================================================================
#  Torch-dependent definitions (lazy)
# =============================================================================
def _torch():
    import torch
    return torch


def build_eegnet(n_channels, n_times, n_classes,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
    import torch
    import torch.nn as nn

    class _MaxNormConv2d(nn.Conv2d):
        """Conv2d with a max-norm constraint applied to its weight (Lawhern uses
        max_norm=1 on the depthwise spatial filter)."""
        def __init__(self, *a, max_norm=1.0, **k):
            super().__init__(*a, **k)
            self.max_norm = max_norm
        def forward(self, x):
            with torch.no_grad():
                norm = torch.linalg.vector_norm(
                    self.weight, dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
                desired = norm.clamp(max=self.max_norm)
                self.weight.mul_(desired / norm)
            return super().forward(x)

    class _MaxNormLinear(nn.Linear):
        def __init__(self, *a, max_norm=0.25, **k):
            super().__init__(*a, **k)
            self.max_norm = max_norm
        def forward(self, x):
            with torch.no_grad():
                norm = torch.linalg.vector_norm(
                    self.weight, dim=1, keepdim=True).clamp_min(1e-8)
                self.weight.mul_(norm.clamp(max=self.max_norm) / norm)
            return super().forward(x)

    class EEGNetFixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = _MaxNormConv2d(F1, F1 * D, (n_channels, 1),
                                            groups=F1, bias=False, max_norm=1.0)
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)
            self.sep_dw = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same",
                                    groups=F1 * D, bias=False)
            self.sep_pw = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
            self.bn3 = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)
            self.act = nn.ELU()
            # dynamic flatten size -- FIX for the T//32 hard-coding
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                flat = self._features(dummy).flatten(1).shape[1]
            self.fc = _MaxNormLinear(flat, n_classes, max_norm=0.25)

        def _features(self, x):
            x = self.bn1(self.conv1(x))
            x = self.drop1(self.pool1(self.act(self.bn2(self.depthwise(x)))))
            x = self.sep_pw(self.sep_dw(x))
            x = self.drop2(self.pool2(self.act(self.bn3(x))))
            return x

        def forward(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(1)                # (B,1,C,T)
            return self.fc(self._features(x).flatten(1))

    return EEGNetFixed()


def build_deepconvnet(n_channels, n_times, n_classes, drop_prob=0.5):
    """Schirrmeister Deep ConvNet via braindecode. Falls back to a faithful
    local re-implementation only if braindecode is unavailable."""
    try:
        from braindecode.models import Deep4Net
        net = Deep4Net(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto",
            drop_prob=drop_prob,
        )
        return net
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "braindecode is required for deepconvnet "
            "(`pip install braindecode`). Original error: %r" % e
        )


# =============================================================================
#  Unified sklearn-style wrapper for the Torch nets
# =============================================================================
class TorchClassifier(BaseEstimator, ClassifierMixin):
    """fit/predict wrapper with internal early-stopping on a *held-out slice of
    the training data only* (never the test fold), class weighting, and AdamW."""

    def __init__(self, builder, n_channels, n_times, n_classes,
                 lr=1e-3, weight_decay=1e-3, batch_size=32, max_epochs=200,
                 patience=20, val_frac=0.2, class_weight=True, seed=42,
                 builder_kwargs=None, device=None):
        self.builder = builder
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.class_weight = class_weight
        self.seed = seed
        self.builder_kwargs = builder_kwargs or {}
        self.device = device

    def _dev(self):
        torch = _torch()
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y, groups=None):
        torch = _torch()
        import torch.nn as nn
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, np.float32); y = np.asarray(y)
        self.classes_ = np.unique(y)
        dev = self._dev()

        # inner train/val split (group-aware if groups given) -- NO test leakage
        if groups is not None:
            tr, va = next(GroupShuffleSplit(1, test_size=self.val_frac,
                                            random_state=self.seed).split(X, y, groups))
        else:
            tr, va = next(StratifiedShuffleSplit(1, test_size=self.val_frac,
                                                 random_state=self.seed).split(X, y))
        self.model_ = self.builder(self.n_channels, self.n_times, self.n_classes,
                                   **self.builder_kwargs).to(dev)

        cw = None
        if self.class_weight:
            counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
            cw = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))),
                              dtype=torch.float32, device=dev)
        crit = nn.CrossEntropyLoss(weight=cw)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(Xt[tr], yt[tr])
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        Xva = Xt[va].to(dev); yva = yt[va].to(dev)

        best, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.model_.train()
            for xb, yb in dl:
                opt.zero_grad()
                crit(self.model_(xb.to(dev)), yb.to(dev)).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                vloss = crit(self.model_(Xva), yva).item()
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _logits(self, X):
        torch = _torch()
        self.model_.eval()
        dev = self._dev()
        with torch.no_grad():
            return self.model_(torch.tensor(np.asarray(X, np.float32)).to(dev)).cpu().numpy()

    def predict_proba(self, X):
        z = self._logits(X)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# =============================================================================
#  Factory
# =============================================================================
class OvO3D(BaseEstimator, ClassifierMixin):
    """One-vs-one wrapper that tolerates 3-D (n, C, T) inputs and forwards
    `groups` to each binary fit -- sklearn's OneVsOneClassifier does neither."""
    def __init__(self, base):
        self.base = base
    def fit(self, X, y, groups=None):
        from sklearn.base import clone
        X = np.asarray(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.pairs_ = []
        import itertools
        for a, b in itertools.combinations(self.classes_, 2):
            m = (y == a) | (y == b)
            est = clone(self.base)
            try:
                est.fit(X[m], y[m], groups=None if groups is None else groups[m])
            except TypeError:
                est.fit(X[m], y[m])
            self.pairs_.append((a, b, est))
        return self
    def predict(self, X):
        X = np.asarray(X)
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        cls_to_col = {c: i for i, c in enumerate(self.classes_)}
        for a, b, est in self.pairs_:
            pred = est.predict(X)
            for k, p in enumerate(pred):
                votes[k, cls_to_col[p]] += 1
        return self.classes_[votes.argmax(1)]


def make_model(name, n_channels, n_times, n_classes, **hp):
    name = name.lower()
    if name == "dda":
        from dda import DDAClassifier
        return DDAClassifier(**hp)
    if name == "cspdnn":
        from csp_patch import CSPDWTDNN
        base = CSPDWTDNN(n_classes=2, **hp)
        return OvO3D(base) if n_classes > 2 else base
    if name == "rusnac":
        from grigore_rusnac import RusnacCNN
        return RusnacCNN(n_classes=n_classes, **hp)
    if name == "eegnet":
        net_hp = {k: hp.pop(k) for k in
                  ["F1", "D", "F2", "kernel_length", "dropout"] if k in hp}
        return TorchClassifier(build_eegnet, n_channels, n_times, n_classes,
                               builder_kwargs=net_hp, **hp)
    if name == "deepconvnet":
        net_hp = {k: hp.pop(k) for k in ["drop_prob"] if k in hp}
        return TorchClassifier(build_deepconvnet, n_channels, n_times, n_classes,
                               builder_kwargs=net_hp, **hp)
    raise ValueError(f"unknown model {name!r}")


MODEL_NAMES = ["eegnet", "deepconvnet", "dda", "cspdnn", "rusnac"]
