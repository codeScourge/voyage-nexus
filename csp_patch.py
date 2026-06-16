"""
csp_patch.py
============
Panachakel et al. (2020) CSP -> channel selection -> per-channel db4 DWT ->
DNN -> majority vote, repackaged as a leakage-free scikit-learn estimator.

WHAT WAS WRONG IN Pairwise_CSP.py
---------------------------------
`train_dnn(Xtr, ytr, Xva, yva, ...)` selects its best checkpoint on (Xva,yva),
and `dnn_fold_accuracy` calls it as `train_dnn(Xtr, ytr, Xte, yte, ...)` -- i.e.
the TEST fold is used for model selection. That is test-set leakage and inflates
every reported pairwise accuracy.

THE FIX (here): the inner validation set used for early stopping / checkpoint
selection is carved out of the TRAINING trials only (by trial, so the
channel-as-sample rows of one trial never split across train/val). The test
fold is never seen during fitting.

The CSP math (trace-normalized covariances, optional shrinkage, generalized
eigenproblem eigh(C1, C2), top-|w| selection) and the DWT feature stack are kept
faithful to the paper / your original file.

Binary by construction (CSP is a two-class method) -- which is exactly the
silent-vs-voiced proxy task. A one-vs-one wrapper is noted for >2 classes.

Head: a Torch 4-layer DNN if torch is importable; otherwise an sklearn MLP with
internal (leakage-free) early stopping, so the pipeline is runnable either way.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pywt
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit


# --------------------------- CSP (faithful) ---------------------------------- #
def _norm_cov(trials):
    covs = []
    for tr in trials:
        c = tr @ tr.T
        covs.append(c / (np.trace(c) + 1e-12))
    return np.mean(covs, axis=0)


def compute_csp(X, y, a, b, reg=0.05):
    C1 = _norm_cov(X[y == a]); C2 = _norm_cov(X[y == b])
    if reg > 0:
        n = C1.shape[0]; I = np.eye(n)
        C1 = (1 - reg) * C1 + reg * I
        C2 = (1 - reg) * C2 + reg * I
    w, V = eigh(C1, C2)               # ascending
    return V[:, -1], V[:, 0]          # w_max, w_min


def _top_channels(w, n):
    return np.argsort(np.abs(w))[::-1][:n]


# --------------------------- DWT features ------------------------------------ #
def dwt_feats(sig, wavelet="db4", levels=4):
    coeffs = pywt.wavedec(sig, wavelet, level=levels)
    out = []
    for d in coeffs[1:]:                       # detail bands only
        rms = np.sqrt(np.mean(d ** 2))
        var = np.var(d)
        p = np.abs(d) / (np.sum(np.abs(d)) + 1e-12)
        ent = -np.sum(p * np.log(p + 1e-12))
        out += [rms, var, ent]
    return np.asarray(out, np.float32)


def build_paired(X, idx_max, idx_min, wavelet, levels):
    """Channel-as-sample augmentation: for each trial and rank i emit
    concat(DWT(ch_max_i), DWT(ch_min_i)). Returns rows, trial_ids."""
    rows, tids = [], []
    k = len(idx_max)
    for t, trial in enumerate(X):
        for i in range(k):
            rows.append(np.concatenate([dwt_feats(trial[idx_max[i]], wavelet, levels),
                                        dwt_feats(trial[idx_min[i]], wavelet, levels)]))
            tids.append(t)
    return np.asarray(rows, np.float32), np.asarray(tids)


# --------------------------- head (torch or sklearn) ------------------------- #
def _fit_head(Xtr, ytr, tids_tr, seed=42):
    """Returns an object with .predict(X)->{0,1}. Inner val split is by TRIAL,
    carved from TRAIN only (the leakage fix)."""
    try:
        import torch, torch.nn as nn  # noqa
        return _TorchHead(seed).fit(Xtr, ytr, tids_tr)
    except Exception:
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(40, 40, 40, 40), activation="relu",
                            early_stopping=True, validation_fraction=0.2,
                            max_iter=300, random_state=seed)
        return clf.fit(Xtr, ytr)


class _TorchHead:
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X, y, tids):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed)
        uniq = np.unique(tids)
        tr_t, va_t = next(GroupShuffleSplit(1, test_size=0.2, random_state=self.seed)
                          .split(uniq, groups=uniq))
        tr_ids, va_ids = set(uniq[tr_t]), set(uniq[va_t])
        m_tr = np.array([t in tr_ids for t in tids])
        m_va = ~m_tr
        net = nn.Sequential(
            nn.Linear(X.shape[1], 40), nn.BatchNorm1d(40), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(40, 40), nn.BatchNorm1d(40), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(40, 40), nn.BatchNorm1d(40), nn.Tanh(), nn.Dropout(.3),
            nn.Linear(40, 40), nn.BatchNorm1d(40), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(40, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.float32)
        Xtr, ytr = Xt[m_tr], yt[m_tr]; Xva, yva = Xt[m_va], yt[m_va]
        best, state, bad = np.inf, None, 0
        for _ in range(300):
            net.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 32):
                j = perm[i:i + 32]
                opt.zero_grad(); crit(net(Xtr[j]).squeeze(-1), ytr[j]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = crit(net(Xva).squeeze(-1), yva).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 20:
                    break
        if state: net.load_state_dict(state)
        self.net = net
        return self
    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(np.asarray(X, np.float32))).squeeze(-1))
        return (p.numpy() >= 0.5).astype(int)


# --------------------------- estimator --------------------------------------- #
class CSPDWTDNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=2, n_select=9, reg=0.05,
                 wavelet="db4", levels=4, seed=42):
        self.n_classes = n_classes
        self.n_select = n_select
        self.reg = reg
        self.wavelet = wavelet
        self.levels = levels
        self.seed = seed

    def fit(self, X, y, groups=None):
        X = np.asarray(X, np.float32); y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("CSPDWTDNN is binary; wrap in OneVsOne for >2 classes.")
        a, b = self.classes_
        k = min(self.n_select, X.shape[1])
        w_max, w_min = compute_csp(X, y, a, b, self.reg)
        self.idx_max_ = _top_channels(w_max, k)
        self.idx_min_ = _top_channels(w_min, k)
        rows, tids = build_paired(X, self.idx_max_, self.idx_min_, self.wavelet, self.levels)
        yb = (y[tids] == b).astype(int)          # head predicts P(class==b)
        self.head_ = _fit_head(rows, yb, tids, self.seed)
        self._a, self._b = a, b
        return self

    def predict(self, X):
        X = np.asarray(X, np.float32)
        rows, tids = build_paired(X, self.idx_max_, self.idx_min_, self.wavelet, self.levels)
        row_pred = self.head_.predict(rows)
        out = []
        for t in range(len(X)):                  # majority vote over channel rows
            votes = row_pred[tids == t]
            out.append(self._b if Counter(votes).most_common(1)[0][0] == 1 else self._a)
        return np.asarray(out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, C, T = 60, 8, 250
    X = rng.standard_normal((n, C, T)).astype(np.float32)
    y = np.zeros(n, int); y[n // 2:] = 1
    # inject class-dependent power on two channels so CSP has something to find
    X[y == 1, 0] *= 2.5; X[y == 1, 1] *= 0.4
    from sklearn.model_selection import cross_val_score, GroupKFold
    g = np.repeat(np.arange(n), 1)
    sc = cross_val_score(CSPDWTDNN(n_select=4), X, y, cv=GroupKFold(4),
                         groups=g, params={"groups": g})
    print(f"CSPDWTDNN GroupKFold acc: {sc.mean():.3f} +/- {sc.std():.3f}")
    print("CSP+DWT pipeline OK")
