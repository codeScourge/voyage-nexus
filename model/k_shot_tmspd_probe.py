#!/usr/bin/env python3
"""
tmspd_probe.py -- T-MSPD architecture probe, single- or multi-subject.

Takes the SHIPPED IntermediateFusionEEGNet (train.py) and reports, for a chosen
set of subjects and the N most-distinct Mandarin words, how it decodes in each
speech mode (overt / silent / imagined), with and without EA.

PROTOCOL SWITCHES ON SUBJECT COUNT
----------------------------------
  --subjects 1        -> stratified k-fold WITHIN that subject. One session, no
                         block id, so this is the only available CV. It is
                         OPTIMISTIC (within-session) and EA is DEGENERATE here
                         (one domain = one fixed whitener; expect ~0 delta).
  --subjects 1 2 ...  -> LEAVE-ONE-SUBJECT-OUT. group = subject. This is the
                         honest cross-subject number, and the run where EA is
                         actually tested (each subject whitened into a shared
                         space, transductively, label-free).

Reuses your code unchanged (numbers comparable to loso.py / loso_align.py):
  tmspd_loader.load_tmspd_subject | loso_align.ea_align / train_intermediate /
  predict | loso.compute_metrics / print_metrics / print_summary_table.

STRICT: both modalities forced to 1 kHz (the model crops to min(T); EEG@250 +
EMG@1000 silently drops ~75% of the EMG). Asserts T_eeg == T_emg.

Usage
-----
  python tmspd_probe.py --root /data/T-MSPD --subjects 1     --words 3 --mode all
  python tmspd_probe.py --root /data/T-MSPD --subjects 1 2   --words 3 --mode all
  python tmspd_probe.py --root /data/T-MSPD --subjects 1 2 3 --words 3 --mode all
  # EA is OFF by default. To test it, opt in per branch (compared vs baseline):
  #   the Voyage-replicating run is EMG-EA under global EMG norm:
  python tmspd_probe.py --root /data/T-MSPD --subjects 2 3 4 --ea emg --emg-norm global
  #   compare several at once:
  python tmspd_probe.py --root /data/T-MSPD --subjects 2 3 4 --ea eeg,emg,both --emg-norm global
  # fix the words a priori (1-based trigger codes) instead of auto-picking:
  python tmspd_probe.py --root /data/T-MSPD --subjects 1 2 --word-codes 2,5,9
  # quick wiring/GPU check:
  python tmspd_probe.py --root /data/T-MSPD --subjects 1 --words 3 --epochs 2
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

import train
from tmspd_loader import load_tmspd_subject
from loso import compute_metrics, print_metrics, print_summary_table
from loso_align import ea_align, predict

MODES = ["overt speech", "silent speech", "imagined speech"]
DEVICE = train.get_device()
CACHE = ".cache_tmspd_probe"
CACHE_VER = "v2"          # bump to invalidate poisoned (pre-sanitize) caches


def sanitize(X):
    """Scrub non-finite + finite-extreme values. The known S01 outlier has inf in
    the EEG and a corrupt silent-EMG cast; left alone, one bad sample poisons a
    whole window (the loader's per-trial std overflows -> nan) and later crashes
    EA's eigendecomposition. Scrubbing to 0 collapses the AFFECTED CHANNELS toward
    zero instead, leaving the good channels intact. Returns (X_clean, n_nonfinite).
    """
    nbad = int((~np.isfinite(X)).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -50.0, 50.0)        # data is per-trial-normed (~unit); backstop
    return X.astype(np.float32), nbad


# ----------------------------------------------------------------------------
# loading: per-subject cache holds ALL 10 classes (so --words is a free filter)
# ----------------------------------------------------------------------------
def subject_paths(root, mode, sid):
    s = f"S{sid:02d}"
    return (os.path.join(root, mode, s, "EEG", f"{s}.cdt"),
            os.path.join(root, mode, s, "sEMG", "data.bdf"), s)


def _chtag(ch):
    return ch if isinstance(ch, str) else "-".join(ch)


def load_one_subject(root, mode, sid, *, eeg_channels, emg_norm, emg_offset_s):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(f"{CACHE_VER}|{os.path.abspath(root)}|{mode}|{sid}|{_chtag(eeg_channels)}|"
                      f"{emg_norm}|{emg_offset_s}".encode()).hexdigest()[:10]
    cf = os.path.join(CACHE, f"s{sid:02d}_{mode.replace(' ', '_')}_{_chtag(eeg_channels)}_{CACHE_VER}_{key}.npz")
    if os.path.exists(cf):
        z = np.load(cf)
        return z["Xe"], z["Xm"], z["y"]
    eeg, emg, s = subject_paths(root, mode, sid)
    if not (os.path.exists(eeg) and os.path.exists(emg)):
        raise FileNotFoundError(f"{s} [{mode}]: missing {eeg} or {emg}")
    print(f"  [load] {s} [{mode}] (windowing once -> {cf})")
    Xe, Xm, y = load_tmspd_subject(
        eeg, emg, eeg_channels=eeg_channels,
        eeg_fs_out=1000, emg_fs_out=1000,          # <-- the rate trap, closed
        emg_offset_s=emg_offset_s,
        norm_eeg="pertrial", norm_emg=emg_norm, verify=False)
    assert Xe.shape[2] == Xm.shape[2], f"T_eeg={Xe.shape[2]} != T_emg={Xm.shape[2]}"
    Xe, nbe = sanitize(Xe)
    Xm, nbm = sanitize(Xm)
    if nbe or nbm:
        print(f"  [sanitize] {s} [{mode}]: scrubbed non-finite EEG={nbe} EMG={nbm} "
              f"-> affected channels zeroed (known S01 outlier; that modality's "
              f"number is on DEGRADED data)")
    y = y.astype(np.int64)
    np.savez_compressed(cf, Xe=Xe, Xm=Xm, y=y)
    return Xe, Xm, y


def load_subjects(root, mode, sids, **kw):
    Xe_l, Xm_l, y_l, g_l = [], [], [], []
    for sid in sids:
        Xe, Xm, y = load_one_subject(root, mode, sid, **kw)
        Xe_l.append(Xe); Xm_l.append(Xm); y_l.append(y)
        g_l.append(np.full(len(y), sid, dtype=int))
    return (np.concatenate(Xe_l), np.concatenate(Xm_l),
            np.concatenate(y_l), np.concatenate(g_l))


# ----------------------------------------------------------------------------
# word selection
# ----------------------------------------------------------------------------
def pick_distinct(Xe, Xm, y, n):
    """Greedily pick the n classes whose rough signal templates are mutually most
    separable (mean evoked EEG + mean rectified EMG, z-scored across classes).
    Mildly OPTIMISTIC -- it sees all included data. Fix --word-codes to quote."""
    codes = np.unique(y)
    feats = []
    for c in codes:
        e = Xe[y == c].mean(0).ravel()
        m = np.abs(Xm[y == c]).mean(0).ravel()
        feats.append(np.concatenate([e, m]))
    F = np.stack(feats).astype(np.float64)
    F = (F - F.mean(0)) / (F.std(0) + 1e-8)
    D = np.linalg.norm(F[:, None] - F[None, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(D)), D.shape)
    chosen = [i, j]
    while len(chosen) < n:
        rest = [k for k in range(len(codes)) if k not in chosen]
        chosen.append(max(rest, key=lambda k: min(D[k, c] for c in chosen)))
    return sorted(int(codes[c]) for c in chosen)


def filter_words(Xe, Xm, y, groups, keep):
    m = np.isin(y, keep)
    remap = {c: i for i, c in enumerate(sorted(keep))}
    yk = np.array([remap[int(v)] for v in y[m]], dtype=np.int64)
    return Xe[m], Xm[m], yk, groups[m]


# ----------------------------------------------------------------------------
# EA + evaluation
# ----------------------------------------------------------------------------
def apply_ea(Xe, Xm, modality, groups):
    if modality == "none":
        return Xe, Xm
    Xe2 = ea_align(Xe, groups) if modality in ("eeg", "both") else Xe
    Xm2 = ea_align(Xm, groups) if modality in ("emg", "both") else Xm
    return Xe2.astype(np.float32), Xm2.astype(np.float32)


def train_intermediate_logged(Xe_t, Xm_t, y_t, train_idx, n_eeg, n_emg, n_classes,
                              T, y_all, device, epochs, bs, lr, seed, *,
                              tag="", log_every=1):
    """Numerically IDENTICAL to loso_align.train_intermediate (same seed, class
    weights, Adam, batching, drop_last, RNG draws) but logs per-epoch train_loss
    and train_acc. Accuracy reuses each training forward, so no extra RNG draw is
    consumed and the trained weights match the un-instrumented version exactly."""
    train.seed_everything(seed)
    tr = torch.as_tensor(train_idx)
    Xe_tr = Xe_t[tr].to(device); Xm_tr = Xm_t[tr].to(device); y_tr = y_t[tr].to(device)
    model = train.IntermediateFusionEEGNet(
        n_eeg=n_eeg, n_emg=n_emg, n_classes=n_classes, T=T, p_drop=0.5).to(device)
    counts = np.bincount(y_all[train_idx], minlength=n_classes).astype(np.float64)
    safe = np.where(counts > 0, counts, 1.0)
    w = torch.tensor(counts.sum() / (safe * n_classes), dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xe_tr.shape[0]; drop_last = n > bs
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        run_loss = correct = seen = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            if drop_last and idx.numel() < bs:
                continue
            opt.zero_grad()
            out = model(Xe_tr[idx], Xm_tr[idx])
            loss = crit(out, y_tr[idx])
            loss.backward(); opt.step()
            b = idx.numel()
            run_loss += float(loss.item()) * b
            correct += int((out.argmax(1) == y_tr[idx]).sum().item())
            seen += b
        if log_every and ((ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1):
            print(f"      {tag} ep {ep+1:3d}/{epochs}  train_loss={run_loss/max(seen,1):.4f}  "
                  f"train_acc={correct/max(seen,1):.4f}", flush=True)
    del Xe_tr, Xm_tr, y_tr
    return model


class SubjectConditioned(nn.Module):
    """Wrap a base IntermediateFusionEEGNet: strip its final classifier, concat a
    learned subject embedding to the fused feature vector, and re-classify. Tests
    whether telling the model 'which subject this is' lets it exploit subject-specific
    features instead of averaging conflicting ones away. The benefit is ONLY
    collectable when the test subject was SEEN in training (protocol=seen); under
    LOSO the held-out subject's id is out-of-vocabulary -> null embedding."""

    def __init__(self, base, n_subjects, d_embed=16):
        super().__init__()
        self.feat_dim = base.classifier.in_features
        self.n_classes = base.classifier.out_features
        base.classifier = nn.Identity()
        self.base = base
        self.null = n_subjects                       # extra index = unseen / dropout token
        self.embed = nn.Embedding(n_subjects + 1, d_embed)
        self.head = nn.Linear(self.feat_dim + d_embed, self.n_classes)

    def forward(self, eeg, emg, sidx):
        f = self.base(eeg, emg)                      # (B, feat_dim) -- classifier is Identity
        return self.head(torch.cat([f, self.embed(sidx)], dim=1))

    def forward_with_embedding(self, eeg, emg, e):
        """Predict using an arbitrary embedding vector e (d,) or (B,d) -- used to
        inject mean/null/soft-weighted embeddings for an unseen subject."""
        f = self.base(eeg, emg)
        if e.dim() == 1:
            e = e.unsqueeze(0).expand(f.shape[0], -1)
        return self.head(torch.cat([f, e], dim=1))


def _fit_predict_cond(Xe, Xm, y, sidx, train_idx, test_idx, n_classes, T, n_subjects,
                      epochs, bs, lr, seed, *, tag="", log_every=1, subj_dropout=0.1):
    """Subject-conditioned twin of _fit_predict. Builds the base trunk under the same
    seed/init (so the trunk start matches the unconditioned baseline exactly), wraps it
    with a subject embedding + new head, trains with subject-id dropout (so the null
    token is trained and unseen subjects degrade gracefully), then predicts."""
    Xe_t = torch.from_numpy(Xe).unsqueeze(1).float()
    Xm_t = torch.from_numpy(Xm).unsqueeze(1).float()
    y_t = torch.from_numpy(y).long()
    s_t = torch.from_numpy(sidx).long()
    train.seed_everything(seed)
    base = train.IntermediateFusionEEGNet(n_eeg=Xe.shape[1], n_emg=Xm.shape[1],
                                          n_classes=n_classes, T=T, p_drop=0.5)
    model = SubjectConditioned(base, n_subjects=n_subjects).to(DEVICE)
    tr = torch.as_tensor(train_idx)
    Xe_tr = Xe_t[tr].to(DEVICE); Xm_tr = Xm_t[tr].to(DEVICE)
    y_tr = y_t[tr].to(DEVICE); s_tr = s_t[tr].to(DEVICE)
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float64)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes),
                     dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xe_tr.shape[0]; drop_last = n > bs; null = model.null
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        run = correct = seen = 0.0
        for s0 in range(0, n, bs):
            idx = perm[s0:s0 + bs]
            if drop_last and idx.numel() < bs:
                continue
            sb = s_tr[idx].clone()
            if subj_dropout > 0:
                sb[torch.rand(sb.shape, device=DEVICE) < subj_dropout] = null
            opt.zero_grad()
            out = model(Xe_tr[idx], Xm_tr[idx], sb)
            loss = crit(out, y_tr[idx]); loss.backward(); opt.step()
            b = idx.numel(); run += float(loss.item()) * b
            correct += int((out.argmax(1) == y_tr[idx]).sum().item()); seen += b
        if log_every and ((ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1):
            print(f"      {tag} ep {ep+1:3d}/{epochs}  train_loss={run/max(seen,1):.4f}  "
                  f"train_acc={correct/max(seen,1):.4f}", flush=True)
    model.eval(); te = torch.as_tensor(test_idx); preds = []
    with torch.no_grad():
        for s0 in range(0, len(te), 256):
            j = te[s0:s0 + 256]
            out = model(Xe_t[j].to(DEVICE), Xm_t[j].to(DEVICE), s_t[j].to(DEVICE))
            preds.append(out.argmax(1).cpu().numpy())
    del model, Xe_tr, Xm_tr, y_tr, s_tr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(preds)


def _fit_predict(Xe, Xm, y, train_idx, test_idx, n_classes, T, epochs, bs, lr, seed,
                 *, tag="", log_every=1):
    Xe_t = torch.from_numpy(Xe).unsqueeze(1).float()
    Xm_t = torch.from_numpy(Xm).unsqueeze(1).float()
    y_t = torch.from_numpy(y).long()
    model = train_intermediate_logged(Xe_t, Xm_t, y_t, train_idx,
                                      Xe.shape[1], Xm.shape[1], n_classes, T, y, DEVICE,
                                      epochs, bs, lr, seed, tag=tag, log_every=log_every)
    yp = predict(model, Xe_t, Xm_t, test_idx, DEVICE)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return yp


def evaluate(Xe, Xm, y, groups, *, n_classes, folds, epochs, bs, lr, seed, name,
             log_every=1, protocol="loso", cond=False):
    """Pooled OOF predictions. protocol='loso' = leave-one-subject-out (>=2 subj) or
    within-subject k-fold (1 subj). protocol='seen' = stratified k-fold over POOLED
    trials so every test subject is also in training (the enrolled-user regime where
    subject conditioning is meaningful). cond=True builds the subject-conditioned
    model and feeds subject ids."""
    T = Xe.shape[2]
    uniq = np.unique(groups)
    if protocol == "seen":
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(y)), y))
        scheme = f"SEEN-subjects stratified {folds}-fold (test subjects also in train)"
    elif len(uniq) >= 2:
        splits = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in uniq]
        scheme = f"leave-one-subject-out ({len(uniq)} subjects)"
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(skf.split(np.zeros(len(y)), y))
        scheme = f"stratified {folds}-fold WITHIN one subject (optimistic)"

    oof_yt, oof_yp, fold_bal = [], [], []
    for fi, (tr, te) in enumerate(splits):
        held = np.unique(groups[te])
        gtag = f" heldS{int(held[0]):02d}" if len(held) == 1 else f" {len(held)}subj"
        tag = f"[{name} f{fi+1}/{len(splits)}{gtag}]"
        print(f"    {tag} train_n={len(tr)} test_n={len(te)}", flush=True)
        if cond:
            train_subj = sorted(np.unique(groups[tr]).tolist())
            smap = {s: k for k, s in enumerate(train_subj)}
            null = len(smap)
            sidx = np.array([smap.get(int(s), null) for s in groups], dtype=np.int64)
            n_unseen = int(sum(int(s) not in smap for s in groups[te]))
            if n_unseen:
                print(f"    {tag} WARNING {n_unseen}/{len(te)} test samples are UNSEEN "
                      f"subjects -> null embedding (conditioning cannot help them)", flush=True)
            yp = _fit_predict_cond(Xe, Xm, y, sidx, tr, te, n_classes, T, null,
                                   epochs, bs, lr, seed, tag=tag, log_every=log_every)
        else:
            yp = _fit_predict(Xe, Xm, y, tr, te, n_classes, T, epochs, bs, lr, seed,
                              tag=tag, log_every=log_every)
        yt = y[te]
        b = compute_metrics(yt, yp, n_classes, "_")["balanced_accuracy"]
        print(f"    {tag} fold bal_acc={b:.4f}", flush=True)
        fold_bal.append(b); oof_yt.append(yt); oof_yp.append(yp)
    yt = np.concatenate(oof_yt); yp = np.concatenate(oof_yp)
    return compute_metrics(yt, yp, n_classes, name), np.array(fold_bal), scheme


def run_mode(Xe, Xm, y, groups, mode, keep, args):
    n_classes = len(keep)
    chance = 1.0 / n_classes
    n_sub = len(np.unique(groups))
    print(f"\n================ {mode}  (subjects {sorted(np.unique(groups).tolist())}) ================")
    print(f"Xe{Xe.shape} Xm{Xm.shape}  classes={n_classes}  "
          f"per-class n={np.bincount(y).tolist()}  chance={chance:.3f}")

    conditions = [("baseline", "none", False)] + [(f"ea[{m}]", m, False) for m in args.ea_list]
    if args.cond == "subject":
        conditions.append(("subj_cond", "none", True))

    pooled, perfold, scheme = [], {}, ""
    for name, modality, cond in conditions:
        Xe_c, Xm_c = apply_ea(Xe, Xm, modality, groups)
        m, fb, scheme = evaluate(Xe_c, Xm_c, y, groups, n_classes=n_classes,
                                 folds=args.folds, epochs=args.epochs,
                                 bs=args.batch_size, lr=args.lr, seed=args.seed,
                                 name=name, log_every=args.log_every,
                                 protocol=args.protocol, cond=cond)
        print(f"\n--- {name}  [{scheme}] ---")
        print_metrics(m, {i: f"w{c+1}" for i, c in enumerate(sorted(keep))})
        print(f"[{name}] per-fold bal_acc {fb.mean():.4f} +/- {fb.std():.4f}  "
              f"(folds={len(fb)})  chance={chance:.3f}")
        pooled.append(m); perfold[name] = fb

    print(f"\nscheme: {scheme}")
    print_summary_table(pooled, chance=chance)
    for cond_name in [c[0] for c in conditions if c[0] != "baseline"]:
        d = perfold[cond_name] - perfold["baseline"]
        print(f"\n[{mode}] {cond_name} delta vs baseline: mean={d.mean():+.4f} "
              f"median={np.median(d):+.4f} folds +{int((d>0).sum())}/-{int((d<0).sum())}")
        if cond_name == "subj_cond" and args.protocol != "seen":
            print("  (protocol=loso: test subject is UNSEEN, conditioning cannot help it. "
                  "Run --protocol seen to test the hypothesis.)")
        elif cond_name == "subj_cond":
            print("  (seen-subjects protocol: this is the enrolled-user regime where "
                  "subject conditioning is meaningful.)")
        elif n_sub < 2:
            print("  (ONE subject => ONE EA domain => degenerate; expect ~0.)")
        else:
            print(f"  (EA per-subject whitening, n={n_sub} folds; few subjects => high variance.)")
    return {n: perfold[n].mean() for n in perfold}, chance, scheme


# ============================================================================
# Q2/Q3 additions: covariance similarity, amortized conditioning, transfer
# ============================================================================
def spatial_cov(X, shrink=0.05):
    """Average spatial covariance over windows: (1/N) sum_i (X_i X_i^T / T), with
    Ledoit-Wolf-style shrinkage so tiny buffers stay well-conditioned for logm."""
    X = X.astype(np.float64)
    N, C, T = X.shape
    R = np.einsum("nct,ndt->cd", X, X) / (N * T)
    return (1 - shrink) * R + shrink * (np.trace(R) / C) * np.eye(C)


def logm_vech(S):
    """vech(log S) -- log-Euclidean tangent vector (lower-triangle incl. diagonal)."""
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    L = (V * np.log(w)) @ V.T
    return L[np.tril_indices(L.shape[0])]


def riemann_dist(Si, Sj):
    """Affine-invariant Riemannian distance: ||log(Si^-1/2 Sj Si^-1/2)||_F.
    Invariant to any shared linear transform (per-donning gain) -- the property
    that makes it the right cross-subject similarity."""
    wi, Vi = np.linalg.eigh(Si)
    wi = np.clip(wi, 1e-12, None)
    inv_sqrt = (Vi * (wi ** -0.5)) @ Vi.T
    w = np.linalg.eigvalsh(inv_sqrt @ Sj @ inv_sqrt)
    w = np.clip(w, 1e-12, None)
    return float(np.sqrt(np.sum(np.log(w) ** 2)))


def subj_covs(Xe, Xm, idx):
    return (spatial_cov(Xe[idx]), spatial_cov(Xm[idx]))      # (Sigma_eeg, Sigma_emg)


def dr_modality(a, b, modality):
    de = riemann_dist(a[0], b[0]); dm = riemann_dist(a[1], b[1])
    return {"eeg": de, "emg": dm, "both": de + dm}[modality]


def phi_modality(pair, modality):
    pe, pm = logm_vech(pair[0]), logm_vech(pair[1])
    return {"eeg": pe, "emg": pm, "both": np.concatenate([pe, pm])}[modality]


class AmortizedConditioned(nn.Module):
    """Subject embedding is a learned FUNCTION of the subject's covariance feature
    phi = vech(log Sigma): e = rho(phi). No lookup table, so seen and unseen subjects
    are handled identically -- there is no missing embedding to impute."""

    def __init__(self, base, phi_dim, d_embed=16, hidden=32):
        super().__init__()
        self.feat_dim = base.classifier.in_features
        self.n_classes = base.classifier.out_features
        base.classifier = nn.Identity()
        self.base = base
        self.rho = nn.Sequential(nn.Linear(phi_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d_embed))
        self.head = nn.Linear(self.feat_dim + d_embed, self.n_classes)

    def forward(self, eeg, emg, phi):
        f = self.base(eeg, emg)
        return self.head(torch.cat([f, self.rho(phi)], dim=1))


def _cond_train_loop(model, Xe_tr, Xm_tr, y_tr, side_tr, y_all_train, n_classes,
                     epochs, bs, lr, seed, tag, log_every, *, kind, null=None,
                     subj_dropout=0.1):
    """Shared conditioned training loop. kind='lookup' -> side_tr is subject index
    (with id-dropout to the null token); kind='amort' -> side_tr is the phi feature."""
    counts = np.bincount(y_all_train, minlength=n_classes).astype(np.float64)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes),
                     dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xe_tr.shape[0]; drop_last = n > bs
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        run = correct = seen = 0.0
        for s0 in range(0, n, bs):
            idx = perm[s0:s0 + bs]
            if drop_last and idx.numel() < bs:
                continue
            side = side_tr[idx]
            if kind == "lookup":
                side = side.clone()
                if subj_dropout > 0:
                    side[torch.rand(side.shape, device=DEVICE) < subj_dropout] = null
            opt.zero_grad()
            out = model(Xe_tr[idx], Xm_tr[idx], side)
            loss = crit(out, y_tr[idx]); loss.backward(); opt.step()
            b = idx.numel(); run += float(loss.item()) * b
            correct += int((out.argmax(1) == y_tr[idx]).sum().item()); seen += b
        if log_every and ((ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1):
            print(f"      {tag} ep {ep+1:3d}/{epochs}  train_loss={run/max(seen,1):.4f}  "
                  f"train_acc={correct/max(seen,1):.4f}", flush=True)
    return model


def fit_lookup_model(Xe_t, Xm_t, y, sidx, train_idx, n_classes, T, K,
                     epochs, bs, lr, seed, tag, log_every):
    y_t = torch.from_numpy(y).long(); s_t = torch.from_numpy(sidx).long()
    train.seed_everything(seed)
    base = train.IntermediateFusionEEGNet(n_eeg=Xe_t.shape[2], n_emg=Xm_t.shape[2],
                                          n_classes=n_classes, T=T, p_drop=0.5)
    model = SubjectConditioned(base, n_subjects=K).to(DEVICE)
    tr = torch.as_tensor(train_idx)
    return _cond_train_loop(model, Xe_t[tr].to(DEVICE), Xm_t[tr].to(DEVICE),
                            y_t[tr].to(DEVICE), s_t[tr].to(DEVICE), y[train_idx],
                            n_classes, epochs, bs, lr, seed, tag, log_every,
                            kind="lookup", null=K)


def fit_amortized_model(Xe_t, Xm_t, y, phi, train_idx, n_classes, T, phi_dim,
                        epochs, bs, lr, seed, tag, log_every):
    y_t = torch.from_numpy(y).long(); p_t = torch.from_numpy(phi).float()
    train.seed_everything(seed)
    base = train.IntermediateFusionEEGNet(n_eeg=Xe_t.shape[2], n_emg=Xm_t.shape[2],
                                          n_classes=n_classes, T=T, p_drop=0.5)
    model = AmortizedConditioned(base, phi_dim).to(DEVICE)
    tr = torch.as_tensor(train_idx)
    return _cond_train_loop(model, Xe_t[tr].to(DEVICE), Xm_t[tr].to(DEVICE),
                            y_t[tr].to(DEVICE), p_t[tr].to(DEVICE), y[train_idx],
                            n_classes, epochs, bs, lr, seed, tag, log_every, kind="amort")


def _score_fixed(model, Xe_t, Xm_t, test_idx, side_vec, y, n_classes, *, lookup):
    model.eval(); te = torch.as_tensor(test_idx); preds = []
    sv = side_vec.to(DEVICE) if torch.is_tensor(side_vec) else \
        torch.from_numpy(side_vec).float().to(DEVICE)
    with torch.no_grad():
        for s0 in range(0, len(te), 256):
            j = te[s0:s0 + 256]; B = len(j)
            eeg, emg = Xe_t[j].to(DEVICE), Xm_t[j].to(DEVICE)
            if lookup:
                out = model.forward_with_embedding(eeg, emg, sv)
            else:
                out = model(eeg, emg, sv.unsqueeze(0).expand(B, -1))
            preds.append(out.argmax(1).cpu().numpy())
    yp = np.concatenate(preds)
    return compute_metrics(y[test_idx], yp, n_classes, "_")["balanced_accuracy"]


def run_unseen_embed(Xe, Xm, y, groups, keep, args):
    """LOSO over subjects. Per held-out subject, train (a) a lookup-embedding model
    and (b) an amortized model, then score the held-out subject under each strategy
    for populating its (unseen) embedding: mean / null / soft-weighted(buffer k) /
    amortized(buffer k). Covariances are PRE-EA."""
    n_classes = len(keep); T = Xe.shape[2]; chance = 1.0 / n_classes
    uniq = sorted(np.unique(groups).tolist())
    buffers = [int(b) for b in args.buffers.split(",")]
    Xe_t = torch.from_numpy(Xe).unsqueeze(1).float()
    Xm_t = torch.from_numpy(Xm).unsqueeze(1).float()
    strategies = ["mean", "null"] + [f"soft_k{b}" for b in buffers] + [f"amort_k{b}" for b in buffers]
    res = {s: [] for s in strategies}
    print(f"\n==== UNSEEN-EMBED  (LOSO over {uniq}, words {[c+1 for c in keep]}, "
          f"sim={args.sim_modality}, tau={args.tau}, chance={chance:.3f}) ====")
    for g in uniq:
        tr = np.where(groups != g)[0]; te = np.where(groups == g)[0]
        train_subj = sorted(np.unique(groups[tr]).tolist())
        smap = {s: k for k, s in enumerate(train_subj)}; K = len(smap)
        sidx = np.array([smap.get(int(s), K) for s in groups], dtype=np.int64)
        tag = f"[unseen heldS{g:02d}]"
        print(f"\n  {tag} train_subj={train_subj} train_n={len(tr)} test_n={len(te)}", flush=True)
        cov_tr = [subj_covs(Xe, Xm, np.where(groups == s)[0]) for s in train_subj]

        lk = fit_lookup_model(Xe_t, Xm_t, y, sidx, tr, n_classes, T, K,
                              args.epochs, args.batch_size, args.lr, args.seed,
                              tag + "[lookup]", args.log_every)
        E = lk.embed.weight.detach()
        b_mean = _score_fixed(lk, Xe_t, Xm_t, te, E[:K].mean(0), y, n_classes, lookup=True)
        b_null = _score_fixed(lk, Xe_t, Xm_t, te, E[K], y, n_classes, lookup=True)
        res["mean"].append(b_mean); res["null"].append(b_null)
        print(f"  {tag} mean={b_mean:.4f}  null={b_null:.4f}", flush=True)
        for b in buffers:
            cov_star = subj_covs(Xe, Xm, te[:b])
            d = np.array([dr_modality(cov_star, ck, args.sim_modality) for ck in cov_tr])
            wts = np.exp(-(d - d.min()) / args.tau); wts /= wts.sum()
            e_soft = (torch.from_numpy(wts).float().to(E.device)[:, None] * E[:K]).sum(0)
            bal = _score_fixed(lk, Xe_t, Xm_t, te, e_soft, y, n_classes, lookup=True)
            res[f"soft_k{b}"].append(bal)
            print(f"  {tag} soft_k{b:<3d}={bal:.4f}  (w_max={wts.max():.2f})", flush=True)

        phi_tr = {s: phi_modality(ck, args.sim_modality) for s, ck in zip(train_subj, cov_tr)}
        phidim = len(next(iter(phi_tr.values())))
        phi_ps = np.zeros((len(y), phidim), dtype=np.float32)
        for s in train_subj:
            phi_ps[groups == s] = phi_tr[s]
        am = fit_amortized_model(Xe_t, Xm_t, y, phi_ps, tr, n_classes, T, phidim,
                                 args.epochs, args.batch_size, args.lr, args.seed,
                                 tag + "[amort]", args.log_every)
        for b in buffers:
            phi_star = phi_modality(subj_covs(Xe, Xm, te[:b]), args.sim_modality).astype(np.float32)
            bal = _score_fixed(am, Xe_t, Xm_t, te, phi_star, y, n_classes, lookup=False)
            res[f"amort_k{b}"].append(bal)
            print(f"  {tag} amort_k{b:<3d}={bal:.4f}", flush=True)

    print(f"\n==== UNSEEN-EMBED SUMMARY (mean +/- std over {len(uniq)} held-out subjects) ====")
    print(f"chance={chance:.3f}")
    print(f"{'strategy':<14}{'bal_acc':>10}{'std':>9}")
    for s in strategies:
        v = np.array(res[s]); print(f"{s:<14}{v.mean():>10.4f}{v.std():>9.4f}")
    print("\nbuffer note: soft_k* / amort_k* use the FIRST k windows of the held-out "
          "subject as an UNLABELED enrollment buffer (transductive, no labels/grads). "
          "mean/null use NO test data.")
    return res


def double_center_offdiag(T):
    M = T.astype(float).copy(); np.fill_diagonal(M, np.nan)
    rm = np.nanmean(M, 1, keepdims=True); cm = np.nanmean(M, 0, keepdims=True)
    return M - rm - cm + np.nanmean(M)


def mantel(A, B, perms=10000, seed=0):
    """Mantel test between two symmetric matrices on the strict upper triangle."""
    n = A.shape[0]; iu = np.triu_indices(n, 1)
    a, b = A[iu], B[iu]
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    r_obs = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(perms):
        p = rng.permutation(n)
        bp = B[np.ix_(p, p)][iu][ok]
        if abs(np.corrcoef(a, bp)[0, 1]) >= abs(r_obs):
            cnt += 1
    return r_obs, (cnt + 1) / (perms + 1)


def run_transfer(Xe, Xm, y, groups, keep, args):
    """Single-source transfer matrix T(i->j) (train on subject i alone, test on j),
    averaged over seeds; PRE-EA Riemannian similarity per modality; then double-center
    + symmetrize T and Mantel-test it against each similarity matrix."""
    n_classes = len(keep); T = Xe.shape[2]
    uniq = sorted(np.unique(groups).tolist()); N = len(uniq)
    seeds = [int(s) for s in args.seeds.split(",")]
    Xe_t = torch.from_numpy(Xe).unsqueeze(1).float()
    Xm_t = torch.from_numpy(Xm).unsqueeze(1).float()
    y_t = torch.from_numpy(y).long()
    print(f"\n==== TRANSFER  (N={N} subjects {uniq}, words {[c+1 for c in keep]}, "
          f"seeds={seeds}) ====")
    Tacc = {}
    for i in uniq:
        tri = np.where(groups == i)[0]
        for seed in seeds:
            tag = f"[src S{i:02d} seed{seed}]"
            model = train_intermediate_logged(Xe_t, Xm_t, y_t, tri, Xe.shape[1],
                                              Xm.shape[1], n_classes, T, y, DEVICE,
                                              args.epochs, args.batch_size, args.lr, seed,
                                              tag=tag, log_every=args.log_every)
            for j in uniq:
                if i == j:
                    continue
                te = np.where(groups == j)[0]
                yp = predict(model, Xe_t, Xm_t, te, DEVICE)
                bal = compute_metrics(y[te], yp, n_classes, "_")["balanced_accuracy"]
                Tacc.setdefault((i, j), []).append(bal)
                print(f"  {tag} -> S{j:02d}  bal_acc={bal:.4f}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    idx = {s: k for k, s in enumerate(uniq)}
    Tmat = np.full((N, N), np.nan)
    for (i, j), v in Tacc.items():
        Tmat[idx[i], idx[j]] = np.mean(v)
    covs = [subj_covs(Xe, Xm, np.where(groups == s)[0]) for s in uniq]
    DR = {m: np.zeros((N, N)) for m in ("eeg", "emg", "both")}
    for a in range(N):
        for b in range(N):
            if a != b:
                for m in DR:
                    DR[m][a, b] = dr_modality(covs[a], covs[b], m)

    Tc = double_center_offdiag(Tmat); Tsym = 0.5 * (Tc + Tc.T)
    print("\n  transfer matrix T(row i -> col j) bal_acc (avg over seeds):")
    print("      " + "".join(f"S{s:02d}".rjust(8) for s in uniq))
    for a, s in enumerate(uniq):
        print(f"  S{s:02d} " + "".join((f"{Tmat[a,b]:.3f}".rjust(8) if not np.isnan(Tmat[a,b]) else "   -   ") for b in range(N)))
    print(f"\n  Mantel: double-centered symmetric transfer  vs  pre-EA Riemannian distance")
    print(f"  (expect NEGATIVE r: more transfer <-> smaller covariance distance)")
    for m in ("eeg", "emg", "both"):
        r, p = mantel(Tsym, DR[m], perms=args.perms, seed=0)
        from scipy.stats import spearmanr
        iu = np.triu_indices(N, 1)
        rho = spearmanr(Tsym[iu], DR[m][iu]).correlation
        print(f"    sim={m:<4}  Mantel r={r:+.3f}  p={p:.4f}   (Spearman rho={rho:+.3f})")
    print("\n  CAVEAT: one donning per subject in T-MSPD -> cross-subject similarity is "
          "confounded with within-subject donning variability; on the rig, compute "
          "Sigma_s from a HELD-OUT donning.")
    return Tmat, DR


def draw_cal(sub_idx, y, k, n_classes, rng):
    """Stratified draw of k labeled trials per class from a subject's indices.
    Returns (cal_idx, test_idx) disjoint; cal=[] when k==0."""
    if k == 0:
        return np.array([], dtype=int), sub_idx
    cal = []
    for c in range(n_classes):
        cls = sub_idx[y[sub_idx] == c].copy()
        rng.shuffle(cls)
        cal.extend(cls[:k].tolist())
    cal = np.array(sorted(cal), dtype=int)
    return cal, np.setdiff1d(sub_idx, cal)


def _features(model, Xe_t, Xm_t, idx, device):
    """Frozen trunk features z = base(eeg,emg) for the given indices (cached on CPU)."""
    model.eval(); te = torch.as_tensor(idx); zs = []
    with torch.no_grad():
        for s0 in range(0, len(te), 256):
            j = te[s0:s0 + 256]
            zs.append(model.base(Xe_t[j].to(device), Xm_t[j].to(device)).cpu())
    return torch.cat(zs, 0) if zs else torch.empty(0)


def fit_embedding(model, z_cal, y_cal, e_init, steps, lr, device, tag, log_every):
    """Gradient-descend ONE embedding vector e_new on k labeled trials; trunk & head
    frozen, features precomputed -> only the head sees e, so this is head-only on a
    16-d input. Returns the fitted embedding."""
    e = e_init.clone().detach().to(device).requires_grad_(True)
    opt = torch.optim.Adam([e], lr=lr)
    zc = z_cal.to(device); yc = y_cal.to(device)
    crit = nn.CrossEntropyLoss()
    for st in range(steps):
        opt.zero_grad()
        out = model.head(torch.cat([zc, e.unsqueeze(0).expand(zc.size(0), -1)], 1))
        loss = crit(out, yc); loss.backward(); opt.step()
        if log_every and ((st + 1) % max(1, steps // 5) == 0 or st == 0):
            print(f"        {tag} cal step {st+1:3d}/{steps}  loss={loss.item():.4f}", flush=True)
    return e.detach()


def score_with_e(model, z_test, e, y_test, n_classes, device):
    with torch.no_grad():
        out = model.head(torch.cat([z_test.to(device),
                                    e.unsqueeze(0).expand(z_test.size(0), -1).to(device)], 1))
        yp = out.argmax(1).cpu().numpy()
    return compute_metrics(y_test, yp, n_classes, "_")["balanced_accuracy"]


def run_kshot_embed(Xe, Xm, y, groups, keep, args):
    """LOSO. Per held-out subject: train the lookup-conditioned model on the others,
    then for each k fit a fresh embedding e_new on k labeled trials/class (trunk+head
    frozen) and score the remainder. k=0 = null token = the Q2 cold floor. Sweeps k."""
    n_classes = len(keep); T = Xe.shape[2]; chance = 1.0 / n_classes
    uniq = sorted(np.unique(groups).tolist())
    ks = [int(x) for x in args.kshots.split(",")]
    Xe_t = torch.from_numpy(Xe).unsqueeze(1).float()
    Xm_t = torch.from_numpy(Xm).unsqueeze(1).float()
    res = {k: [] for k in ks}
    print(f"\n==== K-SHOT EMBEDDING CALIBRATION (LOSO over {uniq}, words {[c+1 for c in keep]}, "
          f"chance={chance:.3f}, cal_draws={args.cal_draws}, cal_steps={args.cal_steps}) ====")
    print("  k=0 uses the trained null token (no labels) = the Q2 cold floor; k>0 fits e_new.")
    for g in uniq:
        tr = np.where(groups != g)[0]; te_all = np.where(groups == g)[0]
        train_subj = sorted(np.unique(groups[tr]).tolist())
        smap = {s: i for i, s in enumerate(train_subj)}; K = len(smap)
        sidx = np.array([smap.get(int(s), K) for s in groups], dtype=np.int64)
        tag = f"[kshot heldS{g:02d}]"
        print(f"\n  {tag} train_subj={train_subj} held_n={len(te_all)}", flush=True)
        lk = fit_lookup_model(Xe_t, Xm_t, y, sidx, tr, n_classes, T, K,
                              args.epochs, args.batch_size, args.lr, args.seed,
                              tag + "[train]", args.log_every)
        e_null = lk.embed.weight.detach()[K]
        z_all = _features(lk, Xe_t, Xm_t, te_all, DEVICE)
        pos = {int(idx): i for i, idx in enumerate(te_all)}
        for k in ks:
            draws = 1 if k == 0 else args.cal_draws
            accs = []
            for d in range(draws):
                rng = np.random.default_rng(args.seed * 1000 + d)
                cal, test = draw_cal(te_all, y, k, n_classes, rng)
                if k == 0:
                    e = e_null
                else:
                    zc = z_all[[pos[int(i)] for i in cal]]
                    yc = torch.from_numpy(y[cal]).long()
                    e = fit_embedding(lk, zc, yc, e_null, args.cal_steps, args.cal_lr,
                                      DEVICE, tag + f"[k{k}d{d}]", args.log_every)
                zt = z_all[[pos[int(i)] for i in test]]
                accs.append(score_with_e(lk, zt, e, y[test], n_classes, DEVICE))
            m = float(np.mean(accs)); res[k].append(m)
            print(f"  {tag} k={k:<2d}  bal_acc={m:.4f} +/- {np.std(accs):.4f}  "
                  f"(draws={draws}, test_n={len(test)})", flush=True)
        del lk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n==== K-SHOT SUMMARY (mean +/- std over {len(uniq)} held-out subjects) ====")
    print(f"chance={chance:.3f}")
    print(f"{'k (per class)':>14}{'bal_acc':>10}{'std':>9}{'vs k=0':>9}")
    base0 = np.mean(res[0]) if 0 in res else np.mean(res[ks[0]])
    for k in ks:
        v = np.array(res[k])
        print(f"{k:>14}{v.mean():>10.4f}{v.std():>9.4f}{v.mean()-base0:>+9.4f}")
    print("\nnote: test set shrinks by k*W as k grows (calibration trials are held out of "
          "scoring), so compare the TREND, not k against k on identical test sets. e_new is "
          "fit on cached frozen-trunk features -> only a 16-d vector is trained.")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="T-MSPD root: {mode}/S{nn}/EEG|sEMG/...")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1],
                    help="subject ids, e.g. --subjects 1   or   --subjects 1 2 3")
    ap.add_argument("--words", type=int, default=3, help="NUMBER of words (classes) to use")
    ap.add_argument("--word-codes", default=None,
                    help="override: comma 1-based trigger codes (e.g. 2,5,9)")
    ap.add_argument("--mode", default="all",
                    help='"all" or one of: "overt speech","silent speech","imagined speech"')
    ap.add_argument("--eeg-channels", default="periauric",
                    help='"periauric" (9ch, montage-honest) | "all" (64ch) | comma names')
    ap.add_argument("--emg-norm", choices=["pertrial", "global"], default="pertrial")
    ap.add_argument("--ea", default="",
                    help="EA conditions to ADD vs baseline: comma list of eeg,emg,both. "
                         "DEFAULT off (baseline only). NOTE: EMG-EA only reproduces the "
                         "Voyage regime under --emg-norm global (see warning at runtime).")
    ap.add_argument("--emg-offset-s", type=float, default=0.0)
    ap.add_argument("--protocol", choices=["loso", "seen"], default="loso",
                    help="loso = leave-one-subject-out (zero-shot cross-subject). "
                         "seen = stratified k-fold over pooled trials so test subjects "
                         "are also in train (enrolled-user regime; required to test --cond).")
    ap.add_argument("--cond", choices=["none", "subject"], default="none",
                    help="subject = add a subject-conditioned model (embedding at the head) "
                         "as an extra condition vs baseline. Only meaningful with --protocol seen.")
    ap.add_argument("--experiment", choices=["sweep", "unseen-embed", "transfer", "kshot-embed"],
                    default="sweep",
                    help="sweep = baseline/EA/cond. unseen-embed = Q2 (mean/null/soft/amort). "
                         "transfer = Q3 (transfer matrix + Mantel). kshot-embed = fit e_new from "
                         "k labeled trials/class of the held-out subject (the conditioning+k-shot bridge).")
    ap.add_argument("--kshots", default="0,1,2,3,4,5,6,7,8,9,10",
                    help="k-shot calibration sizes (labeled trials PER CLASS from held-out subject)")
    ap.add_argument("--cal-draws", type=int, default=3, help="random calibration draws to average (k>0)")
    ap.add_argument("--cal-steps", type=int, default=200, help="Adam steps to fit e_new")
    ap.add_argument("--cal-lr", type=float, default=1e-2, help="lr for the e_new fit")
    ap.add_argument("--buffers", default="1,2,5,10,20,50",
                    help="Q2 enrollment buffer sizes (# windows from start of held-out subject)")
    ap.add_argument("--tau", type=float, default=1.0, help="Q2 soft-weight softmax temperature")
    ap.add_argument("--sim-modality", choices=["eeg", "emg", "both"], default="both",
                    help="Q2/Q3 covariance modality for Riemannian similarity / amort features")
    ap.add_argument("--seeds", default="41,42,43,44,45", help="Q3 transfer seeds (>=5 advised)")
    ap.add_argument("--perms", type=int, default=10000, help="Q3 Mantel permutations")
    ap.add_argument("--folds", type=int, default=5, help="k for the 1-subject case")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=1,
                    help="log train_loss/train_acc every N epochs per fold (1=every epoch, "
                         "0=off). Endpoints (first/last epoch) always logged unless 0.")
    ap.add_argument("--print-words", action="store_true",
                    help="just print the chosen 1-based trigger codes (comma list) and exit; "
                         "use it to LOCK a word set per vocabulary size before a sweep")
    args = ap.parse_args()

    ea_list = [m.strip() for m in args.ea.split(",") if m.strip()]
    bad = [m for m in ea_list if m not in ("eeg", "emg", "both")]
    if bad:
        ap.error(f"--ea: unknown {bad}; choose from eeg,emg,both (comma-separated)")
    args.ea_list = ea_list

    ec = (args.eeg_channels if args.eeg_channels in ("all", "periauric")
          else args.eeg_channels.split(","))
    modes = MODES if args.mode == "all" else [args.mode]
    sids = sorted(set(args.subjects))
    print(f"device={DEVICE} | subjects={sids} | eeg_channels={args.eeg_channels} "
          f"| emg_norm={args.emg_norm} | ea={ea_list or 'off'} | protocol={args.protocol} "
          f"| cond={args.cond}")
    if ("emg" in ea_list or "both" in ea_list) and args.emg_norm != "global":
        print("WARNING: EA on the EMG branch with --emg-norm pertrial is NOT the regime "
              "where Voyage's EMG-EA gain appeared. pertrial z-scores each EMG channel "
              "per window, pre-removing the cross-channel amplitude ratios EA normalizes, "
              "so you may see a misleading null. Use --emg-norm global to replicate it.")
    if args.cond == "subject" and args.protocol != "seen":
        print("WARNING: --cond subject under protocol=loso. The held-out subject is "
              "out-of-vocabulary, so the subject embedding is a null token and conditioning "
              "CANNOT help (and may hurt). Use --protocol seen to test the hypothesis.")
    if args.protocol == "seen":
        print("NOTE: protocol=seen -> stratified k-fold over POOLED trials. Test subjects "
              "are SEEN in training (enrolled-user regime). These numbers are NOT comparable "
              "to the LOSO sweep -- they are an upper bound that assumes user-specific data.")
    elif len(sids) < 2:
        print("NOTE: 1 subject -> within-session k-fold (optimistic) and EA is degenerate.")
    else:
        print(f"NOTE: {len(sids)} subjects -> leave-one-subject-out (the honest protocol).")

    load_kw = dict(eeg_channels=ec, emg_norm=args.emg_norm, emg_offset_s=args.emg_offset_s)

    # ---- choose words ONCE (on overt, pooled over included subjects) ----
    if args.word_codes:
        keep = sorted(int(c) - 1 for c in args.word_codes.split(","))   # 1-based -> y-space
        print(f"fixed words: y-space {keep}, trigger codes {[c+1 for c in keep]}")
    else:
        sel = "overt speech" if "overt speech" in modes else modes[0]
        Xe0, Xm0, y0, _ = load_subjects(args.root, sel, sids, **load_kw)
        keep = pick_distinct(Xe0, Xm0, y0, args.words)
        print(f"[auto] {args.words} most-separable classes on '{sel}': y-space {keep}, "
              f"trigger codes {[c+1 for c in keep]}  (optimistic; use --word-codes to quote)")

    if args.print_words:
        # final stdout line = bare comma list, so bash can `tail -n1` it
        print(",".join(str(c + 1) for c in keep))
        return

    # ---- Q2 / Q3 experiments operate on ONE mode (overt = the only mode w/ signal) ----
    if args.experiment in ("unseen-embed", "transfer", "kshot-embed"):
        emode = "overt speech" if (args.mode == "all" or "overt" in args.mode) else args.mode
        print(f"[experiment={args.experiment}] using mode='{emode}'")
        Xe, Xm, y, g = load_subjects(args.root, emode, sids, **load_kw)
        Xe, Xm, y, g = filter_words(Xe, Xm, y, g, keep)
        if len(np.unique(g)) < 2:
            ap.error("need >=2 subjects for unseen-embed/transfer/kshot-embed")
        if args.experiment == "unseen-embed":
            run_unseen_embed(Xe, Xm, y, g, keep, args)
        elif args.experiment == "transfer":
            run_transfer(Xe, Xm, y, g, keep, args)
        else:
            run_kshot_embed(Xe, Xm, y, g, keep, args)
        return

    summary, chance = {}, None
    for mode in modes:
        try:
            Xe, Xm, y, g = load_subjects(args.root, mode, sids, **load_kw)
        except FileNotFoundError as e:
            print(f"\n[skip] {e}"); continue
        Xe, Xm, y, g = filter_words(Xe, Xm, y, g, keep)
        means, chance, scheme = run_mode(Xe, Xm, y, g, mode, keep, args)
        summary[mode] = means

    cols = (["baseline"] + [f"ea[{m}]" for m in args.ea_list]
            + (["subj_cond"] if args.cond == "subject" else []))
    print("\n================ cross-mode summary (subjects {}, words {}) ================"
          .format(sids, [c + 1 for c in keep]))
    print(f"{'mode':<18}{'chance':>8}" + "".join(f"{c:>12}" for c in cols))
    for mode in modes:
        if mode not in summary:
            continue
        s = summary[mode]
        print(f"{mode:<18}{chance:>8.3f}" + "".join(f"{s[c]:>12.4f}" for c in cols))
    print(f"\nprotocol: {'within-session (optimistic)' if len(sids)<2 else 'leave-one-subject-out'}")


if __name__ == "__main__":
    main()
