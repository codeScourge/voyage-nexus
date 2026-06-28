#!/usr/bin/env python3
"""
loso_align.py — test-time alignment comparison for the Voyage rig (per-session).

Sits next to loso.py / data.py / train.py and reuses loso.py's window cache and
val.py-style metrics. Runs the SAME leave-one-session-out (LOSO) protocol as
loso.py on IntermediateFusionEEGNet, but compares FOUR conditions:

  1. baseline      no alignment (== loso.py intermediate)
  2. adabn         baseline weights, BN running-stats recomputed on the held-out
                   session at test time (label-free, transductive)
  3. ea            Euclidean Alignment as a preprocessing whitener: every session
                   (train AND test) is whitened by its OWN reference covariance
                   R^-1/2 so all donnings share a common space. The model is
                   RETRAINED on EA-aligned data; the test session is whitened by
                   its own (label-free) windows.
  4. adabn+ea      EA-aligned model + BN recompute on the EA-aligned test session.

Cost note: this trains TWO models per fold (one on raw, one on EA-aligned), so it
is ~2x a normal loso intermediate run. AdaBN adds only forward passes.

What this can and cannot settle is discussed at the bottom of the printout.

Usage
-----
    python loso_align.py --cache loso_cache_global.npz --emg-norm global
    python loso_align.py --emg-norm global --ea eeg          # EA on EEG branch only
    python loso_align.py --selftest                          # synthetic sanity check

(c) EA whiteners and AdaBN both use only INPUT windows of the held-out session,
never its labels. Legitimate but transductive: real-time deployment needs an
unlabeled calibration buffer per donning to estimate the same statistics.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Euclidean Alignment (He & Wu 2020) — per-session whitener, label-free.
# ----------------------------------------------------------------------------
def session_whitener(X_sess, reg=1e-6):
    """R^-1/2 for one session. X_sess: (n_win, C, T). Returns (C, C) float64.

    R = mean over windows of (X X^T / T)  ->  average spatial covariance.
    Whitening by R^-1/2 maps the session's MEAN covariance to identity, so
    after alignment every session lives in a common whitened space.
    """
    X = X_sess.astype(np.float64)
    n, C, T = X.shape
    R = np.zeros((C, C), np.float64)
    for i in range(n):
        R += X[i] @ X[i].T / T
    R /= max(1, n)
    evals, evecs = np.linalg.eigh(R)             # R is symmetric PSD
    floor = reg * float(evals.max() if evals.size else 1.0)
    evals = np.clip(evals, floor, None)          # regularize tiny eigenvalues
    return (evecs * (evals ** -0.5)) @ evecs.T   # = U diag(λ^-1/2) U^T


def ea_align(X, fold_of, reg=1e-6):
    """Whiten every window by its OWN session's R^-1/2. X: (N, C, T) -> (N, C, T)."""
    Xa = np.empty_like(X)
    for s in np.unique(fold_of):
        sel = np.where(fold_of == s)[0]
        W = session_whitener(X[sel], reg=reg).astype(np.float32)   # (C, C)
        # (C,C) @ (C,T) per window
        Xa[sel] = np.einsum("ij,njt->nit", W, X[sel])
    return Xa


# ----------------------------------------------------------------------------
# AdaBN (Li et al. 2016) — recompute BN running stats on target, freeze weights.
# ----------------------------------------------------------------------------
def _bn_modules(model):
    bns = []
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d)):
            if getattr(m, "track_running_stats", False) and m.running_mean is not None:
                bns.append(m)
    return bns


def adabn_recompute(model, Xe_te, Xm_te, device, batch=256, passes=1):
    """In-place: reset BN running stats, recompute them from the target session's
    inputs (no labels, no grad), leave all learned weights/affine untouched.

    Returns the number of BN modules adapted (0 => AdaBN is a no-op for this net).
    """
    model.eval()                                   # dropout OFF
    bns = _bn_modules(model)
    if not bns:
        return 0
    saved_mom = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()                    # mean=0, var=1, count=0
        m.momentum = None                          # cumulative moving average
        m.train()                                  # BN uses+updates batch stats
    with torch.no_grad():
        for _ in range(passes):
            for s in range(0, Xe_te.shape[0], batch):
                model(Xe_te[s:s + batch], Xm_te[s:s + batch])
    for m, mom in zip(bns, saved_mom):
        m.eval()
        m.momentum = mom
    return len(bns)


# ----------------------------------------------------------------------------
# train / predict (lifted from loso.py intermediate_fold so numbers are
# directly comparable to your existing LOSO runs).
# ----------------------------------------------------------------------------
def train_intermediate(train_obj, Xe_t, Xm_t, y_t, train_idx, n_eeg, n_emg,
                       n_classes, T, y_all, device, epochs, batch_size, lr, seed):
    train_obj.seed_everything(seed)
    tr = torch.as_tensor(train_idx)
    Xe_tr = Xe_t[tr].to(device); Xm_tr = Xm_t[tr].to(device); y_tr = y_t[tr].to(device)
    model = train_obj.IntermediateFusionEEGNet(
        n_eeg=n_eeg, n_emg=n_emg, n_classes=n_classes, T=T, p_drop=0.5).to(device)
    counts = np.bincount(y_all[train_idx], minlength=n_classes).astype(np.float64)
    safe = np.where(counts > 0, counts, 1.0)
    w = torch.tensor(counts.sum() / (safe * n_classes), dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xe_tr.shape[0]; drop_last = n > batch_size
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if drop_last and idx.numel() < batch_size:
                continue
            opt.zero_grad()
            loss = crit(model(Xe_tr[idx], Xm_tr[idx]), y_tr[idx])
            loss.backward(); opt.step()
    del Xe_tr, Xm_tr, y_tr
    return model


def predict(model, Xe_t, Xm_t, test_idx, device, batch=256):
    te = torch.as_tensor(test_idx)
    Xe_te = Xe_t[te].to(device); Xm_te = Xm_t[te].to(device)
    model.eval(); preds = []
    with torch.no_grad():
        for s in range(0, Xe_te.shape[0], batch):
            preds.append(model(Xe_te[s:s + batch], Xm_te[s:s + batch]).argmax(1).cpu().numpy())
    out = np.concatenate(preds) if preds else np.array([], np.int64)
    del Xe_te, Xm_te
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
CONDITIONS = ["baseline", "adabn", "ea", "adabn+ea"]


def load_windows(loso, data, train, cache_path, recordings, pre_ms, post_ms, emg_norm):
    """Load windows straight from the .npz if present (no recordings dir needed);
    otherwise fall back to loso.load_or_build_windows (which rebuilds)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached_norm = str(z["emg_norm"]) if "emg_norm" in z.files else "per_channel"
        if cached_norm != emg_norm:
            print(f"!!! cache emg_norm={cached_norm} but --emg-norm={emg_norm}; "
                  f"using cached arrays AS-IS ({cached_norm}).")
        labels = [str(v) for v in z["labels"]]
        label_to_idx = {l: i for i, l in enumerate(labels)}
        if str(z["recordings"]) != str(Path(recordings).resolve()):
            print(f"[cache] note: recordings path differs from cache; reading arrays anyway.")
        print(f"[cache] loaded windows directly from {cache_path}")
        return (z["Xe"], z["Xm"], z["y_all"].astype(np.int64),
                z["fold_of"].astype(np.int64), [str(s) for s in z["session_names"]],
                int(z["T"]), label_to_idx)
    print(f"[cache] {cache_path} absent -> building via loso.load_or_build_windows")
    return loso.load_or_build_windows(data, train, cache_path, recordings,
                                      pre_ms, post_ms, emg_norm=emg_norm)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, default=Path("recordings"))
    ap.add_argument("--cache", type=Path, default=None,
                    help="window cache (.npz); default loso_cache_<emgnorm>.npz")
    ap.add_argument("--emg-norm", choices=["per_channel", "global"], default="global",
                    help="must match the cache you built with loso.py")
    ap.add_argument("--ea", choices=["none", "eeg", "emg", "both"], default="both",
                    help="which branch(es) Euclidean Alignment whitens")
    ap.add_argument("--ea-reg", type=float, default=1e-6,
                    help="eigenvalue floor (fraction of max eig) for R^-1/2")
    ap.add_argument("--adabn-passes", type=int, default=1)
    ap.add_argument("--conditions", type=str, default=",".join(CONDITIONS),
                    help="comma list of: " + ", ".join(CONDITIONS))
    ap.add_argument("--pre-ms", type=float, default=300.0)
    ap.add_argument("--post-ms", type=float, default=700.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=0,
                    help="limit number of folds (0 = all); for quick smoke tests")
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic EA/AdaBN sanity checks and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in wanted:
        assert c in CONDITIONS, f"unknown condition {c!r}"
    needs_ea = any("ea" in c for c in wanted)

    Path("checkpoints").mkdir(exist_ok=True)   # train.py touches this at import
    import data
    import train
    import loso                                  # reuse cache loader + metrics

    DEVICE = train.get_device()
    print(f"torch {torch.__version__} | device: {DEVICE} | cuda: {torch.cuda.is_available()}")
    if str(DEVICE) == "cpu":
        print("WARNING: CPU is very slow for the 500-tap temporal conv; attach a GPU.")

    cache_path = args.cache or Path(f"loso_cache_{args.emg_norm}.npz")
    print(f"config: emg_norm={args.emg_norm} | ea={args.ea} (reg={args.ea_reg}) "
          f"| adabn_passes={args.adabn_passes} | cache={cache_path}")
    Xe, Xm, y_all, fold_of, session_names, T, label_to_idx = load_windows(
        loso, data, train, cache_path, args.recordings, args.pre_ms, args.post_ms,
        args.emg_norm)

    idx_to_label = {i: l for l, i in label_to_idx.items()}
    n_classes = len(label_to_idx)
    AE = train.ACTIVE_EEG_INDICES; AM = train.ACTIVE_EMG_INDICES
    present = sorted(set(y_all.tolist()))
    chance = 1.0 / max(1, len(present))
    n_folds = len(session_names)
    print(f"\nT={T} | sessions={n_folds} | EEG ch={len(AE)} | EMG ch={len(AM)}")
    print(f"classes present: {[idx_to_label[c] for c in present]}")
    for c in present:
        print(f"  [{c}] {idx_to_label[c]:<14} n={(y_all == c).sum()}")
    print(f"chance (1/#present) = {chance:.3f}")

    # ---- EA-aligned copies (label-free, fold-independent: each session by its own R) ----
    if needs_ea and args.ea != "none":
        t0 = time.time()
        Xe_ea = ea_align(Xe, fold_of, reg=args.ea_reg) if args.ea in ("eeg", "both") else Xe
        Xm_ea = ea_align(Xm, fold_of, reg=args.ea_reg) if args.ea in ("emg", "both") else Xm
        print(f"[ea] whitened branches={args.ea} in {time.time()-t0:.1f}s")
    else:
        Xe_ea, Xm_ea = Xe, Xm

    # model-ready tensors: (N,1,C,T) — matches FusionDataset / loso.py
    Xe_t  = torch.from_numpy(Xe ).unsqueeze(1); Xm_t  = torch.from_numpy(Xm ).unsqueeze(1)
    Xe_et = torch.from_numpy(Xe_ea).unsqueeze(1); Xm_et = torch.from_numpy(Xm_ea).unsqueeze(1)
    y_t = torch.from_numpy(y_all)

    # per-fold balanced acc + pooled OOF predictions, per condition
    fold_bal = {c: [] for c in wanted}
    oof_yt = {c: [] for c in wanted}
    oof_yp = {c: [] for c in wanted}
    bn_seen = 0

    def tr_model(Xe_src, Xm_src, train_idx):
        return train_intermediate(train, Xe_src, Xm_src, y_t, train_idx,
                                  len(AE), len(AM), n_classes, T, y_all, DEVICE,
                                  args.epochs, args.batch_size, args.lr, args.seed)

    def record(cond, yt, yp):
        b = loso.compute_metrics(yt, yp, n_classes, "_")["balanced_accuracy"]
        fold_bal[cond].append(b); oof_yt[cond].append(yt); oof_yp[cond].append(yp)
        return b

    n_eval_folds = n_folds if args.max_folds in (0, None) else min(n_folds, args.max_folds)
    print(f"\n########## LOSO x {len(wanted)} conditions ({n_eval_folds} folds) ##########")
    for f in range(n_eval_folds):
        test_idx = np.where(fold_of == f)[0]
        train_idx = np.where(fold_of != f)[0]
        if len(test_idx) == 0:
            continue
        t0 = time.time(); msg = [f"f{f+1}/{n_folds} {session_names[f]} n={len(test_idx)}"]
        yt = y_all[test_idx]

        # --- raw-trained model serves baseline + adabn ---
        if any(c in ("baseline", "adabn") for c in wanted):
            m_raw = tr_model(Xe_t, Xm_t, train_idx)
            if "baseline" in wanted:
                yp = predict(m_raw, Xe_t, Xm_t, test_idx, DEVICE)
                msg.append(f"base={record('baseline', yt, yp):.3f}")
            if "adabn" in wanted:
                m = copy.deepcopy(m_raw)
                te = torch.as_tensor(test_idx)
                k = adabn_recompute(m, Xe_t[te].to(DEVICE), Xm_t[te].to(DEVICE),
                                    DEVICE, passes=args.adabn_passes)
                bn_seen = max(bn_seen, k)
                yp = predict(m, Xe_t, Xm_t, test_idx, DEVICE)
                msg.append(f"adabn={record('adabn', yt, yp):.3f}")
                del m
            del m_raw

        # --- EA-trained model serves ea + adabn+ea ---
        if any(c in ("ea", "adabn+ea") for c in wanted):
            m_ea = tr_model(Xe_et, Xm_et, train_idx)
            if "ea" in wanted:
                yp = predict(m_ea, Xe_et, Xm_et, test_idx, DEVICE)
                msg.append(f"ea={record('ea', yt, yp):.3f}")
            if "adabn+ea" in wanted:
                m = copy.deepcopy(m_ea)
                te = torch.as_tensor(test_idx)
                k = adabn_recompute(m, Xe_et[te].to(DEVICE), Xm_et[te].to(DEVICE),
                                    DEVICE, passes=args.adabn_passes)
                bn_seen = max(bn_seen, k)
                yp = predict(m, Xe_et, Xm_et, test_idx, DEVICE)
                msg.append(f"adabn+ea={record('adabn+ea', yt, yp):.3f}")
                del m
            del m_ea

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  " + " | ".join(msg) + f" | {time.time()-t0:.0f}s", flush=True)

    if ("adabn" in wanted or "adabn+ea" in wanted) and bn_seen == 0:
        print("\n!!! WARNING: zero BatchNorm modules found in IntermediateFusionEEGNet.")
        print("    AdaBN was a NO-OP; its columns equal the corresponding non-AdaBN run.")

    # ---- pooled OOF report per condition ----
    pooled = []
    for c in wanted:
        yt = np.concatenate(oof_yt[c]); yp = np.concatenate(oof_yp[c])
        m = loso.compute_metrics(yt, yp, n_classes, c)
        loso.print_metrics(m, idx_to_label)
        print(f"\n[{c}] LOSO per-fold balanced_accuracy: "
              f"{np.mean(fold_bal[c]):.4f} +/- {np.std(fold_bal[c]):.4f} "
              f"(n_folds={len(fold_bal[c])})")
        pooled.append(m)
    loso.print_summary_table(pooled, chance=chance)

    # ---- per-fold matrix ----
    print("\n=== per-fold balanced_accuracy (rows=session) ===")
    hdr = f"{'session':<40}" + "".join(f"{c:>12}" for c in wanted)
    print(hdr)
    for f in range(n_folds):
        if f >= len(fold_bal[wanted[0]]):
            continue
        row = f"{session_names[f][:40]:<40}"
        row += "".join(f"{fold_bal[c][f]:>12.3f}" for c in wanted)
        print(row)

    # ---- paired test vs baseline (the honest gate, not the pooled number) ----
    if "baseline" in wanted:
        print("\n=== paired delta vs baseline across folds (Wilcoxon signed-rank) ===")
        base = np.array(fold_bal["baseline"])
        for c in wanted:
            if c == "baseline":
                continue
            arr = np.array(fold_bal[c]); d = arr - base
            improved = int((d > 0).sum()); worse = int((d < 0).sum())
            try:
                from scipy.stats import wilcoxon
                stat, p = wilcoxon(arr, base, zero_method="wilcox",
                                   alternative="two-sided")
                pstr = f"p={p:.4f}"
            except Exception as e:
                pstr = f"p=NA ({type(e).__name__})"
            print(f"  {c:<10} mean_delta={d.mean():+.4f}  median={np.median(d):+.4f}  "
                  f"folds +{improved}/-{worse}/={len(d)-improved-worse}  {pstr}")
        print("\nRead this, not the pooled bal_acc: if mean_delta is within fold noise")
        print("and p is not small, the alignment method did NOT beat global-norm.")


# ----------------------------------------------------------------------------
# synthetic self-test (no train.py / data.py needed)
# ----------------------------------------------------------------------------
def selftest():
    print("[selftest] EA: aligned per-session mean covariance should be ~identity")
    rng = np.random.default_rng(0)
    N, C, T = 60, 8, 400
    fold = np.repeat([0, 1, 2], N // 3)
    X = np.zeros((N, C, T), np.float32)
    for s in range(3):
        A = rng.normal(size=(C, C)).astype(np.float32)        # session-specific mixing
        sel = np.where(fold == s)[0]
        X[sel] = np.einsum("ij,njt->nit", A, rng.normal(size=(len(sel), C, T)).astype(np.float32))
    Xa = ea_align(X, fold)
    for s in range(3):
        sel = np.where(fold == s)[0]
        R = np.mean([Xa[i] @ Xa[i].T / T for i in sel], axis=0)
        err = np.abs(R - np.eye(C)).max()
        print(f"  session {s}: max|R-I| = {err:.4f}  {'OK' if err < 1e-2 else 'FAIL'}")
        assert err < 1e-2

    print("[selftest] AdaBN: BN running stats should shift toward target stats")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(1, 4, (1, 3), padding=(0, 1), bias=False)
            self.bn = torch.nn.BatchNorm2d(4)
            self.head = torch.nn.Linear(4, 3)

        def forward(self, xe, xm):
            h = self.bn(self.c(xe)).mean(dim=(2, 3))
            return self.head(h)

    dev = torch.device("cpu")
    model = Tiny().to(dev)
    # give BN non-trivial "source" stats
    src = torch.randn(32, 1, 1, 50)
    model.train()
    with torch.no_grad():
        model(src, None)
    pre = model.bn.running_mean.clone()
    target = torch.randn(40, 1, 1, 50) * 5.0 + 10.0          # very different domain
    k = adabn_recompute(model, target, target, dev, passes=1)
    post = model.bn.running_mean.clone()
    shift = (post - pre).abs().max().item()
    print(f"  bn modules adapted = {k}  max|Δrunning_mean| = {shift:.3f}  "
          f"{'OK' if (k == 1 and shift > 0.1) else 'FAIL'}")
    assert k == 1 and shift > 0.1

    class NoBN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(4, 3)

        def forward(self, xe, xm):
            return self.l(xe.mean(dim=(2, 3)))

    k0 = adabn_recompute(NoBN().to(dev), torch.randn(5, 1, 1, 4),
                         torch.randn(5, 1, 1, 4), dev)
    print(f"  no-BN model: adapted = {k0}  {'OK' if k0 == 0 else 'FAIL'}")
    assert k0 == 0
    print("[selftest] all checks passed.")


if __name__ == "__main__":
    main()
