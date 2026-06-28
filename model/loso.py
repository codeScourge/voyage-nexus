#!/usr/bin/env python3
"""
loso.py — leave-session-out (LOSO) model comparison for the Voyage rig.

Runs three models under ONE honest protocol (leave-one-session-out, pooled
out-of-fold predictions, val.py-style report):

  1. intermediate   IntermediateFusionEEGNet (the real train.py model). Trained
                    fixed-epoch per fold, NO early stopping / checkpoint selection
                    on the held-out session. Fed from precomputed GPU tensors.
  2. feature        FeatureFusionClassifier (row2) — native multitask training.
  3. late           LateFusionStacker (row3) — native OOF-stacking (the slow one).

Imports your data.py / train.py / row2 / row3 unchanged.

(c) Window cache: the slow notch + windowing runs ONCE and is saved to --cache
(an .npz). Later runs reload it instantly. It auto-rebuilds if --recordings,
--pre-ms/--post-ms, the active channels, or the label set change (delete the
file to force a rebuild).

Usage
-----
    python loso.py --recordings ./recordings --models intermediate,feature,late
    python loso.py --recordings ./recordings --models intermediate --epochs 2  # quick

Place next to data.py / train.py / row2_feature_fusion.py / row3_late_fusion.py.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


# ----------------------------------------------------------------------------
# val.py-style metrics + printing
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, n_classes, name, n_samples=None):
    from sklearn.metrics import cohen_kappa_score
    y_true = np.asarray(y_true, np.int64)
    y_pred = np.asarray(y_pred, np.int64)
    cm = np.zeros((n_classes, n_classes), np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = []
    for c in range(n_classes):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": int(support)})
    total = len(y_true)
    accuracy = float((y_true == y_pred).mean()) if total else 0.0
    recalls = [r["recall"] for r in per_class if r["support"] > 0]
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean([r["f1"] for r in per_class])) if per_class else 0.0
    w = np.array([r["support"] for r in per_class], np.float64)
    weighted_f1 = float(np.average([r["f1"] for r in per_class], weights=w)) if w.sum() > 0 else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred)) if total else 0.0
    return {"split": name, "n_samples": (n_samples or total), "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy, "macro_f1": macro_f1,
            "weighted_f1": weighted_f1, "kappa": kappa, "confusion_matrix": cm,
            "per_class": per_class}


def print_metrics(m, idx_to_label):
    print(f"\n=== {m['split']} ===")
    print(f"samples:           {m['n_samples']}")
    print(f"accuracy:          {m['accuracy']:.4f}")
    print(f"balanced_accuracy: {m['balanced_accuracy']:.4f}")
    print(f"cohen_kappa:       {m['kappa']:.4f}")
    print(f"macro_f1:          {m['macro_f1']:.4f}")
    print(f"weighted_f1:       {m['weighted_f1']:.4f}")
    print("\nper-class:")
    print(f"{'label':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for c, row in enumerate(m["per_class"]):
        lab = idx_to_label.get(c, str(c))
        print(f"{lab:<20} {row['precision']:>10.4f} {row['recall']:>10.4f} "
              f"{row['f1']:>10.4f} {row['support']:>10d}")
    print("\nconfusion matrix (rows=true, cols=pred):")
    labels = [idx_to_label.get(i, str(i)) for i in range(len(m['per_class']))]
    print("true\\pred".ljust(20) + "".join(l[:12].rjust(12) for l in labels))
    for c, row in enumerate(m["confusion_matrix"]):
        lab = idx_to_label.get(c, str(c))
        print(f"{lab[:20]:<20}" + "".join(str(int(v)).rjust(12) for v in row))


def print_summary_table(metrics_list, chance=None):
    print("\n=== summary (pooled out-of-fold predictions, leave-session-out) ===")
    if chance is not None:
        print(f"chance = {chance:.3f}")
    print(f"{'model':<16} {'samples':>8} {'accuracy':>10} {'bal_acc':>10} "
          f"{'kappa':>9} {'macro_f1':>10} {'weighted_f1':>12}")
    for m in metrics_list:
        print(f"{m['split']:<16} {m['n_samples']:>8} {m['accuracy']:>10.4f} "
              f"{m['balanced_accuracy']:>10.4f} {m['kappa']:>9.4f} "
              f"{m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f}")


# ----------------------------------------------------------------------------
# (c) cached window build
# ----------------------------------------------------------------------------
def load_or_build_windows(data, train, cache_path, recordings, pre_ms, post_ms, emg_norm="per_channel"):
    """Build (or load from cache) Xe/Xm (N,C,T), y_all, fold_of, session_names, T.

    The expensive notch + windowing runs only on a cache miss. Cache auto-rebuilds
    if recordings/pre/post/active-channels/labels change.
    """
    label_to_idx = data.default_label_to_idx()
    AE = list(train.ACTIVE_EEG_INDICES); AM = list(train.ACTIVE_EMG_INDICES)
    labels = list(label_to_idx.keys())
    cache_path = Path(cache_path)

    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached_norm = str(z["emg_norm"]) if "emg_norm" in z.files else "per_channel"
        same = (str(z["recordings"]) == str(Path(recordings).resolve())
                and float(z["pre_ms"]) == float(pre_ms)
                and float(z["post_ms"]) == float(post_ms)
                and cached_norm == emg_norm
                and [int(v) for v in z["active_eeg"]] == AE
                and [int(v) for v in z["active_emg"]] == AM
                and [str(v) for v in z["labels"]] == labels)
        if same:
            print(f"[cache] loaded windows from {cache_path}")
            return (z["Xe"], z["Xm"], z["y_all"].astype(np.int64),
                    z["fold_of"].astype(np.int64),
                    [str(s) for s in z["session_names"]], int(z["T"]), label_to_idx)
        print(f"[cache] config changed -> rebuilding {cache_path}")

    from _viewer_core import discover_sessions
    sessions = discover_sessions(Path(recordings))
    print(f"[build] {len(sessions)} sessions; windowing (notch/band-pass, runs once)...")
    t0 = time.time()
    dataset = data.SessionEventDataset(sessions, pre_ms=pre_ms, post_ms=post_ms)
    N = len(dataset)
    names = [Path(dataset._session_dirs[i]).name for i in range(N)]
    uniq = sorted(set(names))
    s2f = {s: k for k, s in enumerate(uniq)}
    fold_of = np.array([s2f[s] for s in names], dtype=np.int64)
    y_all = np.array([label_to_idx[dataset[i]["label"]] for i in range(N)], dtype=np.int64)
    T = dataset[0]["x"].shape[0]
    Xe = np.zeros((N, len(AE), T), np.float32)
    Xm = np.zeros((N, len(AM), T), np.float32)
    for i in range(N):
        x = dataset[i]["x"]                       # (T, 32)
        xe = x[:, AE]                             # EEG: always per-channel z-score
        xe = (xe - xe.mean(0, keepdim=True)) / (xe.std(0, keepdim=True) + 1e-6)
        xm = x[:, AM] - x[:, AM].mean(0, keepdim=True)   # center each EMG channel
        if emg_norm == "global":
            # one scalar scale over all EMG channels+time: removes donning gain but
            # KEEPS cross-channel amplitude ratios (the muscle-activation pattern)
            xm = xm / (xm.std() + 1e-6)
        else:                                    # per_channel (original behaviour)
            xm = xm / (xm.std(0, keepdim=True) + 1e-6)
        Xe[i] = xe.numpy().T; Xm[i] = xm.numpy().T
    print(f"[build] {N} windows, T={T} in {time.time()-t0:.1f}s")
    np.savez(cache_path, Xe=Xe, Xm=Xm, y_all=y_all, fold_of=fold_of,
             session_names=np.array(uniq), T=T,
             recordings=str(Path(recordings).resolve()),
             pre_ms=float(pre_ms), post_ms=float(post_ms), emg_norm=emg_norm,
             active_eeg=np.array(AE), active_emg=np.array(AM), labels=np.array(labels))
    print(f"[cache] saved windows to {cache_path}  (delete to force rebuild)")
    return Xe, Xm, y_all, fold_of, uniq, T, label_to_idx


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, default=Path("recordings"))
    ap.add_argument("--models", type=str, default="intermediate,feature,late",
                    help="comma list of: intermediate, feature, late")
    ap.add_argument("--cache", type=Path, default=None,
                    help="window cache file (.npz); default loso_cache_<emgnorm>.npz")
    ap.add_argument("--emg-norm", choices=["per_channel", "global"], default="per_channel",
                    help="EMG window normalization: per_channel (z-score each ch) or "
                         "global (one scalar scale; keeps cross-channel amplitude ratios)")
    ap.add_argument("--no-hudgins", action="store_true",
                    help="drop Hudgins amplitude features in feature/late fusion "
                         "(pure learned conv EMG front-end)")
    ap.add_argument("--pre-ms", type=float, default=300.0)
    ap.add_argument("--post-ms", type=float, default=700.0)
    ap.add_argument("--epochs", type=int, default=80,
                    help="fixed epochs for the intermediate model")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fusion-max-epochs", type=int, default=120)
    ap.add_argument("--late-oof-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wanted = {s.strip() for s in args.models.split(",") if s.strip()}

    Path("checkpoints").mkdir(exist_ok=True)   # train.py touches this at import
    import data
    import train

    DEVICE = train.get_device()
    print(f"torch {torch.__version__} | device: {DEVICE} | cuda: {torch.cuda.is_available()}")
    if str(DEVICE) == "cpu":
        print("WARNING: running on CPU — the 500-tap temporal conv at 1 kHz is very slow here. "
              "Attach a GPU runtime if you can.")

    cache_path = args.cache or Path(f"loso_cache_{args.emg_norm}.npz")
    print(f"config: emg_norm={args.emg_norm} | hudgins={'off' if args.no_hudgins else 'on'} "
          f"| cache={cache_path}")
    Xe, Xm, y_all, fold_of, session_names, T, label_to_idx = load_or_build_windows(
        data, train, cache_path, args.recordings, args.pre_ms, args.post_ms,
        emg_norm=args.emg_norm)

    idx_to_label = {i: l for l, i in label_to_idx.items()}
    n_classes = len(label_to_idx)
    AE = train.ACTIVE_EEG_INDICES
    AM = train.ACTIVE_EMG_INDICES
    present = sorted(set(y_all.tolist()))
    chance = 1.0 / max(1, len(present))
    n_folds = len(session_names)
    print(f"\nT={T} | sessions={n_folds} | EEG ch={len(AE)} | EMG ch={len(AM)}")
    print(f"classes present: {[idx_to_label[c] for c in present]}")
    for c in range(n_classes):
        print(f"  [{c}] {idx_to_label[c]:<14} n={(y_all == c).sum()}")
    print(f"chance (1/#present) = {chance:.3f}")

    # (b) model-ready tensors for the intermediate net (== FusionDataset output)
    Xe_t = torch.from_numpy(Xe).unsqueeze(1)
    Xm_t = torch.from_numpy(Xm).unsqueeze(1)
    y_t = torch.from_numpy(y_all)

    def loso_run(run_fold, name):
        yts, yps, fold_bal = [], [], []
        print(f"\n########## {name}  ({n_folds} folds) ##########")
        bar = tqdm(range(n_folds), desc=name, unit="fold")
        for f in bar:
            test_idx = np.where(fold_of == f)[0]
            train_idx = np.where(fold_of != f)[0]
            if len(test_idx) == 0:
                continue
            t0 = time.time()
            yp = np.asarray(run_fold(train_idx, test_idx, f"{name} f{f+1}/{n_folds}"), np.int64)
            yt = y_all[test_idx]
            b = compute_metrics(yt, yp, n_classes, "_")["balanced_accuracy"]
            fold_bal.append(b); yts.append(yt); yps.append(yp)
            bar.set_postfix(sess=session_names[f], bal=f"{b:.3f}",
                            mean=f"{np.mean(fold_bal):.3f}", t=f"{time.time()-t0:.0f}s")
        bar.close()
        yt = np.concatenate(yts); yp = np.concatenate(yps)
        m = compute_metrics(yt, yp, n_classes, name)
        print_metrics(m, idx_to_label)
        print(f"\n[{name}] LOSO per-fold balanced_accuracy: "
              f"{np.mean(fold_bal):.4f} +/- {np.std(fold_bal):.4f}  (n_folds={len(fold_bal)})")
        print(f"[{name}] chance = {chance:.3f}")
        return m

    def intermediate_fold(train_idx, test_idx, tag=""):
        train.seed_everything(args.seed)
        tr = torch.as_tensor(train_idx)
        Xe_tr = Xe_t[tr].to(DEVICE); Xm_tr = Xm_t[tr].to(DEVICE); y_tr = y_t[tr].to(DEVICE)
        model = train.IntermediateFusionEEGNet(
            n_eeg=len(AE), n_emg=len(AM), n_classes=n_classes, T=T, p_drop=0.5).to(DEVICE)
        counts = np.bincount(y_all[train_idx], minlength=n_classes).astype(np.float64)
        safe = np.where(counts > 0, counts, 1.0)
        w = torch.tensor(counts.sum() / (safe * n_classes), dtype=torch.float32, device=DEVICE)
        crit = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        n = Xe_tr.shape[0]; drop_last = n > args.batch_size
        model.train()
        ep_bar = tqdm(range(args.epochs), desc=f"  train {tag}", unit="ep", leave=False)
        for _ in ep_bar:
            perm = torch.randperm(n, device=DEVICE); last = 0.0
            for s in range(0, n, args.batch_size):
                idx = perm[s:s + args.batch_size]
                if drop_last and idx.numel() < args.batch_size:
                    continue
                opt.zero_grad()
                loss = crit(model(Xe_tr[idx], Xm_tr[idx]), y_tr[idx])
                loss.backward(); opt.step(); last = float(loss.item())
            ep_bar.set_postfix(loss=f"{last:.3f}")
        ep_bar.close()
        te = torch.as_tensor(test_idx)
        Xe_te = Xe_t[te].to(DEVICE); Xm_te = Xm_t[te].to(DEVICE)
        model.eval(); preds = []
        with torch.no_grad():
            for s in range(0, Xe_te.shape[0], 256):
                preds.append(model(Xe_te[s:s + 256], Xm_te[s:s + 256]).argmax(1).cpu().numpy())
        del Xe_tr, Xm_tr, y_tr, Xe_te, Xm_te
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(preds)

    def feature_fold(train_idx, test_idx, tag=""):
        from row2_feature_fusion import FeatureFusionClassifier
        clf = FeatureFusionClassifier(n_classes=n_classes, device=str(DEVICE),
                                      max_epochs=args.fusion_max_epochs, seed=args.seed,
                                      use_hudgins=not args.no_hudgins, verbose=False)
        clf.fit(Xe[train_idx], Xm[train_idx], y_all[train_idx],
                groups=fold_of[train_idx], classes=np.arange(n_classes))
        return clf.predict(Xe[test_idx], Xm[test_idx])

    def late_fold(train_idx, test_idx, tag=""):
        from row3_late_fusion import LateFusionStacker
        clf = LateFusionStacker(n_classes=n_classes, n_times_eeg=T, n_times_emg=T,
                                device=str(DEVICE), max_epochs=args.fusion_max_epochs,
                                oof_folds=args.late_oof_folds, seed=args.seed,
                                emg_use_hudgins=not args.no_hudgins, verbose=False)
        clf.fit(Xe[train_idx], Xm[train_idx], y_all[train_idx],
                groups=fold_of[train_idx], classes=np.arange(n_classes))
        return clf.predict(Xe[test_idx], Xm[test_idx])

    runners = [("intermediate", intermediate_fold), ("feature", feature_fold), ("late", late_fold)]
    results = [loso_run(fn, name) for name, fn in runners if name in wanted]
    if results:
        print_summary_table(results, chance=chance)


if __name__ == "__main__":
    main()
