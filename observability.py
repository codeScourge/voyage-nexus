"""
observability.py
=================
Interpretation / observability suite for the silent-speech decoders. Everything
here is about answering the question that matters most for an over-ear EEG+EMG
device: *is the model decoding speech-relevant signal, or an artifact/confound?*
-- plus standard trustworthiness checks.

Each analysis is a standalone function taking a FITTED estimator and a held-out,
group-defined Dataset. `run_all` does one leakage-free group split, fits, and
runs the whole battery, writing PNGs + a JSON report.

Battery
-------
  1. permutation_test         null distribution by label-shuffle -> p-value + CI.
                              SHOWS: is accuracy above chance for real, not luck?
  2. confusion_and_report     confusion matrix + per-class precision/recall/F1.
                              SHOWS: which classes are confused, class imbalance effects.
  3. calibration             reliability curve + Expected Calibration Error.
                              SHOWS: are probabilities trustworthy (not over-confident)?
  4. occlusion_sensitivity   model-agnostic; zero out time-bins / channels and
                              measure prob drop. SHOWS: WHERE in time & WHICH
                              channels the decision depends on (artifact localizer).
  5. integrated_gradients    Torch-only; attribution over (channel,time).
                              SHOWS: fine-grained spatiotemporal evidence.
  6. temporal_generalization train on sub-window @t, test @t' (King & Dehaene).
                              SHOWS: transient vs sustained neural/EMG codes.
  7. learning_curve          balanced acc vs #training groups.
                              SHOWS: are you data-limited (still rising) or
                              architecture-limited (plateau)? -> answers "collect more?"
  8. channel_dropout         degrade by removing k channels.
                              SHOWS: robustness to electrode loss/shift on your rig.
  9. stream_ablation         EEG-only / EMG-only / both (uses data.select_streams).
                              SHOWS (your rig): is "EEG+EMG" really just EMG?
 10. dda_coefficient_space   DDA-specific; 2-D projection of [a1,a2,a3,rho] features.
                              SHOWS: class separability in the DDA feature space.

Confound note: on the Gaddy proxy (EMG-only) #9 reduces to channel-subset
ablation. On your 16+16 rig it becomes the decisive EEG-vs-EMG contribution test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             classification_report)

import data as D
from models import make_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# --------------------------------------------------------------------------- #
def _proba(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)
    # fall back to one-hot of predict
    p = est.predict(X)
    K = int(p.max()) + 1
    oh = np.zeros((len(p), K)); oh[np.arange(len(p)), p] = 1
    return oh


# 1 ----------------------------------------------------------------------------
def permutation_test(est_factory, X, y, groups, n_perm=200, n_splits=4, seed=0):
    """Group-CV balanced acc vs a null built by permuting labels WITHIN the CV.
    est_factory() -> fresh estimator."""
    rng = np.random.default_rng(seed)

    def cv_score(yy):
        gkf = GroupKFold(min(n_splits, len(np.unique(groups))))
        s = []
        for itr, ite in gkf.split(X, yy, groups):
            est = est_factory()
            try: est.fit(X[itr], yy[itr], groups=groups[itr])
            except TypeError: est.fit(X[itr], yy[itr])
            s.append(balanced_accuracy_score(yy[ite], est.predict(X[ite])))
        return float(np.mean(s))

    observed = cv_score(y)
    null = np.empty(n_perm)
    uniq = np.unique(groups)
    for i in range(n_perm):
        # permute labels at the GROUP level to respect structure
        glabel = {g: y[groups == g][0] for g in uniq}
        shuffled_vals = rng.permutation(list(glabel.values()))
        gmap = dict(zip(glabel.keys(), shuffled_vals))
        yperm = np.array([gmap[g] for g in groups])
        null[i] = cv_score(yperm)
    p = (1 + np.sum(null >= observed)) / (1 + n_perm)
    ci = np.percentile(null, [2.5, 97.5])
    return dict(observed=observed, null_mean=float(null.mean()),
                null_ci=[float(ci[0]), float(ci[1])], p_value=float(p),
                n_perm=n_perm)


# 2 ----------------------------------------------------------------------------
def confusion_and_report(est, X, y, label_names=None, out="confusion.png"):
    pred = est.predict(X)
    cm = confusion_matrix(y, pred)
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    if _HAS_PLT:
        fig, ax = plt.subplots(figsize=(4 + 0.4 * len(cm), 3.5 + 0.4 * len(cm)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        names = label_names or [str(i) for i in range(len(cm))]
        ax.set_xticks(range(len(cm))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(cm))); ax.set_yticklabels(names)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(im); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(confusion=cm.tolist(), report=rep)


# 3 ----------------------------------------------------------------------------
def calibration(est, X, y, n_bins=10, out="calibration.png"):
    proba = _proba(est, X)
    conf = proba.max(1)
    pred = proba.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, xs, ys = 0.0, [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc = correct[m].mean(); avg_conf = conf[m].mean()
        ece += (m.sum() / len(conf)) * abs(acc - avg_conf)
        xs.append(avg_conf); ys.append(acc)
    if _HAS_PLT:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(xs, ys, "o-")
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability (ECE={ece:.3f})")
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(ece=float(ece), bins=list(zip(xs, ys)))


# 4 ----------------------------------------------------------------------------
def occlusion_sensitivity(est, X, y, n_time_bins=10, out="occlusion.png"):
    """Model-agnostic. Zero a time band (or a channel) on correctly-handled
    inputs and measure mean drop in P(true class)."""
    base = _proba(est, X)
    true_p = base[np.arange(len(y)), y]
    C, T = X.shape[1], X.shape[2]
    edges = np.linspace(0, T, n_time_bins + 1).astype(int)
    time_drop = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        Xo = X.copy(); Xo[:, :, lo:hi] = 0.0
        po = _proba(est, Xo)[np.arange(len(y)), y]
        time_drop.append(float((true_p - po).mean()))
    chan_drop = []
    for c in range(C):
        Xo = X.copy(); Xo[:, c, :] = 0.0
        po = _proba(est, Xo)[np.arange(len(y)), y]
        chan_drop.append(float((true_p - po).mean()))
    if _HAS_PLT:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        axs[0].bar(range(n_time_bins), time_drop); axs[0].set_title("time-band importance")
        axs[0].set_xlabel("time bin"); axs[0].set_ylabel("drop in P(true)")
        axs[1].bar(range(C), chan_drop); axs[1].set_title("channel importance")
        axs[1].set_xlabel("channel")
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(time_importance=time_drop, channel_importance=chan_drop)


# 5 ----------------------------------------------------------------------------
def integrated_gradients(est, X, y, steps=32, out="integrated_gradients.png"):
    """Torch-only attribution over (channel,time), averaged over samples per class.
    Requires est.model_ (a TorchClassifier)."""
    try:
        import torch
    except Exception:
        return dict(skipped="torch unavailable")
    if not hasattr(est, "model_"):
        return dict(skipped="not a torch model")
    # models whose CNN consumes a derived image (e.g. RusnacCNN's C x C matrix)
    # cannot be attributed over the raw (C,T) input -> skip cleanly.
    if est.__class__.__name__ == "RusnacCNN":
        return dict(skipped="image-input model (cross-cov); IG over raw signal N/A")
    try:
        model = est.model_.eval()
    except Exception:
        return dict(skipped="no fitted torch model")
    dev = next(model.parameters()).device
    Xt = torch.tensor(np.asarray(X, np.float32), device=dev)
    baseline = torch.zeros_like(Xt)
    attr = torch.zeros_like(Xt)
    for a in torch.linspace(0, 1, steps):
        x = (baseline + a * (Xt - baseline)).clone().requires_grad_(True)
        logits = model(x)
        sel = logits[np.arange(len(y)), torch.tensor(y, device=dev)].sum()
        grad = torch.autograd.grad(sel, x)[0]
        attr += grad / steps
    ig = ((Xt - baseline) * attr).abs().detach().cpu().numpy()  # (n,C,T)
    per_class = {}
    for k in np.unique(y):
        per_class[int(k)] = ig[y == k].mean(0).tolist()
    if _HAS_PLT:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(ig.mean(0), aspect="auto", cmap="magma")
        ax.set_xlabel("time"); ax.set_ylabel("channel")
        ax.set_title("Integrated gradients |attribution| (mean)")
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(mean_abs_attr_per_class=per_class)


# 6 ----------------------------------------------------------------------------
def temporal_generalization(est_factory, X, y, groups, n_sub=6, sub_frac=0.4,
                            n_splits=3, out="tempgen.png"):
    """Train a classifier on a sub-window centered at t_i, test on sub-window at
    t_j -> matrix of balanced accuracies. Diagonal = transient code, full square =
    sustained code."""
    T = X.shape[2]
    w = max(8, int(sub_frac * T))
    centers = np.linspace(w // 2, T - w // 2, n_sub).astype(int)
    def crop(c):
        lo = c - w // 2; return X[:, :, lo:lo + w]
    M = np.full((n_sub, n_sub), np.nan)
    gkf = GroupKFold(min(n_splits, len(np.unique(groups))))
    for i, ci in enumerate(centers):
        Xi = crop(ci)
        for j, cj in enumerate(centers):
            Xj = crop(cj)
            sc = []
            for itr, ite in gkf.split(Xi, y, groups):
                est = est_factory()
                try: est.fit(Xi[itr], y[itr], groups=groups[itr])
                except TypeError: est.fit(Xi[itr], y[itr])
                sc.append(balanced_accuracy_score(y[ite], est.predict(Xj[ite])))
            M[i, j] = np.mean(sc)
    if _HAS_PLT:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(M, cmap="viridis", vmin=0.5 if y.max() == 1 else None, origin="lower")
        ax.set_xlabel("test time"); ax.set_ylabel("train time")
        ax.set_title("Temporal generalization")
        fig.colorbar(im); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(matrix=M.tolist(), centers=centers.tolist())


# 7 ----------------------------------------------------------------------------
def learning_curve(est_factory, X, y, groups, fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
                   n_splits=4, seed=0, out="learning_curve.png"):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    curve = []
    for f in fractions:
        keep = set(rng.choice(uniq, max(2, int(f * len(uniq))), replace=False))
        m = np.array([g in keep for g in groups])
        Xs, ys, gs = X[m], y[m], groups[m]
        if len(np.unique(ys)) < 2:
            curve.append((f, np.nan)); continue
        gkf = GroupKFold(min(n_splits, len(np.unique(gs))))
        sc = []
        for itr, ite in gkf.split(Xs, ys, gs):
            est = est_factory()
            try: est.fit(Xs[itr], ys[itr], groups=gs[itr])
            except TypeError: est.fit(Xs[itr], ys[itr])
            sc.append(balanced_accuracy_score(ys[ite], est.predict(Xs[ite])))
        curve.append((float(f), float(np.mean(sc))))
    if _HAS_PLT:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot([c[0] for c in curve], [c[1] for c in curve], "o-")
        ax.set_xlabel("fraction of training groups"); ax.set_ylabel("balanced acc")
        ax.set_title("Learning curve"); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(curve=curve)


# 8 ----------------------------------------------------------------------------
def channel_dropout(est, X, y, max_drop=None, reps=20, seed=0, out="channel_dropout.png"):
    rng = np.random.default_rng(seed)
    C = X.shape[1]
    max_drop = max_drop or (C - 1)
    res = []
    for k in range(0, max_drop + 1):
        accs = []
        for _ in range(reps if k > 0 else 1):
            Xo = X.copy()
            if k > 0:
                drop = rng.choice(C, k, replace=False)
                Xo[:, drop, :] = 0.0
            accs.append(balanced_accuracy_score(y, est.predict(Xo)))
        res.append((k, float(np.mean(accs)), float(np.std(accs))))
    if _HAS_PLT:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ks = [r[0] for r in res]; ms = [r[1] for r in res]; ss = [r[2] for r in res]
        ax.errorbar(ks, ms, yerr=ss, fmt="o-")
        ax.set_xlabel("# channels dropped"); ax.set_ylabel("balanced acc")
        ax.set_title("Channel-dropout robustness"); fig.tight_layout()
        fig.savefig(out, dpi=140); plt.close(fig)
    return dict(dropout=res)


# 9 ----------------------------------------------------------------------------
def stream_ablation(model_name, ds: D.Dataset, hp, n_splits=4, out="stream_ablation.json"):
    """Refit and group-CV on EEG-only / EMG-only / both. Needs ds.meta stream_index
    populated (your rig). On EMG-only Gaddy data, 'eeg' is empty -> only emg/both run."""
    res = {}
    avail = [s for s in ("eeg", "emg") if ds.meta.get("stream_index", {}).get(s)]
    combos = [(s,) for s in avail] + ([("eeg", "emg")] if len(avail) == 2 else [])
    for combo in combos:
        sub = D.select_streams(ds, combo)
        gkf = GroupKFold(min(n_splits, len(np.unique(sub.groups))))
        sc = []
        for itr, ite in gkf.split(sub.X, sub.y, sub.groups):
            n_cls = int(sub.y.max()) + 1
            est = make_model(model_name, sub.X.shape[1], sub.X.shape[2], n_cls, **hp)
            try: est.fit(sub.X[itr], sub.y[itr], groups=sub.groups[itr])
            except TypeError: est.fit(sub.X[itr], sub.y[itr])
            sc.append(balanced_accuracy_score(sub.y[ite], est.predict(sub.X[ite])))
        res["+".join(combo)] = dict(balanced_acc=float(np.mean(sc)), std=float(np.std(sc)))
    Path(out).write_text(json.dumps(res, indent=2))
    return res


# 10 ---------------------------------------------------------------------------
def dda_coefficient_space(est, X, y, out="dda_space.png"):
    if not hasattr(est, "transform") or not hasattr(est, "delays_"):
        return dict(skipped="not a DDA model")
    F = est.transform(X)
    from sklearn.decomposition import PCA
    Z = PCA(2).fit_transform(F)
    if _HAS_PLT:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for k in np.unique(y):
            ax.scatter(Z[y == k, 0], Z[y == k, 1], s=12, label=str(k), alpha=.6)
        ax.legend(); ax.set_title(f"DDA features (delays={est.delays_})")
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(delays=list(est.delays_))


# --------------------------------------------------------------------------- #
def run_all(model_name, ds: D.Dataset, hp, outdir="obs_out", seed=0,
            n_perm=200, test_frac=0.3):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    X, y, g = ds.X, ds.y, ds.groups
    n_cls = ds.n_classes
    factory = lambda: make_model(model_name, X.shape[1], X.shape[2], n_cls, **hp)

    # one leakage-free group split for the fitted-model analyses
    tr, te = next(GroupShuffleSplit(1, test_size=test_frac, random_state=seed).split(X, y, g))
    est = factory()
    try: est.fit(X[tr], y[tr], groups=g[tr])
    except TypeError: est.fit(X[tr], y[tr])

    report = {}
    report["permutation_test"] = permutation_test(factory, X, y, g, n_perm=n_perm, seed=seed)
    report["confusion"] = confusion_and_report(est, X[te], y[te],
                                                ds.meta.get("label_names"),
                                                out=str(outdir / "confusion.png"))
    report["calibration"] = calibration(est, X[te], y[te], out=str(outdir / "calibration.png"))
    report["occlusion"] = occlusion_sensitivity(est, X[te], y[te], out=str(outdir / "occlusion.png"))
    report["integrated_gradients"] = integrated_gradients(est, X[te], y[te],
                                                           out=str(outdir / "integrated_gradients.png"))
    report["temporal_generalization"] = temporal_generalization(factory, X, y, g,
                                                                 out=str(outdir / "tempgen.png"))
    report["learning_curve"] = learning_curve(factory, X, y, g, out=str(outdir / "learning_curve.png"))
    report["channel_dropout"] = channel_dropout(est, X[te], y[te], out=str(outdir / "channel_dropout.png"))
    report["stream_ablation"] = stream_ablation(model_name, ds, hp, out=str(outdir / "stream_ablation.json"))
    if model_name == "dda":
        report["dda_space"] = dda_coefficient_space(est, X[te], y[te], out=str(outdir / "dda_space.png"))

    (outdir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"observability report -> {outdir/'report.json'}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="rusnac",
                    choices=["eegnet", "deepconvnet", "dda", "cspdnn", "rusnac"])
    ap.add_argument("--win_sec", type=float, default=0.5)
    ap.add_argument("--hop_sec", type=float, default=0.25)
    ap.add_argument("--granularity", default="word", choices=["word", "phoneme"])
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--outdir", default="obs_out")
    a = ap.parse_args()
    # Grigore & Rusnac preprocessing: notch only, keep high frequencies.
    cfg = D.PreprocConfig(bp_low=1.0, bp_high=480.0, notch=60.0,
                          decimate_to=(250.0 if a.model in ("eegnet", "deepconvnet") else None))
    ds = D.load_gaddy(a.root, cfg, a.win_sec, a.hop_sec,
                      granularity=a.granularity, top_k=a.top_k)
    hp = {} if a.model not in ("cspdnn",) else dict(n_select=5)
    run_all(a.model, ds, hp, outdir=a.outdir, n_perm=a.n_perm)
