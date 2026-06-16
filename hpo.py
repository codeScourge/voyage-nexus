"""
hpo.py
======
Comprehensive, leakage-free hyperparameter search for the silent-speech project.

Design goals (you said: runtime doesn't matter, just find the true best config):
  * Search the WHOLE pipeline, not just the model: preprocessing (band, notch,
    decimation, window length/hop, normalization, artifact clip) AND per-model
    architecture/training HPs.
  * NESTED cross-validation for an *unbiased* estimate:
        outer = GroupKFold  -> reports generalization (groups = utterance/session)
        inner = GroupKFold  -> selects HPs (Optuna optimizes the inner mean)
    Groups guarantee windows from one utterance/session never cross a split.
  * One Optuna study PER (model, outer_fold). TPE sampler + median pruner.
  * Persistent SQLite storage -> resumable; kill and restart anytime.
  * Re-windowing happens INSIDE the objective because window length / decimation
    are themselves tuned, so the raw continuous signal is the true input.

Because preprocessing is tuned, the search operates on a RawCorpus (continuous
per-utterance signals + labels + groups), not pre-windowed arrays.

Usage
-----
  python hpo.py --root /path/to/{silent,voiced} --model eegnet --outer 5 \
      --inner 4 --trials 200 --study_db sqlite:///hpo_eegnet.db

  # all models sequentially:
  for m in eegnet deepconvnet dda cspdnn; do python hpo.py --model $m ... ; done
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

import data as D
from models import make_model


# ----------------------------------------------------------------------------- #
#  Raw corpus: continuous signals kept un-windowed so preprocessing is tunable
# ----------------------------------------------------------------------------- #
@dataclass
class RawUtt:
    sig: np.ndarray   # (T, C) raw float
    segments: list    # list[(label:int, s0:float, s1:float)]
    group: int


def load_raw_corpus(root, task=None, modes=("silent", "voiced"),
                    max_utts_per_mode=None, granularity="word", top_k=10):
    """Continuous per-utterance signals kept un-windowed (so preprocessing/window
    length stay tunable). DEFAULT task = word/phoneme segments (closed top_k vocab).
    Returns (utts, label_names)."""
    root = Path(root)
    label_names = []
    if task is None:
        vocab = D.build_vocab(root, granularity, top_k, modes, max_utts_per_mode)
        task = D.make_task_word_segments(vocab, granularity)
        label_names = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    utts, gc = [], 0
    for mode in modes:
        folder = root / mode
        ids = D._list_utterances(folder)
        if max_utts_per_mode:
            ids = ids[:max_utts_per_mode]
        for idx in ids:
            emg_p = D._find(folder, idx, "emg.npy")
            if emg_p is None:
                continue
            info_p = D._find(folder, idx, "info.json")
            info = json.loads(info_p.read_text()) if info_p else {}
            out = task(mode, info)
            if out is None:
                continue
            segs = [(int(out), 0.0, 1.0)] if isinstance(out, (int, np.integer)) else out
            raw = np.load(emg_p)
            if raw.ndim != 2:
                continue
            utts.append(RawUtt(raw.astype(np.float32), segs, gc))
            gc += 1
    if not utts:
        raise RuntimeError("empty corpus")
    return utts, label_names


def windows_from(utts, cfg: D.PreprocConfig, win_sec, hop_sec, idx=None):
    """Materialize windows for a subset of utterances (by list index), honoring
    each utterance's labeled segments."""
    sel = utts if idx is None else [utts[i] for i in idx]
    Xs, ys, gs = [], [], []
    fs = cfg.fs_out
    for u in sel:
        sig, fs = D.preprocess_continuous(u.sig, cfg)
        T = sig.shape[0]
        for label, s0, s1 in u.segments:
            a, b = int(s0 * T), int(s1 * T)
            if b - a < 8:
                continue
            # fixed-length windows (short segments are padded inside window_signal)
            w = D.window_signal(sig[a:b], fs, win_sec, hop_sec, cfg)
            if len(w) == 0:
                continue
            Xs.append(w)
            ys.append(np.full(len(w), int(label), np.int64))
            gs.append(np.full(len(w), u.group, np.int64))
    if not Xs:
        return None
    L = int(round(win_sec * fs))
    Xs = [D._pad_time(w, L) for w in Xs]
    y = np.concatenate(ys)
    uniq = np.unique(y); remap = {c: i for i, c in enumerate(uniq)}
    y = np.array([remap[c] for c in y], np.int64)
    return (np.concatenate(Xs), y, np.concatenate(gs), fs)


# ----------------------------------------------------------------------------- #
#  Search spaces
# ----------------------------------------------------------------------------- #
def suggest_preproc(trial, is_eeg_path) -> tuple[D.PreprocConfig, float, float]:
    """Preprocessing + windowing HPs. EMG path keeps high band & full rate;
    EEG/CNN path allows decimation. Both are explored; the model decides which
    makes sense for it."""
    bp_low = trial.suggest_float("bp_low", 1.0, 30.0)
    bp_high = trial.suggest_float("bp_high", 80.0, 480.0)
    notch = trial.suggest_categorical("notch", [0, 50, 60])
    normalize = trial.suggest_categorical("normalize", ["zscore", "robust", "none"])
    clip = trial.suggest_categorical("clip_sigma", [0, 4, 6])
    decimate = trial.suggest_categorical("decimate_to", [0, 250, 500])
    win_sec = trial.suggest_float("win_sec", 0.25, 2.0)
    hop_sec = trial.suggest_float("hop_sec", 0.1, 1.0)
    cfg = D.PreprocConfig(
        fs_in=1000.0, bp_low=bp_low, bp_high=bp_high,
        notch=(None if notch == 0 else float(notch)),
        normalize=normalize, clip_sigma=(None if clip == 0 else float(clip)),
        decimate_to=(None if decimate == 0 else float(decimate)),
    )
    return cfg, win_sec, hop_sec


def suggest_model_hp(trial, model):
    if model == "eegnet":
        return dict(
            F1=trial.suggest_categorical("F1", [4, 8, 16]),
            D=trial.suggest_categorical("D", [1, 2, 4]),
            kernel_length=trial.suggest_categorical("kernel_length", [32, 64, 128]),
            dropout=trial.suggest_float("dropout", 0.1, 0.6),
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        )
    if model == "deepconvnet":
        return dict(
            drop_prob=trial.suggest_float("drop_prob", 0.1, 0.6),
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        )
    if model == "dda":
        hi = trial.suggest_categorical("delay_hi", [10, 16, 24, 40])
        return dict(
            delay_grid=tuple(range(1, hi)),
            classifier=trial.suggest_categorical("classifier", ["lda", "logreg"]),
        )
    if model == "cspdnn":
        return dict(
            n_select=trial.suggest_categorical("n_select", [3, 5, 7, 9]),
            reg=trial.suggest_float("reg", 0.0, 0.3),
            wavelet=trial.suggest_categorical("wavelet", ["db4", "db8", "sym5"]),
            levels=trial.suggest_categorical("levels", [3, 4, 5]),
        )
    if model == "rusnac":
        arch = trial.suggest_categorical(
            "arch", ["64/64", "64-128/64", "128-64/64", "64-128-64/64", "128-256-128/128"])
        return dict(
            domain=trial.suggest_categorical("domain", ["time", "freq"]),
            mean_filter_k=trial.suggest_categorical("mean_filter_k", [1, 3, 5]),
            normalize=trial.suggest_categorical("normalize_xcov", [True, False]),
            conv_act=trial.suggest_categorical("conv_act", ["relu", "tanh"]),
            dense_act=trial.suggest_categorical("dense_act", ["tanh", "relu"]),
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            **_arch_to_kwargs(arch),
        )
    raise ValueError(model)


def _arch_to_kwargs(arch: str) -> dict:
    conv, dense = arch.split("/")
    return dict(conv_filters=tuple(int(x) for x in conv.split("-")), dense=int(dense))


# F2 is tied to F1*D in EEGNet; set after suggestion
def _post(model, hp):
    if model == "eegnet":
        hp["F2"] = hp["F1"] * hp["D"]
    return hp


# ----------------------------------------------------------------------------- #
#  Inner objective (HP selection)  &  outer evaluation (unbiased)
# ----------------------------------------------------------------------------- #
def evaluate_config(utts, train_idx, model, cfg, win_sec, hop_sec, hp,
                    n_splits, prune_cb=None):
    """Mean balanced-accuracy over an inner GroupKFold on the TRAIN utterances."""
    groups_u = np.array([utts[i].group for i in train_idx])
    labels_u = np.array([utts[i].segments[0][0] for i in train_idx])
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups_u))))
    scores = []
    for fi, (itr, ite) in enumerate(gkf.split(train_idx, labels_u, groups_u)):
        tr = windows_from(utts, cfg, win_sec, hop_sec, [train_idx[i] for i in itr])
        te = windows_from(utts, cfg, win_sec, hop_sec, [train_idx[i] for i in ite])
        if tr is None or te is None or len(np.unique(tr[1])) < 2:
            return -1.0
        Xtr, ytr, gtr, fs = tr
        Xte, yte, _, _ = te
        n_cls = int(max(ytr.max(), yte.max())) + 1
        est = make_model(model, Xtr.shape[1], Xtr.shape[2], n_cls, **hp)
        try:
            est.fit(Xtr, ytr, groups=gtr)
        except TypeError:
            est.fit(Xtr, ytr)
        scores.append(balanced_accuracy_score(yte, est.predict(Xte)))
        if prune_cb:
            prune_cb(np.mean(scores), fi)
    return float(np.mean(scores)) if scores else -1.0


def make_objective(utts, train_idx, model, inner, is_eeg_path):
    def objective(trial):
        cfg, win_sec, hop_sec = suggest_preproc(trial, is_eeg_path)
        hp = _post(model, suggest_model_hp(trial, model))

        def prune_cb(running, step):
            trial.report(running, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score = evaluate_config(utts, train_idx, model, cfg, win_sec, hop_sec, hp,
                                inner, prune_cb)
        return score
    return objective


def run(root, model, outer, inner, trials, study_db, seed, max_utts, save_dir=None):
    from dataclasses import asdict
    import persistence
    utts, label_names = load_raw_corpus(root, max_utts_per_mode=max_utts)
    groups = np.array([u.group for u in utts])
    labels = np.array([u.segments[0][0] for u in utts])   # representative label/group
    idx_all = np.arange(len(utts))
    is_eeg_path = model in ("eegnet", "deepconvnet")  # allow decimation
    okf = GroupKFold(n_splits=outer)
    save_dir = save_dir or f"hpo_{model}_artifacts"

    results = []
    for ofold, (tr, te) in enumerate(okf.split(idx_all, labels, groups)):
        study = optuna.create_study(
            study_name=f"{model}_outer{ofold}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed + ofold),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
            storage=study_db, load_if_exists=True,
        )
        study.optimize(make_objective(utts, tr, model, inner, is_eeg_path),
                       n_trials=trials, gc_after_trial=True)
        best = study.best_params
        # ---- refit best config on ALL train utts, score on held-out OUTER test ----
        cfg, win_sec, hop_sec, hp = rebuild_from_params(model, best)
        tr_w = windows_from(utts, cfg, win_sec, hop_sec, list(tr))
        te_w = windows_from(utts, cfg, win_sec, hop_sec, list(te))
        Xtr, ytr, gtr, _ = tr_w
        Xte, yte, gte, _ = te_w
        n_cls = int(max(ytr.max(), yte.max())) + 1
        est = make_model(model, Xtr.shape[1], Xtr.shape[2], n_cls, **hp)
        try:
            est.fit(Xtr, ytr, groups=gtr)
        except TypeError:
            est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        outer_score = balanced_accuracy_score(yte, pred)

        # ---- persist: loadable model + predictions for this outer fold ----
        recipe = persistence.save_fold(
            save_dir, model, ofold, est, Xte, yte, pred, gte, label_names,
            dims=(Xtr.shape[1], Xtr.shape[2], n_cls), hp=hp,
            preproc_dict=asdict(cfg), win_sec=win_sec, hop_sec=hop_sec,
            inner_score=study.best_value, outer_score=outer_score)

        results.append(dict(outer_fold=ofold, outer_balanced_acc=outer_score,
                            best_inner=study.best_value, best_params=best,
                            model_file=recipe["save_info"].get("model_file"),
                            predictions_file=recipe["predictions_file"],
                            save_format=recipe["save_info"].get("format")))
        print(f"[{model}] outer fold {ofold}: inner*={study.best_value:.3f} "
              f"OUTER={outer_score:.3f}  (saved -> {recipe['save_info'].get('model_file')})")

    out = dict(model=model, save_dir=save_dir,
               outer_mean=float(np.mean([r["outer_balanced_acc"] for r in results])),
               outer_std=float(np.std([r["outer_balanced_acc"] for r in results])),
               folds=results)
    Path(f"hpo_{model}_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{model}] UNBIASED outer balanced-acc = "
          f"{out['outer_mean']:.3f} +/- {out['outer_std']:.3f}")
    print(f"Saved hpo_{model}_summary.json  +  per-fold models/predictions in {save_dir}/")
    return out


def rebuild_from_params(model, p):
    """Inverse of the suggest_* functions, for refit on the outer split."""
    cfg = D.PreprocConfig(
        fs_in=1000.0, bp_low=p["bp_low"], bp_high=p["bp_high"],
        notch=(None if p["notch"] == 0 else float(p["notch"])),
        normalize=p["normalize"],
        clip_sigma=(None if p["clip_sigma"] == 0 else float(p["clip_sigma"])),
        decimate_to=(None if p["decimate_to"] == 0 else float(p["decimate_to"])),
    )
    win_sec, hop_sec = p["win_sec"], p["hop_sec"]
    if model == "eegnet":
        hp = dict(F1=p["F1"], D=p["D"], F2=p["F1"] * p["D"],
                  kernel_length=p["kernel_length"], dropout=p["dropout"],
                  lr=p["lr"], weight_decay=p["weight_decay"], batch_size=p["batch_size"])
    elif model == "deepconvnet":
        hp = dict(drop_prob=p["drop_prob"], lr=p["lr"],
                  weight_decay=p["weight_decay"], batch_size=p["batch_size"])
    elif model == "dda":
        hp = dict(delay_grid=tuple(range(1, p["delay_hi"])), classifier=p["classifier"])
    elif model == "cspdnn":
        hp = dict(n_select=p["n_select"], reg=p["reg"],
                  wavelet=p["wavelet"], levels=p["levels"])
    elif model == "rusnac":
        hp = dict(domain=p["domain"], mean_filter_k=p["mean_filter_k"],
                  normalize=p["normalize_xcov"], conv_act=p["conv_act"],
                  dense_act=p["dense_act"], lr=p["lr"],
                  weight_decay=p["weight_decay"], batch_size=p["batch_size"],
                  **_arch_to_kwargs(p["arch"]))
    return cfg, win_sec, hop_sec, hp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True, choices=["eegnet", "deepconvnet", "dda", "cspdnn", "rusnac"])
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--study_db", default=None, help="e.g. sqlite:///hpo_eegnet.db")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_utts", type=int, default=None, help="cap utts/mode for quick runs")
    ap.add_argument("--save_dir", default=None,
                    help="where per-fold models + predictions go (default hpo_<model>_artifacts)")
    a = ap.parse_args()
    db = a.study_db or f"sqlite:///hpo_{a.model}.db"
    run(a.root, a.model, a.outer, a.inner, a.trials, db, a.seed, a.max_utts, a.save_dir)
