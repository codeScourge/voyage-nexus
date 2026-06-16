# Silent-speech decoding: models, HPO, and observability

A leakage-free, group-aware codebase to validate five decoders on silent-speech
data, tune them exhaustively, and interpret them. Built around your project's
reference methods.

## Default task: word / phoneme segments on fixed windows

The default labeling is now **word/phoneme segments on fixed windows** (not the
old silent-vs-voiced binary). Each utterance is cut into one segment per text
token; each segment is sliced into fixed windows (Grigore & Rusnac use
0.25/0.5/1 s) that inherit the token label; a closed `top_k` vocabulary is built
from the corpus. Windows from one utterance share a group id, so nothing leaks
across the train/test boundary.

> **Honest caveat on the proxy alignment.** The Gaddy `info.json` `chunks` field
> is *finer than word level* (e.g. text "09:48 AM" has ~3 word tokens but 32
> chunks) and its units don't match the EMG sample count, so we do **not** trust
> it for word boundaries — segments are split into `len(tokens)` equal fractions.
> This is an explicit approximation for the EMG proxy. **On your real rig the
> segment label is the exact word/phoneme from `events.csv`** (`data.window_epoched`
> slices each trial into fixed windows that inherit the trial's true label), so
> the default is exact there. `granularity="phoneme"` uses a character-level proxy
> on Gaddy and your true phoneme labels on the rig.

## TL;DR verdicts on your existing code

**`EEGNet.py` — valid, faithful to Lawhern et al. (2018). Three fixes applied in
`models.py:build_eegnet` (`EEGNetFixed`):**
1. **max-norm constraints** restored — `max_norm=1.0` on the depthwise spatial
   conv, `max_norm=0.25` on the dense layer. These are the regularizer that
   keeps EEGNet from overfitting tiny EEG sets; your version dropped them.
2. **dynamic flatten size** from a dummy forward — your `F2*(T//32)` is wrong
   whenever `T` isn't a multiple of 32.
3. **channels from data**, not hard-coded `C=64`.
   *(Canonical max-norm is applied after `optimizer.step()`; here it's clamped in
   `forward()` under `no_grad`, which is simpler and equivalent at convergence.)*

**`Pairwise_CSP.py` — faithful to Panachakel, but has a real test-leakage bug.**
`train_dnn` selects its best checkpoint on the array passed as `(Xva, yva)`, and
`dnn_fold_accuracy` passes the **test fold** there (`train_dnn(Xtr,ytr,Xte,yte,…)`).
The model early-stops on the test fold → every pairwise accuracy is optimistically
biased. Fixed in `csp_patch.py:CSPDWTDNN`: the inner validation set is carved from
the **training trials only** (split by trial so channel-as-sample rows never leak).
The CSP math (trace-normalized covariances, shrinkage, `eigh(C1,C2)`, top-|w|
selection) and the db4 DWT stack are correct and kept.

## What the Drive data actually is — and the consequences

It's the **Gaddy & Klein "Digital Voicing of Silent Speech" EMG corpus** (the two
Gaddy PDFs in your project). Verified layout, per utterance `{i}`:
`{i}_emg.npy` (float64, shape `(T, 8)` — **8 EMG channels, time-major**),
`{i}_audio[_clean].flac`, `{i}_info.json` (`book, sentence_index, text, chunks`),
`{i}_button.npy`. Folders `silent/` and `voiced/`.

Three consequences you must accept when using it as a proxy:
1. **No EEG.** It exercises only your EMG pathway. EEGNet runs over 8 EMG
   electrodes; Panachakel's *cortical* channel-selection premise is vacuous on
   EMG even though the code executes. Read results as "plumbing + EMG branch
   work," not "EEG imagined-speech decoding works."
2. **Open-vocabulary continuous sentences ⇒ no built-in label.** The four methods
   are fixed-window closed-set classifiers; "accuracy" is undefined until you pick
   a label. Default task here is **silent-vs-voiced** (`data.task_silent_vs_voiced`)
   — well-defined, uses both folders, and maps to a real question for your device.
   Switch via `data.make_task_word_set([...])` or write your own.
3. **Variable length** ⇒ everything is windowed to fixed `T`, grouped by utterance
   so windows from one utterance never cross a train/test split.

## Files

| File | What it is | Executed-tested here? |
|---|---|---|
| `data.py` | Loader for Gaddy (`load_gaddy`) + your rig (`load_rig_epoched`), tunable preprocessing, windowing, `select_streams` | ✅ yes |
| `dda.py` | Carvalho ST-DDA (`u̇=a₁u(t−τ₁)+a₂u(t−τ₂)+a₃u(t−τ₁)²`), supervised delay search, sklearn API | ✅ yes (100% on synthetic 2-dynamics) |
| `grigore_rusnac.py` | Rusnac & Grigore: channel×channel cross-covariance (time/freq) + spectral mean-filter + CNN on the C×C image | ✅ yes (features + classifier; freq>time, as in paper) |
| `csp_patch.py` | Leakage-free CSP+DWT+DNN (Torch head, sklearn-MLP fallback) | ✅ yes |
| `models.py` | `EEGNetFixed`, braindecode `Deep4Net`, `RusnacCNN`, unified `TorchClassifier`, `OvO3D`, `make_model` | ✅ yes (real Torch 2.8 + braindecode) |
| `hpo.py` | Nested group-CV Optuna search over preprocessing + model HPs; saves per-fold model + predictions | ✅ yes (full nested loop + artifacts) |
| `persistence.py` | Save/reload fitted fold models + outer-fold predictions (joblib, with Torch state_dict fallback) | ✅ yes (round-trip on dda/cspdnn/rusnac) |
| `observability.py` | 10-analysis interpretation battery | ✅ yes (each analysis run, incl. real-Torch integrated gradients) |

> **Validation status.** All five models were run end-to-end on real **Torch 2.8.0
> (CPU) + braindecode** in the sandbox: `eegnet`, `deepconvnet` (braindecode
> `Deep4Net`), `rusnac`, the `cspdnn` Torch DNN head, and `dda`. The
> `persistence` Torch round-trip (state_dict save → architecture rebuild → exact
> prediction match) and `integrated_gradients` autograd attribution were verified
> too. Running live surfaced and fixed three bugs that static checks had missed:
>   1. `models.py` EEGNet max-norm used `Tensor.norm(dim=(1,2,3))`, which routes to
>      `matrix_norm` and rejects a 3-tuple dim → switched to
>      `torch.linalg.vector_norm` (conv + linear constraints).
>   2. `persistence.py` left a truncated `.joblib` when `joblib.dump` failed on a
>      closure-defined Torch class → the partial file is now removed and `load_fold`
>      falls through to the state_dict rebuild.
>   3. `observability.py` `integrated_gradients` shadowed its `out` filename with the
>      model output tensor → renamed to `logits`.
>
> To reproduce locally: `pip install torch braindecode optuna pywavelets matplotlib`.

## Grigore & Rusnac (2022) methodology — `grigore_rusnac.py`

Faithful to *CNN Architectures and Feature Extraction Methods for EEG Imaginary
Speech Recognition* (Sensors 22:4679), which reports ~37% on the 11-class KaraOne
phoneme/word problem:

- **Preprocessing**: notch at mains + harmonics, **keep all high-frequency
  content** (no narrow bandpass). Set via `PreprocConfig(bp_low=1, bp_high=480,
  notch=60)`.
- **Segmentation**: fixed non-overlapping windows; they compare 0.25/0.5/1 s and
  find **0.25 s best** (short-term quasi-stationarity).
- **Feature**: channel×channel **cross-covariance** — either *time domain*
  (their Eq. 1; observations = time samples) or *frequency domain* (Eq. 3;
  cross-covariance of per-channel FFT magnitude — **their best**). Optional
  spectral **mean filter** (kernel 3/5; they found it slightly hurts) and
  optional normalization to a [−1,1] correlation image (matches their Fig. 1/2).
- **Classifier**: small CNN on the C×C matrix as a 1-channel image. Their best
  for frequency features: **Conv64 → Conv128 → Dense64 → softmax**, ReLU on conv,
  Tanh on dense. The full Table 3 architecture sweep is reachable through the HPO
  `arch` parameter (`64/64`, `64-128/64`, `128-64/64`, `64-128-64/64`,
  `128-256-128/128`).

`RusnacCNN` consumes the same `(n, C, T)` windows as every other model and builds
the cross-covariance image internally, so it plugs into `make_model`, `hpo.py`,
and `observability.py` with no special-casing (integrated-gradients is skipped for
it, since attribution over the raw signal doesn't apply to a covariance image).
On the 8-channel EMG proxy the feature is an 8×8 matrix; on your 16+16 rig it's
32×32 — exactly the regime the paper targets.

> Binary methods (`cspdnn`) are auto-wrapped in a 3-D-aware one-vs-one (`OvO3D`)
> for the multiclass word/phoneme default.

## How it fits together

```
raw signals ──(data.PreprocConfig: band/notch/decimate/window/normalize)──► (n, C, T)
                                   │
         ┌─────────────────────────┼──────────────────────────┐
         ▼                         ▼                          ▼
   make_model(...)            hpo.run(...)             observability.run_all(...)
   common fit/predict     nested group-CV search        permutation, occlusion,
   (eegnet/deepconvnet/    (preproc + arch + train)      IG, temporal-gen, learning
    dda/cspdnn)            → unbiased outer score        curve, dropout, stream ablation
```

Every split is **GroupKFold / GroupShuffleSplit keyed on `groups`** (utterance on
Gaddy; `collection_block_id`/session on your rig). This is the single most
important guard against the inflated numbers that plague this literature.

## Run commands

```bash
pip install torch braindecode optuna pywavelets matplotlib scikit-learn scipy

# 0. sanity-check the data layer (word/phoneme segments are the default)
python data.py --root /path/to/folder_with_silent_and_voiced --granularity word --top_k 10

# 1. exhaustive tuning (runtime-unbounded by design). One study per model:
python hpo.py --root /path/... --model eegnet      --outer 5 --inner 4 --trials 300
python hpo.py --root /path/... --model deepconvnet --outer 5 --inner 4 --trials 300
python hpo.py --root /path/... --model dda         --outer 5 --inner 4 --trials 150
python hpo.py --root /path/... --model cspdnn      --outer 5 --inner 4 --trials 200
python hpo.py --root /path/... --model rusnac      --outer 5 --inner 4 --trials 300
#   Each run writes THREE kinds of result:
#     hpo_<model>.db               resumable Optuna study (every trial, all folds)
#     hpo_<model>_summary.json     unbiased outer mean/std + per-fold best params,
#                                  with paths to the saved model & predictions
#     hpo_<model>_artifacts/       per outer fold k:
#                                    fold{k}_model.joblib   (or .state_dict.pt +
#                                       fold{k}_recipe.json for Torch nets)
#                                    fold{k}_predictions.npz  y_true,y_pred,groups,labels
#   Override the artifact dir with --save_dir.

# Reload a saved fold model and rebuild its confusion matrix WITHOUT refitting:
python -c "import persistence as P; \
est, data = P.load_fold('hpo_rusnac_artifacts', 0); \
cm, names = P.confusion_from_fold('hpo_rusnac_artifacts', 0); \
print('labels:', names); print(cm); \
print('reloaded model ready:', hasattr(est, 'predict'))"

# 2. interpret a model (uses best HPs you copy from the summary json)
python observability.py --root /path/... --model rusnac --n_perm 500 --outdir obs_rusnac
#   → obs_*/report.json + PNGs (confusion, calibration, occlusion, IG,
#     tempgen, learning_curve, channel_dropout, stream_ablation)
```

## Moving to your real 16 EEG + 16 EMG rig

1. Epoch `eeg_frames.bin` against `events.csv` into an `.npz` with keys
   `X (n,32,T)` (channels EEG0..15 then EMG0..15), `y`, `groups`
   (`collection_block_id`), `fs`, `eeg_idx`, `emg_idx`, `label_names`.
2. `ds = data.load_rig_epoched(path.npz)` — identical contract. For the default
   word/phoneme-segment task, `ds = data.window_epoched(ds, win_sec, hop_sec, cfg)`
   slices each trial into fixed windows that inherit the trial's true label.
3. `stream_ablation` now becomes the decisive **EEG-only vs EMG-only vs both**
   test — the honest measure of whether your EEG channels contribute beyond the
   articulatory EMG. This is the experiment your whole thesis rests on.

## The observability battery and what each shows

1. **permutation_test** — label-shuffle null + p-value & CI → is accuracy real?
2. **confusion_and_report** — per-class precision/recall/F1.
3. **calibration** — reliability curve + ECE → are probabilities trustworthy?
4. **occlusion_sensitivity** — zero time-bands/channels → *where/when* the
   decision lives (artifact localizer); model-agnostic.
5. **integrated_gradients** — Torch attribution over (channel, time).
6. **temporal_generalization** — train@t / test@t′ → transient vs sustained code.
7. **learning_curve** — acc vs #training groups → are you data-limited?
8. **channel_dropout** — robustness to electrode loss/shift.
9. **stream_ablation** — EEG vs EMG vs both (the confound test on your rig).
10. **dda_coefficient_space** — DDA feature separability.
