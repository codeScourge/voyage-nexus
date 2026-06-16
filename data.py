"""
data.py
=======
Unified data layer for the silent-speech decoding project.

Two sources are supported behind ONE common output contract:

  load_dataset(...) -> Dataset(
      X         : float32  (n_windows, n_channels, n_times)
      y         : int64    (n_windows,)            class labels
      groups    : int64    (n_windows,)            grouping key for leakage-free CV
      meta      : dict                             channel names, fs, label names, ...
  )

Source A — Gaddy & Klein "Digital Voicing of Silent Speech" corpus (the proxy
           data in your Drive: silent/ and voiced/ folders). EMG-ONLY, 8 ch,
           float64, time-major (T, 8). Continuous open-vocabulary sentences, so
           we DEFINE a classification task on top of it (default: silent-vs-voiced).

Source B — Your real over-ear rig (16 EEG + 16 EMG @ 1 kHz). A loader stub is
           provided that yields the SAME contract so every model / the HPO / the
           observability suite work unchanged once your epoched data exists.

WHY GROUPS MATTER: windows cut from the same utterance (or, on your rig, the same
collection_block_id / session) are highly correlated. If they straddle a
train/test split the reported accuracy is inflated. Every split in this project
is GROUP-AWARE and keyed on `groups`.

Verified against the real files: emg npy header is {'descr':'<f8',
'shape':(T,8)} and info.json carries {book, sentence_index, text, chunks}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, decimate


# ----------------------------------------------------------------------------- #
#  Container
# ----------------------------------------------------------------------------- #
@dataclass
class Dataset:
    X: np.ndarray            # (n_windows, n_channels, n_times) float32
    y: np.ndarray            # (n_windows,) int64
    groups: np.ndarray       # (n_windows,) int64
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.X.ndim == 3, f"X must be 3D, got {self.X.shape}"
        assert len(self.X) == len(self.y) == len(self.groups)
        self.X = self.X.astype(np.float32, copy=False)
        self.y = self.y.astype(np.int64, copy=False)
        self.groups = self.groups.astype(np.int64, copy=False)

    @property
    def n_channels(self) -> int: return self.X.shape[1]
    @property
    def n_times(self) -> int:    return self.X.shape[2]
    @property
    def n_classes(self) -> int:  return int(self.y.max()) + 1

    def subset(self, idx) -> "Dataset":
        return Dataset(self.X[idx], self.y[idx], self.groups[idx], dict(self.meta))


# ----------------------------------------------------------------------------- #
#  Signal preprocessing  (parameters are HPO-tunable -- see hpo.py)
# ----------------------------------------------------------------------------- #
@dataclass
class PreprocConfig:
    fs_in: float = 1000.0          # native sampling rate (Gaddy EMG & your rig)
    bp_low: float = 20.0           # EMG band default; for EEG use ~1 Hz
    bp_high: float = 450.0         # EMG band default; for EEG use ~50 Hz
    notch: Optional[float] = 60.0  # mains; None disables
    notch_q: float = 30.0
    decimate_to: Optional[float] = None   # e.g. 250.0 for the EEG/CNN path; None = keep
    normalize: str = "zscore"      # {"none","zscore","robust"} per-channel within window
    clip_sigma: Optional[float] = None    # artifact clip at +/- k*sigma after norm; None off

    @property
    def fs_out(self) -> float:
        return self.decimate_to or self.fs_in


def _design_bandpass(low, high, fs, order=4):
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    return butter(order, [low_n, high_n], btype="band", output="sos")


def preprocess_continuous(sig: np.ndarray, cfg: PreprocConfig) -> tuple[np.ndarray, float]:
    """sig: (T, C) raw -> (T', C) filtered/decimated. Returns (signal, fs_out)."""
    x = np.asarray(sig, dtype=np.float64)
    fs = cfg.fs_in
    # bandpass (zero-phase)
    sos = _design_bandpass(cfg.bp_low, cfg.bp_high, fs)
    x = sosfiltfilt(sos, x, axis=0)
    # notch
    if cfg.notch is not None and cfg.notch < fs / 2:
        b, a = iirnotch(cfg.notch, cfg.notch_q, fs)
        x = filtfilt(b, a, x, axis=0)
    # decimate
    if cfg.decimate_to is not None and cfg.decimate_to < fs:
        factor = int(round(fs / cfg.decimate_to))
        if factor > 1:
            x = decimate(x, factor, axis=0, ftype="fir", zero_phase=True)
            fs = fs / factor
    return x.astype(np.float32), fs


def _normalize_window(w: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """w: (C, T) -> per-channel normalized."""
    if cfg.normalize == "zscore":
        mu = w.mean(axis=1, keepdims=True)
        sd = w.std(axis=1, keepdims=True) + 1e-8
        w = (w - mu) / sd
    elif cfg.normalize == "robust":
        med = np.median(w, axis=1, keepdims=True)
        mad = np.median(np.abs(w - med), axis=1, keepdims=True) + 1e-8
        w = (w - med) / (1.4826 * mad)
    if cfg.clip_sigma is not None:
        w = np.clip(w, -cfg.clip_sigma, cfg.clip_sigma)
    return w


def window_signal(sig_TC: np.ndarray, fs: float, win_sec: float,
                  hop_sec: float, cfg: PreprocConfig, pad_short: bool = True) -> np.ndarray:
    """(T, C) -> (n_win, C, w_len) with per-window normalization.

    Every returned window has EXACTLY w_len = round(win_sec*fs) time samples, so a
    fixed-input model (EEGNet/DeepConvNet/Rusnac) sees one consistent length across
    train, test, and inference. Segments shorter than one window are center zero-
    padded up to w_len (pad_short=True) rather than emitted at a variable length,
    which previously caused train/test flatten-size mismatches.
    """
    T, C = sig_TC.shape
    w_len = int(round(win_sec * fs))
    hop = max(1, int(round(hop_sec * fs)))
    if T < w_len:
        if not pad_short or T < 8:
            return np.empty((0, C, w_len), dtype=np.float32)
        padded = np.zeros((w_len, C), dtype=sig_TC.dtype)
        s = (w_len - T) // 2
        padded[s:s + T] = sig_TC
        sig_TC, T = padded, w_len
    starts = range(0, T - w_len + 1, hop)
    out = np.stack([_normalize_window(sig_TC[s:s + w_len].T, cfg) for s in starts], axis=0)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- #
#  Source A: Gaddy silent/voiced corpus
# ----------------------------------------------------------------------------- #
def _list_utterances(folder: Path) -> list[int]:
    idxs = set()
    for p in folder.rglob("*_emg.npy"):
        try:
            idxs.add(int(p.name.split("_")[0]))
        except ValueError:
            continue
    return sorted(idxs)


def _find(folder: Path, idx: int, suffix: str) -> Optional[Path]:
    hits = list(folder.rglob(f"{idx}_{suffix}"))
    return hits[0] if hits else None


# ---- task definitions --------------------------------------------------------
# A task maps an utterance to labels. The generalized return contract is ONE of:
#   * None                                  -> drop this utterance
#   * int                                   -> one label for the whole utterance
#   * list[(label:int, s0:float, s1:float)] -> labeled SEGMENTS, fractional
#                                              [s0,s1) positions along the utterance
# The loader windows within each segment, so segment-level (word/phoneme) labels
# become fixed windows -- the project DEFAULT requested for this work.

import re as _re


def task_silent_vs_voiced(mode: str, info: dict):
    """0 = silent, 1 = voiced (kept available; not the default)."""
    return 0 if mode == "silent" else 1


def _tokens(text: str, granularity: str) -> list[str]:
    text = str(text).strip().lower()
    if not text:
        return []
    if granularity == "word":
        return [t for t in _re.split(r"\s+", text) if t]
    # crude phoneme proxy: alphabetic characters (no public phonemizer dependency).
    # On your real rig the label IS the phoneme/word from events.csv -- exact there.
    return [c for c in _re.sub(r"[^a-z]", "", text)]


def build_vocab(root, granularity="word", top_k=10, modes=("silent", "voiced"),
                max_utts_per_mode=None) -> dict:
    """Scan the corpus and return {token: class_id} for the top_k most frequent
    tokens, so the open-vocabulary proxy is turned into a CLOSED set."""
    from collections import Counter
    root = Path(root); cnt = Counter()
    for mode in modes:
        folder = root / mode
        if not folder.exists():
            continue
        ids = _list_utterances(folder)
        if max_utts_per_mode:
            ids = ids[:max_utts_per_mode]
        for idx in ids:
            ip = _find(folder, idx, "info.json")
            if ip is None:
                continue
            try:
                text = json.loads(ip.read_text()).get("text", "")
            except Exception:
                text = ""
            cnt.update(_tokens(text, granularity))
    vocab = {tok: i for i, (tok, _) in enumerate(cnt.most_common(top_k))}
    return vocab


def make_task_word_segments(vocab: dict, granularity="word",
                            use_chunks=True) -> Callable:
    """DEFAULT task factory: cut each utterance into equal segments, one per token
    of its text, and label each segment by that token (kept only if in `vocab`).

    NOTE on alignment: the Gaddy info.json 'chunks' field is finer than word level
    (counts do not equal token counts -- e.g. text "09:48 AM" has ~3 word tokens
    but 32 chunks), and its units differ from the EMG sample count. We therefore do
    NOT trust chunks for word boundaries; we split the utterance into len(tokens)
    EQUAL fractional segments. This is an explicit approximation for the proxy; on
    your real rig the segment label is the true word/phoneme from events.csv.
    Set use_chunks=True only if you later verify chunk<->EMG alignment."""
    def _task(mode: str, info: dict):
        toks = _tokens(info.get("text", ""), granularity)
        toks = [t for t in toks if t in vocab]
        if not toks:
            return None
        n = len(_tokens(info.get("text", ""), granularity)) or 1
        # map each kept token to its positional segment among ALL tokens
        all_toks = _tokens(info.get("text", ""), granularity)
        segs = []
        for pos, t in enumerate(all_toks):
            if t in vocab:
                segs.append((vocab[t], pos / n, (pos + 1) / n))
        return segs or None
    return _task


def load_gaddy(root: str | Path,
               cfg: PreprocConfig = PreprocConfig(),
               win_sec: float = 0.5,
               hop_sec: float = 0.25,
               task: Optional[Callable] = None,
               modes=("silent", "voiced"),
               max_utts_per_mode: Optional[int] = None,
               granularity: str = "word",
               top_k: int = 10) -> Dataset:
    """
    root/ contains silent/ and voiced/. EMG-only, 8 ch.

    DEFAULT task = word/phoneme SEGMENTS on fixed windows: each utterance is cut
    into one equal segment per text token, each segment is windowed to fixed length
    and labeled by its token (closed top_k vocabulary). Windows from one utterance
    share a group id (no cross-utterance leakage).

    Pass task=task_silent_vs_voiced (or your own) to override. A task returns None,
    an int (whole-utterance label), or a list[(label, s0, s1)] of fractional segments.
    """
    root = Path(root)
    label_names = None
    if task is None:
        vocab = build_vocab(root, granularity, top_k, modes, max_utts_per_mode)
        if not vocab:
            raise RuntimeError("Could not build a vocabulary -- no text tokens found.")
        task = make_task_word_segments(vocab, granularity)
        label_names = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]

    Xs, ys, gs = [], [], []
    group_counter = 0
    fs_out = cfg.fs_out
    for mode in modes:
        folder = root / mode
        if not folder.exists():
            raise FileNotFoundError(f"{folder} not found")
        utts = _list_utterances(folder)
        if max_utts_per_mode:
            utts = utts[:max_utts_per_mode]
        for idx in utts:
            emg_p = _find(folder, idx, "emg.npy")
            if emg_p is None:
                continue
            info_p = _find(folder, idx, "info.json")
            info = {}
            if info_p is not None:
                try:
                    info = json.loads(info_p.read_text())
                except Exception:
                    info = {}
            out = task(mode, info)
            if out is None:
                continue
            segments = [(int(out), 0.0, 1.0)] if isinstance(out, (int, np.integer)) else out
            raw = np.load(emg_p)
            if raw.ndim != 2:
                continue
            sig, fs_out = preprocess_continuous(raw, cfg)
            T = sig.shape[0]
            for label, s0, s1 in segments:
                a, b = int(s0 * T), int(s1 * T)
                if b - a < 8:
                    continue
                # window_signal pads short segments up to the fixed w_len internally,
                # so every window has the SAME length (no train/test size mismatch).
                w = window_signal(sig[a:b], fs_out, win_sec, hop_sec, cfg)
                if len(w) == 0:
                    continue
                Xs.append(w)
                ys.append(np.full(len(w), int(label), dtype=np.int64))
                gs.append(np.full(len(w), group_counter, dtype=np.int64))
            group_counter += 1

    if not Xs:
        raise RuntimeError("No windows produced -- check root / task / win_sec / top_k.")
    # all windows share the canonical length; pad as a no-op safety against rounding
    L = int(round(win_sec * fs_out))
    Xs = [_pad_time(w, L) for w in Xs]
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys); g = np.concatenate(gs)
    # relabel to contiguous 0..K-1 in case some vocab classes never appeared
    uniq = np.unique(y); remap = {c: i for i, c in enumerate(uniq)}
    y = np.array([remap[c] for c in y], dtype=np.int64)
    if label_names is not None:
        label_names = [label_names[c] for c in uniq]
    meta = dict(source="gaddy", fs=fs_out, n_emg=8, n_eeg=0,
                channel_names=[f"EMG{i+1}" for i in range(X.shape[1])],
                stream_index={"emg": list(range(X.shape[1])), "eeg": []},
                label_names=label_names or [])
    return Dataset(X, y, g, meta)


def _pad_time(w: np.ndarray, L: int) -> np.ndarray:
    """w: (n, C, t) -> (n, C, L) by zero-pad or center-truncate."""
    t = w.shape[2]
    if t == L:
        return w
    if t > L:
        s = (t - L) // 2
        return w[:, :, s:s + L]
    out = np.zeros((w.shape[0], w.shape[1], L), dtype=w.dtype)
    s = (L - t) // 2
    out[:, :, s:s + t] = w
    return out


def _infer_label_names(task):
    return ["silent", "voiced"] if task is task_silent_vs_voiced else []


# ----------------------------------------------------------------------------- #
#  Source B: your real over-ear rig (16 EEG + 16 EMG)  -- contract-compatible stub
# ----------------------------------------------------------------------------- #
def load_rig_epoched(npz_path: str | Path) -> Dataset:
    """
    Expects an .npz you build from eeg_frames.bin + events.csv with keys:
        X       (n_trials, 32, T)   channels ordered EEG0..15 then EMG0..15
        y       (n_trials,)         int label per word/class
        groups  (n_trials,)         collection_block_id or session index
        fs, label_names, eeg_idx, emg_idx
    Kept separate so EEG/EMG stream-ablation (observability.py) works on your rig.
    """
    d = np.load(npz_path, allow_pickle=True)
    meta = dict(source="rig", fs=float(d["fs"]),
                n_eeg=len(d["eeg_idx"]), n_emg=len(d["emg_idx"]),
                stream_index={"eeg": list(d["eeg_idx"]), "emg": list(d["emg_idx"])},
                channel_names=list(d.get("channel_names", [])),
                label_names=list(d.get("label_names", [])))
    return Dataset(d["X"], d["y"], d["groups"], meta)


def window_epoched(ds: Dataset, win_sec: float, hop_sec: float,
                   cfg: PreprocConfig) -> Dataset:
    """Turn trial-epoched rig data into word/phoneme SEGMENTS on fixed windows:
    each trial (already a single word/phoneme from events.csv) is sliced into fixed
    windows that inherit the trial's label; group = trial index (leakage-free).
    This is the exact realization of the requested default on your real rig."""
    fs = ds.meta.get("fs", cfg.fs_in)
    Xs, ys, gs = [], [], []
    for ti in range(len(ds.X)):
        sig = ds.X[ti].T                      # (T, C)
        w = window_signal(sig, fs, win_sec, hop_sec, cfg)
        if len(w) == 0:
            continue
        Xs.append(w)
        ys.append(np.full(len(w), ds.y[ti], np.int64))
        gs.append(np.full(len(w), ds.groups[ti], np.int64))
    out = Dataset(np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), dict(ds.meta))
    return out


def select_streams(ds: Dataset, streams=("eeg", "emg")) -> Dataset:
    """Return a copy keeping only EEG, only EMG, or both -- for stream ablation."""
    idx = []
    for s in streams:
        idx += list(ds.meta.get("stream_index", {}).get(s, []))
    idx = sorted(set(int(i) for i in idx))
    if not idx:
        raise ValueError(f"No channels for streams={streams}")
    out = ds.subset(slice(None))
    out.X = ds.X[:, idx, :]
    out.meta = dict(ds.meta)
    out.meta["selected_streams"] = streams
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing silent/ and voiced/")
    ap.add_argument("--win_sec", type=float, default=0.5)
    ap.add_argument("--hop_sec", type=float, default=0.25)
    ap.add_argument("--granularity", default="word", choices=["word", "phoneme"])
    ap.add_argument("--top_k", type=int, default=10)
    a = ap.parse_args()
    # Grigore & Rusnac preprocessing: notch only, keep all high-frequency content.
    cfg = PreprocConfig(bp_low=1.0, bp_high=480.0, notch=60.0, decimate_to=None)
    ds = load_gaddy(a.root, cfg, a.win_sec, a.hop_sec,
                    granularity=a.granularity, top_k=a.top_k)
    print(f"X={ds.X.shape} y={np.bincount(ds.y)} classes={ds.n_classes} "
          f"groups={len(np.unique(ds.groups))} fs={ds.meta['fs']} "
          f"labels={ds.meta['label_names']}")
