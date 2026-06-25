from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from _preprocessors import (
    BandpassConfig,
    DEFAULT_EEG_BANDPASS_CONFIG,
    DEFAULT_EMG_BANDPASS_CONFIG,
    DEFAULT_LINE_NOISE_CONFIG,
    LineNoiseConfig,
    preprocess_session_channels,
)

# Include negative-label classes when building windows and label maps.
INCLUDE_UNKNOWN_WORD_LABEL = False
INCLUDE_SILENCE_LABEL = True
INCLUDE_TRANSITION_LABELS = False

# --- scramble-breaks transition sampling (edit these)
SCRAMBLE_BREAKS_SHIFTS_PER_TRANSITION = 3
SCRAMBLE_BREAKS_SHIFT_MIN_S = -1.2
SCRAMBLE_BREAKS_SHIFT_MAX_S = 1.2
SCRAMBLE_BREAKS_DOMINANT_FRACTION = 0.75

EEG_RECORD_FORMAT = "<QQ32f"
EEG_RECORD_FORMAT_CODES_LEGACY = "<QQ32i"

# Match client/app.py — duration of each silent_speech_word "say" window.
COLLECTION_SAY_S = 1.6

TARGET_WORDS: tuple[str, ...] = (
    "highlight",
    "bullshit",
    "gogogo",
    "shitbull",
    "hangar",
    "teaspoon",
    "naan",
    "quail"
)
UNKNOWN_WORD_LABEL = "unknown word"
SILENCE_LABEL = "silence"
WORD_STARTING_LABEL = "word starting"
WORD_ENDING_LABEL = "word ending"
SCRAMBLE_BREAKS_MODE = "scramble-breaks"
SCRAMBLE_BREAKS_BLOCK_START_EVENT = "silent_speech_scramble_start"


def all_labels() -> tuple[str, ...]:
    labels: list[str] = list(TARGET_WORDS)
    if INCLUDE_TRANSITION_LABELS:
        labels.extend([WORD_STARTING_LABEL, WORD_ENDING_LABEL])
    if INCLUDE_UNKNOWN_WORD_LABEL:
        labels.append(UNKNOWN_WORD_LABEL)
    if INCLUDE_SILENCE_LABEL:
        labels.append(SILENCE_LABEL)
    return tuple(labels)


ALL_LABELS: tuple[str, ...] = all_labels()
SILENT_SPEECH_WORD_EVENT = "silent_speech_word"
NEGATIVE_LABELS_BLOCK_START_EVENT = "silent_speech_block_start"
NEGATIVE_LABELS_BLOCK_END_EVENT = "silent_speech_block_end"

EEG_RECORD_DTYPE = np.dtype(
    [
        ("sample_index", "<u8"),
        ("mcu_time_us", "<u8"),
        ("channels", "<f4", (32,)),
    ]
)
EEG_RECORD_DTYPE_CODES_LEGACY = np.dtype(
    [
        ("sample_index", "<u8"),
        ("mcu_time_us", "<u8"),
        ("channels", "<i4", (32,)),
    ]
)


def eeg_record_dtype(meta: dict[str, Any]) -> np.dtype:
    fmt = str(meta.get("eeg_record_format", EEG_RECORD_FORMAT))
    if fmt == EEG_RECORD_FORMAT:
        return EEG_RECORD_DTYPE
    if fmt == EEG_RECORD_FORMAT_CODES_LEGACY:
        return EEG_RECORD_DTYPE_CODES_LEGACY
    raise ValueError(f"Unsupported eeg_record_format: {fmt!r}")


def _lsb_uv_from_meta(meta: dict[str, Any]) -> tuple[float, float]:
    cal = meta.get("ads_calibration")
    if isinstance(cal, dict) and "eeg_lsb_uv" in cal and "emg_lsb_uv" in cal:
        return float(cal["eeg_lsb_uv"]), float(cal["emg_lsb_uv"])
    # Defaults aligned with host/_protocol.py when metadata has no calibration block.
    vref = 4.5
    lsb = lambda gain: (2.0 * vref / gain) / ((2**24) - 1) * 1e6
    return lsb(12.0), lsb(12.0)


def frames_channels_uv(frames: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Return (n_frames, 32) channel matrix in microvolts."""
    channels = frames["channels"]
    if meta.get("channel_units") == "uV" or meta.get("eeg_record_format", EEG_RECORD_FORMAT) == EEG_RECORD_FORMAT:
        return channels.astype(np.float32, copy=False)
    eeg_lsb, emg_lsb = _lsb_uv_from_meta(meta)
    out = channels.astype(np.float32, copy=True)
    out[:, :16] *= np.float32(eeg_lsb)
    out[:, 16:] *= np.float32(emg_lsb)
    return out


@dataclass(frozen=True, slots=True)
class SessionChannels:
    session_dir: Path
    channels: np.ndarray
    sample_indices: np.ndarray
    sample_rate_hz: float
    meta: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.channels.shape[0])


def load_session_channels(
    session_dir: Path,
    *,
    line_noise: Optional[LineNoiseConfig] = DEFAULT_LINE_NOISE_CONFIG,
    eeg_bandpass: Optional[BandpassConfig] = DEFAULT_EEG_BANDPASS_CONFIG,
    emg_bandpass: Optional[BandpassConfig] = DEFAULT_EMG_BANDPASS_CONFIG,
    filter_order: int = 4,
) -> SessionChannels:
    """Load session channels in µV with line-noise notches and EEG/EMG band-pass."""
    session_dir = Path(session_dir)
    meta = load_session_meta(session_dir)
    frames = load_eeg_frames(session_dir)
    fs = float(meta["sample_rate_hz"])
    channels = frames_channels_uv(frames, meta)
    if (
        line_noise is not None
        or eeg_bandpass is not None
        or emg_bandpass is not None
    ):
        channels = preprocess_session_channels(
            channels,
            fs,
            line_noise=line_noise,
            eeg=eeg_bandpass,
            emg=emg_bandpass,
            order=filter_order,
        )
    return SessionChannels(
        session_dir=session_dir,
        channels=channels,
        sample_indices=frames["sample_index"].astype(np.int64),
        sample_rate_hz=fs,
        meta=meta,
    )


def load_session_meta(session_dir: Path) -> dict[str, Any]:
    meta_path = session_dir / "session_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing session metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def default_channel_names(channel_count: int = 32) -> tuple[str, ...]:
    return tuple(
        f"EEG{i + 1}" if i < 16 else f"EMG{i - 15}"
        for i in range(channel_count)
    )


def channel_names_from_meta(meta: dict[str, Any]) -> tuple[str, ...]:
    channel_count = int(meta.get("channel_count", 32))
    raw = meta.get("channel_order")
    if not isinstance(raw, dict):
        return default_channel_names(channel_count)

    if all(str(key).isdigit() for key in raw):
        return tuple(str(raw[str(i)]) for i in range(channel_count))

    return default_channel_names(channel_count)


def load_channel_names(session_dir: Path) -> tuple[str, ...]:
    return channel_names_from_meta(load_session_meta(session_dir))


def load_eeg_frames(session_dir: Path) -> np.ndarray:
    eeg_path = session_dir / "eeg_frames.bin"
    if not eeg_path.exists():
        raise FileNotFoundError(f"Missing EEG frame file: {eeg_path}")
    meta = load_session_meta(session_dir)
    dtype = eeg_record_dtype(meta)
    frames = np.fromfile(eeg_path, dtype=dtype)
    if frames.size == 0:
        raise RuntimeError(f"No EEG frames in session {session_dir}")
    return frames


def load_events(session_dir: Path) -> list[dict[str, str]]:
    events_path = session_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    with events_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_event_payload(event: dict[str, str]) -> dict[str, Any]:
    raw = event.get("payload_json", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def default_label_to_idx() -> dict[str, int]:
    return {label: idx for idx, label in enumerate(all_labels())}


def normalize_word_label(label: str) -> Optional[str]:
    word = label.strip().lower()
    if not word:
        return None
    if word in TARGET_WORDS:
        return word
    return UNKNOWN_WORD_LABEL


def _event_sample_end(
    event: dict[str, str],
    start_idx: int,
    start_row: int,
    *,
    fs: float,
    index_to_row: dict[int, int],
) -> tuple[int, int]:
    end_text = event.get("sample_index_end", "")
    if end_text:
        end_idx = int(end_text)
        end_row = index_to_row.get(end_idx)
        if end_row is None:
            end_row = start_row + max(0, end_idx - start_idx)
    else:
        duration_samples = int(round(COLLECTION_SAY_S * fs))
        end_idx = start_idx + duration_samples
        end_row = start_row + duration_samples
    return end_idx, end_row


def _extract_channel_window(
    channels: np.ndarray,
    start_row: int,
    end_row: int,
    *,
    pre_samples: int,
    post_samples: int,
) -> Optional[np.ndarray]:
    if end_row < start_row:
        return None
    start = start_row - pre_samples
    end = end_row + post_samples + 1
    if start < 0 or end > channels.shape[0]:
        return None
    window = channels[start:end, :]
    if window.shape[0] == 0:
        return None
    return window


def _negative_labels_blocks(events: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    pending: dict[str, int] = {}
    blocks: list[dict[str, Any]] = []
    for event in events:
        payload = parse_event_payload(event)
        block_id = str(payload.get("collection_block_id", "")).strip()
        if not block_id:
            continue
        event_type = event.get("event_type", "")
        start_text = event.get("sample_index_start", "")
        if not start_text:
            continue
        if (
            event_type == NEGATIVE_LABELS_BLOCK_START_EVENT
            and payload.get("mode") == "negative_labels"
        ):
            pending[block_id] = int(start_text)
            continue
        if (
            event_type == NEGATIVE_LABELS_BLOCK_END_EVENT
            and payload.get("mode") == "negative_labels"
        ):
            block_start = pending.pop(block_id, None)
            if block_start is None:
                continue
            block_end = int(start_text)
            if block_end <= block_start:
                continue
            blocks.append(
                {
                    "block_id": block_id,
                    "start_idx": block_start,
                    "end_idx": block_end,
                }
            )
    return blocks


def _negative_labels_word_spans(
    events: Sequence[dict[str, str]],
    block_id: str,
    *,
    fs: float,
    index_to_row: dict[int, int],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for event in events:
        if event.get("event_type", "") != SILENT_SPEECH_WORD_EVENT:
            continue
        payload = parse_event_payload(event)
        if str(payload.get("collection_block_id", "")).strip() != block_id:
            continue
        start_text = event.get("sample_index_start", "")
        if not start_text:
            continue
        start_idx = int(start_text)
        start_row = index_to_row.get(start_idx)
        if start_row is None:
            continue
        end_idx, _end_row = _event_sample_end(
            event,
            start_idx,
            start_row,
            fs=fs,
            index_to_row=index_to_row,
        )
        if end_idx <= start_idx:
            continue
        spans.append((start_idx, end_idx))
    spans.sort(key=lambda span: span[0])
    return spans


def _negative_labels_silence_spans(
    block_start_idx: int,
    block_end_idx: int,
    word_spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cursor = block_start_idx
    for word_start, word_end in word_spans:
        if word_start > cursor:
            gaps.append((cursor, word_start))
        cursor = max(cursor, word_end)
    if block_end_idx > cursor:
        gaps.append((cursor, block_end_idx))
    return [(start, end) for start, end in gaps if end > start]


def _append_silence_windows(
    *,
    channels: np.ndarray,
    events: Sequence[dict[str, str]],
    index_to_row: dict[int, int],
    fs: float,
    pre_samples: int,
    post_samples: int,
    raw_windows: list[np.ndarray],
    labels: list[str],
    event_type_list: list[str],
    event_ids: list[str],
    center_samples: list[int],
    skipped: int,
) -> int:
    for block in _negative_labels_blocks(events):
        block_id = str(block["block_id"])
        word_spans = _negative_labels_word_spans(
            events,
            block_id,
            fs=fs,
            index_to_row=index_to_row,
        )
        for gap_start, gap_end in _negative_labels_silence_spans(
            int(block["start_idx"]),
            int(block["end_idx"]),
            word_spans,
        ):
            start_row = index_to_row.get(gap_start)
            if start_row is None:
                skipped += 1
                continue
            end_row = index_to_row.get(gap_end)
            if end_row is None:
                end_row = start_row + max(0, gap_end - gap_start)
            window = _extract_channel_window(
                channels,
                start_row,
                end_row,
                pre_samples=pre_samples,
                post_samples=post_samples,
            )
            if window is None:
                skipped += 1
                continue
            center = gap_start + (gap_end - gap_start) // 2
            raw_windows.append(window)
            labels.append(SILENCE_LABEL)
            event_type_list.append("negative_labels_silence")
            event_ids.append(f"silence:{block_id}:{gap_start}-{gap_end}")
            center_samples.append(center)
    return skipped


def _stable_event_seed(event_id: str) -> int:
    digest = hashlib.md5(event_id.encode(), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16)


def _interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _transition_phase_fractions(*, shift_s: float, window_s: float, kind: str) -> tuple[float, float]:
    """Fraction of a transition-centered window in silence vs word regions."""
    half = window_s / 2.0
    win_start = shift_s - half
    win_end = shift_s + half
    if kind == "silence_to_word":
        silence_len = _interval_overlap(win_start, win_end, -1e9, 0.0)
        word_len = _interval_overlap(win_start, win_end, 0.0, 1e9)
    else:
        word_len = _interval_overlap(win_start, win_end, -1e9, 0.0)
        silence_len = _interval_overlap(win_start, win_end, 0.0, 1e9)
    total = max(window_s, silence_len + word_len)
    if total <= 0.0:
        return 0.5, 0.5
    return silence_len / total, word_len / total


def _transition_shift_label(
    *,
    kind: str,
    word: str,
    silence_frac: float,
    word_frac: float,
) -> str:
    if silence_frac >= SCRAMBLE_BREAKS_DOMINANT_FRACTION:
        return SILENCE_LABEL
    if word_frac >= SCRAMBLE_BREAKS_DOMINANT_FRACTION:
        return word
    if kind == "silence_to_word":
        return WORD_STARTING_LABEL
    return WORD_ENDING_LABEL


def _scramble_breaks_transition_shifts_s(boundary_key: str) -> tuple[float, ...]:
    rng = random.Random(_stable_event_seed(boundary_key))
    return tuple(
        rng.uniform(SCRAMBLE_BREAKS_SHIFT_MIN_S, SCRAMBLE_BREAKS_SHIFT_MAX_S)
        for _ in range(SCRAMBLE_BREAKS_SHIFTS_PER_TRANSITION)
    )


def _scramble_breaks_blocks(events: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    pending: dict[str, int] = {}
    blocks: list[dict[str, Any]] = []
    for event in events:
        payload = parse_event_payload(event)
        block_id = str(payload.get("collection_block_id", "")).strip()
        if not block_id:
            continue
        event_type = event.get("event_type", "")
        start_text = event.get("sample_index_start", "")
        if not start_text:
            continue
        if (
            event_type == SCRAMBLE_BREAKS_BLOCK_START_EVENT
            and payload.get("mode") == SCRAMBLE_BREAKS_MODE
        ):
            pending[block_id] = int(start_text)
            continue
        if (
            event_type == NEGATIVE_LABELS_BLOCK_END_EVENT
            and payload.get("mode") == SCRAMBLE_BREAKS_MODE
        ):
            block_start = pending.pop(block_id, None)
            if block_start is None:
                continue
            block_end = int(start_text)
            if block_end <= block_start:
                continue
            blocks.append(
                {
                    "block_id": block_id,
                    "start_idx": block_start,
                    "end_idx": block_end,
                }
            )
    return blocks


def _scramble_breaks_word_spans(
    events: Sequence[dict[str, str]],
    block_id: str,
    *,
    fs: float,
    index_to_row: dict[int, int],
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for event in events:
        if event.get("event_type", "") != SILENT_SPEECH_WORD_EVENT:
            continue
        payload = parse_event_payload(event)
        if str(payload.get("collection_block_id", "")).strip() != block_id:
            continue
        if payload.get("mode") != SCRAMBLE_BREAKS_MODE:
            continue
        word = normalize_word_label(event.get("label_text", ""))
        if word is None or word == UNKNOWN_WORD_LABEL:
            continue
        start_text = event.get("sample_index_start", "")
        if not start_text:
            continue
        start_idx = int(start_text)
        start_row = index_to_row.get(start_idx)
        if start_row is None:
            continue
        end_idx, _end_row = _event_sample_end(
            event,
            start_idx,
            start_row,
            fs=fs,
            index_to_row=index_to_row,
        )
        if end_idx <= start_idx:
            continue
        spans.append((start_idx, end_idx, word))
    spans.sort(key=lambda span: span[0])
    return spans


def _scramble_breaks_transitions(
    word_spans: Sequence[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for start_idx, end_idx, word in word_spans:
        transitions.append(
            {
                "boundary_idx": start_idx,
                "kind": "silence_to_word",
                "word": word,
            }
        )
        transitions.append(
            {
                "boundary_idx": end_idx,
                "kind": "word_to_silence",
                "word": word,
            }
        )
    return transitions


def _extract_centered_transition_window(
    channels: np.ndarray,
    boundary_row: int,
    *,
    shift_samples: int,
    window_half_samples: int,
    pre_samples: int,
    post_samples: int,
) -> Optional[np.ndarray]:
    center_row = boundary_row + shift_samples
    return _extract_channel_window(
        channels,
        center_row - window_half_samples,
        center_row + window_half_samples,
        pre_samples=pre_samples,
        post_samples=post_samples,
    )


def _append_scramble_breaks_transition_windows(
    *,
    channels: np.ndarray,
    events: Sequence[dict[str, str]],
    index_to_row: dict[int, int],
    fs: float,
    pre_samples: int,
    post_samples: int,
    raw_windows: list[np.ndarray],
    labels: list[str],
    event_type_list: list[str],
    event_ids: list[str],
    center_samples: list[int],
) -> int:
    skipped = 0
    window_s = COLLECTION_SAY_S
    window_half_samples = max(1, int(round(window_s * fs / 2.0)))

    for block in _scramble_breaks_blocks(events):
        block_id = str(block["block_id"])
        word_spans = _scramble_breaks_word_spans(
            events,
            block_id,
            fs=fs,
            index_to_row=index_to_row,
        )
        for transition in _scramble_breaks_transitions(word_spans):
            boundary_idx = int(transition["boundary_idx"])
            boundary_row = index_to_row.get(boundary_idx)
            if boundary_row is None:
                skipped += SCRAMBLE_BREAKS_SHIFTS_PER_TRANSITION
                continue
            kind = str(transition["kind"])
            word = str(transition["word"])
            boundary_key = f"{block_id}:{kind}:{boundary_idx}:{word}"
            for shift_s in _scramble_breaks_transition_shifts_s(boundary_key):
                shift_samples = int(round(shift_s * fs))
                window = _extract_centered_transition_window(
                    channels,
                    boundary_row,
                    shift_samples=shift_samples,
                    window_half_samples=window_half_samples,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                )
                if window is None:
                    skipped += 1
                    continue
                silence_frac, word_frac = _transition_phase_fractions(
                    shift_s=shift_s,
                    window_s=window_s,
                    kind=kind,
                )
                label = _transition_shift_label(
                    kind=kind,
                    word=word,
                    silence_frac=silence_frac,
                    word_frac=word_frac,
                )
                center = boundary_idx + shift_samples
                raw_windows.append(window)
                labels.append(label)
                event_type_list.append("scramble_breaks_transition")
                event_ids.append(f"{boundary_key}:shift={shift_s:+.3f}")
                center_samples.append(center)
    return skipped


def _append_scramble_breaks_silence_windows(
    *,
    channels: np.ndarray,
    events: Sequence[dict[str, str]],
    index_to_row: dict[int, int],
    fs: float,
    pre_samples: int,
    post_samples: int,
    raw_windows: list[np.ndarray],
    labels: list[str],
    event_type_list: list[str],
    event_ids: list[str],
    center_samples: list[int],
) -> int:
    skipped = 0
    for block in _scramble_breaks_blocks(events):
        block_id = str(block["block_id"])
        word_spans = _scramble_breaks_word_spans(
            events,
            block_id,
            fs=fs,
            index_to_row=index_to_row,
        )
        span_pairs = [(start_idx, end_idx) for start_idx, end_idx, _word in word_spans]
        for gap_start, gap_end in _negative_labels_silence_spans(
            int(block["start_idx"]),
            int(block["end_idx"]),
            span_pairs,
        ):
            start_row = index_to_row.get(gap_start)
            if start_row is None:
                skipped += 1
                continue
            end_row = index_to_row.get(gap_end)
            if end_row is None:
                end_row = start_row + max(0, gap_end - gap_start)
            window = _extract_channel_window(
                channels,
                start_row,
                end_row,
                pre_samples=pre_samples,
                post_samples=post_samples,
            )
            if window is None:
                skipped += 1
                continue
            center = gap_start + (gap_end - gap_start) // 2
            raw_windows.append(window)
            labels.append(SILENCE_LABEL)
            event_type_list.append("scramble_breaks_silence")
            event_ids.append(f"silence:{block_id}:{gap_start}-{gap_end}")
            center_samples.append(center)
    return skipped


def window_sample_counts(
    sample_rate_hz: float,
    pre_ms: float,
    post_ms: float,
) -> tuple[int, int, int]:
    pre_samples = int(round((pre_ms / 1000.0) * sample_rate_hz))
    post_samples = int(round((post_ms / 1000.0) * sample_rate_hz))
    window_len = pre_samples + post_samples + 1
    return pre_samples, post_samples, window_len


@dataclass(frozen=True, slots=True)
class EventWindowBatch:
    x: np.ndarray
    labels: tuple[str, ...]
    event_types: tuple[str, ...]
    event_ids: tuple[str, ...]
    center_sample_index: np.ndarray
    session_dirs: tuple[Path, ...]
    sample_rate_hz: float
    pre_samples: int
    post_samples: int
    skipped: int

    @property
    def window_len(self) -> int:
        return int(self.x.shape[1]) if self.x.ndim == 3 else 0

    @property
    def channel_count(self) -> int:
        return int(self.x.shape[2]) if self.x.ndim == 3 else 32


def _fix_windows_to_length(
    windows: Union[np.ndarray, Sequence[np.ndarray]],
    target_len: int,
    *,
    n_ch: Optional[int] = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center-crop/pad windows to a common target_len.

    Accepts either a stacked (n, time, channels) array or a list of (time, channels)
    arrays with possibly different lengths.
    """
    if isinstance(windows, np.ndarray):
        if windows.ndim != 3:
            raise ValueError(f"expected (n, time, channels), got {windows.shape}")
        if windows.shape[1] == target_len:
            return windows
        raw: Sequence[np.ndarray] = [windows[i] for i in range(windows.shape[0])]
        n_ch = int(windows.shape[2])
    else:
        if not windows:
            raise ValueError("windows must not be empty")
        raw = windows
        if n_ch is None:
            n_ch = int(raw[0].shape[1])

    fixed = np.full((len(raw), target_len, n_ch), pad_value, dtype=np.float32)
    for i, w in enumerate(raw):
        L = w.shape[0]
        if L >= target_len:
            off = (L - target_len) // 2
            fixed[i] = w[off : off + target_len, :]
        else:
            off = (target_len - L) // 2
            fixed[i, off : off + L, :] = w
    return fixed


def build_event_windows(
    session_dir: Path,
    *,
    pre_ms: float = 0.0,
    post_ms: float = 0.0,
    target_len: Optional[int] = None,
    pad_value: float = 0.0,
    event_types: Optional[set[str]] = None,
    line_noise: Optional[LineNoiseConfig] = DEFAULT_LINE_NOISE_CONFIG,
    eeg_bandpass: Optional[BandpassConfig] = DEFAULT_EEG_BANDPASS_CONFIG,
    emg_bandpass: Optional[BandpassConfig] = DEFAULT_EMG_BANDPASS_CONFIG,
    filter_order: int = 4,
) -> EventWindowBatch:
    """Build fixed-length windows from labelled events.

    Events with sample_index_end use the start..end span (+ optional pre/post padding).
    When end is missing, infer start + COLLECTION_SAY_S using the session sample rate.
    All windows are center-cropped/padded to a common target_len.
    """
    session_dir = Path(session_dir)
    meta = load_session_meta(session_dir)
    events = load_events(session_dir)

    session_channels = load_session_channels(
        session_dir,
        line_noise=line_noise,
        eeg_bandpass=eeg_bandpass,
        emg_bandpass=emg_bandpass,
        filter_order=filter_order,
    )
    fs = session_channels.sample_rate_hz
    channels = session_channels.channels
    sample_indices = session_channels.sample_indices
    pre_samples = int(round((pre_ms / 1000.0) * fs))
    post_samples = int(round((post_ms / 1000.0) * fs))

    index_to_row = {int(idx): row for row, idx in enumerate(sample_indices)}

    raw_windows: list[np.ndarray] = []
    labels: list[str] = []
    event_type_list: list[str] = []
    event_ids: list[str] = []
    center_samples: list[int] = []
    skipped = 0

    for event in events:
        event_type = event.get("event_type", "")
        if event_types is not None and event_type not in event_types:
            continue
        if event_type != SILENT_SPEECH_WORD_EVENT:
            continue

        payload = parse_event_payload(event)
        if payload.get("mode") == SCRAMBLE_BREAKS_MODE and INCLUDE_TRANSITION_LABELS:
            continue

        label = normalize_word_label(event.get("label_text", ""))
        if label is None:
            skipped += 1
            continue
        if label == UNKNOWN_WORD_LABEL and not INCLUDE_UNKNOWN_WORD_LABEL:
            skipped += 1
            continue

        start_text = event.get("sample_index_start", "")
        if not start_text:
            skipped += 1
            continue

        start_idx = int(start_text)
        start_row = index_to_row.get(start_idx)
        if start_row is None:
            skipped += 1
            continue

        end_idx, end_row = _event_sample_end(
            event,
            start_idx,
            start_row,
            fs=fs,
            index_to_row=index_to_row,
        )
        window = _extract_channel_window(
            channels,
            start_row,
            end_row,
            pre_samples=pre_samples,
            post_samples=post_samples,
        )
        if window is None:
            skipped += 1
            continue

        raw_windows.append(window)
        labels.append(label)
        event_type_list.append(event_type)
        event_ids.append(event.get("event_id", ""))
        center_samples.append(start_idx)

    if INCLUDE_TRANSITION_LABELS:
        skipped += _append_scramble_breaks_transition_windows(
            channels=channels,
            events=events,
            index_to_row=index_to_row,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            raw_windows=raw_windows,
            labels=labels,
            event_type_list=event_type_list,
            event_ids=event_ids,
            center_samples=center_samples,
        )
    elif INCLUDE_SILENCE_LABEL:
        skipped += _append_scramble_breaks_silence_windows(
            channels=channels,
            events=events,
            index_to_row=index_to_row,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            raw_windows=raw_windows,
            labels=labels,
            event_type_list=event_type_list,
            event_ids=event_ids,
            center_samples=center_samples,
        )

    if INCLUDE_SILENCE_LABEL:
        skipped = _append_silence_windows(
            channels=channels,
            events=events,
            index_to_row=index_to_row,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            raw_windows=raw_windows,
            labels=labels,
            event_type_list=event_type_list,
            event_ids=event_ids,
            center_samples=center_samples,
            skipped=skipped,
        )

    if raw_windows:
        if target_len is None:
            # median is more robust than max when lengths vary slightly
            target_len = int(np.median([w.shape[0] for w in raw_windows]))
            target_len = max(target_len, 1)
        n_ch = channels.shape[1]
        x = _fix_windows_to_length(
            raw_windows,
            target_len,
            n_ch=n_ch,
            pad_value=pad_value,
        )
    else:
        target_len = target_len or 1
        x = np.zeros((0, target_len, channels.shape[1]), dtype=np.float32)

    return EventWindowBatch(
        x=x,
        labels=tuple(labels),
        event_types=tuple(event_type_list),
        event_ids=tuple(event_ids),
        center_sample_index=np.asarray(center_samples, dtype=np.int64),
        session_dirs=(session_dir,),
        sample_rate_hz=fs,
        pre_samples=pre_samples,
        post_samples=post_samples,
        skipped=skipped,
    )

def _merge_batches(batches: Sequence[EventWindowBatch]) -> EventWindowBatch:
    if not batches:
        raise ValueError("At least one session batch is required")

    sample_rate_hz = batches[0].sample_rate_hz
    pre_samples = batches[0].pre_samples
    post_samples = batches[0].post_samples
    for batch in batches[1:]:
        if batch.sample_rate_hz != sample_rate_hz:
            raise ValueError("All sessions must share the same sample_rate_hz")
        if batch.pre_samples != pre_samples or batch.post_samples != post_samples:
            raise ValueError("All sessions must use the same pre_ms/post_ms window")

    if len(batches) == 1:
        return batches[0]

    target_len = max(batch.window_len for batch in batches)
    aligned_x = [
        _fix_windows_to_length(batch.x, target_len)
        if batch.window_len != target_len
        else batch.x
        for batch in batches
    ]

    return EventWindowBatch(
        x=np.concatenate(aligned_x, axis=0),
        labels=tuple(label for batch in batches for label in batch.labels),
        event_types=tuple(event_type for batch in batches for event_type in batch.event_types),
        event_ids=tuple(event_id for batch in batches for event_id in batch.event_ids),
        center_sample_index=np.concatenate(
            [batch.center_sample_index for batch in batches],
            axis=0,
        ),
        session_dirs=tuple(
            session_dir for batch in batches for session_dir in batch.session_dirs
        ),
        sample_rate_hz=sample_rate_hz,
        pre_samples=pre_samples,
        post_samples=post_samples,
        skipped=sum(batch.skipped for batch in batches),
    )


class SessionEventDataset(Dataset):
    """PyTorch dataset of EEG windows cut around labelled session events."""

    def __init__(
        self,
        session_dirs: Union[Path, str, Sequence[Union[Path, str]]],
        *,
        pre_ms: float = 300.0,
        post_ms: float = 700.0,
        event_types: Optional[set[str]] = None,
        line_noise: Optional[LineNoiseConfig] = DEFAULT_LINE_NOISE_CONFIG,
        eeg_bandpass: Optional[BandpassConfig] = DEFAULT_EEG_BANDPASS_CONFIG,
        emg_bandpass: Optional[BandpassConfig] = DEFAULT_EMG_BANDPASS_CONFIG,
        filter_order: int = 4,
        label_to_idx: Optional[dict[str, int]] = None,
        dtype: torch.dtype = torch.float32,
        channel_first: bool = False,
        show_progress: bool = False,
    ) -> None:
        if isinstance(session_dirs, (str, Path)):
            dirs = [Path(session_dirs)]
        else:
            dirs = [Path(path) for path in session_dirs]
        if not dirs:
            raise ValueError("session_dirs must not be empty")

        batches: list[EventWindowBatch] = []
        progress = tqdm(
            dirs,
            desc="Building event windows",
            unit="session",
            disable=not show_progress,
        )
        build_started = time.perf_counter()
        for session_dir in progress:
            session_started = time.perf_counter()
            batch = build_event_windows(
                session_dir,
                pre_ms=pre_ms,
                post_ms=post_ms,
                event_types=event_types,
                line_noise=line_noise,
                eeg_bandpass=eeg_bandpass,
                emg_bandpass=emg_bandpass,
                filter_order=filter_order,
            )
            batches.append(batch)
            session_elapsed = time.perf_counter() - session_started
            progress.set_postfix(
                windows=len(batch.labels),
                skipped=batch.skipped,
                last_s=f"{session_elapsed:.1f}",
            )
        self.build_elapsed_s = time.perf_counter() - build_started
        batch = _merge_batches(batches)

        self._batch = batch
        self._label_to_idx = label_to_idx
        self._dtype = dtype
        self._channel_first = channel_first

        per_event_sessions: list[Path] = []
        for session_batch in batches:
            per_event_sessions.extend([session_batch.session_dirs[0]] * len(session_batch.labels))
        self._session_dirs = tuple(per_event_sessions)

    @property
    def batch(self) -> EventWindowBatch:
        return self._batch

    @property
    def sample_rate_hz(self) -> float:
        return self._batch.sample_rate_hz

    @property
    def pre_samples(self) -> int:
        return self._batch.pre_samples

    @property
    def post_samples(self) -> int:
        return self._batch.post_samples

    @property
    def skipped(self) -> int:
        return self._batch.skipped

    def __len__(self) -> int:
        return int(self._batch.x.shape[0])

    def _window_tensor(self, index: int) -> torch.Tensor:
        window = self._batch.x[index]
        tensor = torch.from_numpy(window).to(dtype=self._dtype)
        if self._channel_first:
            tensor = tensor.transpose(0, 1)
        return tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        label = self._batch.labels[index]
        item: dict[str, Any] = {
            "x": self._window_tensor(index),
            "label": label,
            "event_type": self._batch.event_types[index],
            "event_id": self._batch.event_ids[index],
            "center_sample_index": int(self._batch.center_sample_index[index]),
            "session_dir": self._session_dirs[index],
        }
        if self._label_to_idx is not None:
            if label not in self._label_to_idx:
                raise KeyError(f"Unknown label '{label}'")
            item["label_idx"] = self._label_to_idx[label]
        return item

    @classmethod
    def from_batch(
        cls,
        batch: EventWindowBatch,
        *,
        per_sample_session_dirs: Sequence[Union[Path, str]],
        label_to_idx: Optional[dict[str, int]] = None,
        dtype: torch.dtype = torch.float32,
        channel_first: bool = False,
    ) -> SessionEventDataset:
        dataset = cls.__new__(cls)
        dataset._batch = batch
        dataset._session_dirs = tuple(Path(path) for path in per_sample_session_dirs)
        dataset._label_to_idx = label_to_idx
        dataset._dtype = dtype
        dataset._channel_first = channel_first
        dataset.build_elapsed_s = 0.0
        return dataset


SPLITS_MANIFEST_NAME = "splits_manifest.json"
SPLITS_WINDOWS_NAME = "splits_windows.npz"
SPLIT_SEED = 42


def seed_everything(seed: int = 0, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


_rng = np.random.default_rng(SPLIT_SEED)


def set_split_seed(seed: int) -> None:
    global _rng, SPLIT_SEED
    SPLIT_SEED = seed
    _rng = np.random.default_rng(seed)


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    dataset: SessionEventDataset
    train: Subset
    val: Subset
    test: Subset
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_sessions: tuple[Path, ...]
    val_sessions: tuple[Path, ...]
    recordings_path: Path
    pre_ms: float
    post_ms: float
    extra_session_test_split: float
    intra_session_test_split: float


def split_sample_indices(
    sessions: list[Path],
    per_sample_sessions: Sequence[Path],
    *,
    extra_session_test_split: float = 0.2,
    intra_session_test_split: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Path, ...], tuple[Path, ...]]:
    """Return disjoint train, val, and test sample indices.

    train = events from train sessions not assigned to test
    val   = all events from held-out extra sessions
    test  = held-out intra-session events from train sessions
    """
    if not sessions:
        raise ValueError("At least one session is required")
    if not (0.0 <= extra_session_test_split < 1.0):
        raise ValueError("extra_session_test_split must be between 0 and 1")
    if not (0.0 <= intra_session_test_split < 1.0):
        raise ValueError("intra_session_test_split must be between 0 and 1")

    session_to_indices: dict[Path, list[int]] = defaultdict(list)
    for index, session_dir in enumerate(per_sample_sessions):
        session_to_indices[Path(session_dir)].append(index)

    session_order = _rng.permutation(len(sessions))

    n_val_sessions = int(round(len(sessions) * extra_session_test_split))
    if len(sessions) > 1:
        n_val_sessions = min(n_val_sessions, len(sessions) - 1)
    else:
        n_val_sessions = 0

    val_sessions = {sessions[int(i)] for i in session_order[:n_val_sessions]}
    remaining_sessions = [sessions[int(i)] for i in session_order[n_val_sessions:]]

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for session_dir in sorted(val_sessions):
        val_indices.extend(session_to_indices[session_dir])

    for session_dir in remaining_sessions:
        indices = session_to_indices.get(session_dir, [])
        if not indices:
            continue
        if len(indices) == 1 or intra_session_test_split == 0.0:
            train_indices.extend(indices)
            continue

        event_order = _rng.permutation(len(indices))
        n_test = max(1, int(round(len(indices) * intra_session_test_split)))
        n_test = min(n_test, len(indices) - 1)
        test_indices.extend(indices[int(i)] for i in event_order[:n_test])
        train_indices.extend(indices[int(i)] for i in event_order[n_test:])

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(val_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
        tuple(sorted(set(remaining_sessions))),
        tuple(sorted(val_sessions)),
    )


def build_dataset_splits(
    recordings_path: Path,
    *,
    pre_ms: float = 300.0,
    post_ms: float = 700.0,
    extra_session_test_split: float = 0.2,
    intra_session_test_split: float = 0.15,
    show_progress: bool = True,
) -> DatasetSplits:
    from _viewer_core import discover_sessions

    recordings_path = Path(recordings_path)
    sessions = discover_sessions(recordings_path)
    dataset = SessionEventDataset(
        sessions,
        pre_ms=pre_ms,
        post_ms=post_ms,
        show_progress=show_progress,
    )
    (
        train_indices,
        val_indices,
        test_indices,
        train_sessions,
        val_sessions,
    ) = split_sample_indices(
        sessions,
        dataset._session_dirs,
        extra_session_test_split=extra_session_test_split,
        intra_session_test_split=intra_session_test_split,
    )
    return DatasetSplits(
        dataset=dataset,
        train=Subset(dataset, train_indices.tolist()),
        val=Subset(dataset, val_indices.tolist()),
        test=Subset(dataset, test_indices.tolist()),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        recordings_path=recordings_path,
        pre_ms=pre_ms,
        post_ms=post_ms,
        extra_session_test_split=extra_session_test_split,
        intra_session_test_split=intra_session_test_split,
    )


def save_dataset_splits(output_dir: Path, splits: DatasetSplits) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = splits.dataset.batch
    np.savez_compressed(
        output_dir / SPLITS_WINDOWS_NAME,
        x=batch.x,
        center_sample_index=batch.center_sample_index,
    )

    manifest = {
        "version": 2,
        "seed": SPLIT_SEED,
        "recordings_path": str(splits.recordings_path.resolve()),
        "pre_ms": splits.pre_ms,
        "post_ms": splits.post_ms,
        "extra_session_test_split": splits.extra_session_test_split,
        "intra_session_test_split": splits.intra_session_test_split,
        "sample_rate_hz": batch.sample_rate_hz,
        "pre_samples": batch.pre_samples,
        "post_samples": batch.post_samples,
        "skipped": batch.skipped,
        "labels": list(batch.labels),
        "event_types": list(batch.event_types),
        "event_ids": list(batch.event_ids),
        "session_dirs": [str(path) for path in splits.dataset._session_dirs],
        "train_indices": splits.train_indices.tolist(),
        "val_indices": splits.val_indices.tolist(),
        "test_indices": splits.test_indices.tolist(),
        "train_sessions": [str(path) for path in splits.train_sessions],
        "val_sessions": [str(path) for path in splits.val_sessions],
    }
    (output_dir / SPLITS_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def load_dataset_splits(splits_dir: Path) -> DatasetSplits:
    splits_dir = Path(splits_dir)
    manifest_path = splits_dir / SPLITS_MANIFEST_NAME
    windows_path = splits_dir / SPLITS_WINDOWS_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing split windows: {windows_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows = np.load(windows_path)

    batch = EventWindowBatch(
        x=windows["x"],
        labels=tuple(manifest["labels"]),
        event_types=tuple(manifest["event_types"]),
        event_ids=tuple(manifest["event_ids"]),
        center_sample_index=windows["center_sample_index"],
        session_dirs=tuple(Path(path) for path in manifest["session_dirs"]),
        sample_rate_hz=float(manifest["sample_rate_hz"]),
        pre_samples=int(manifest["pre_samples"]),
        post_samples=int(manifest["post_samples"]),
        skipped=int(manifest.get("skipped", 0)),
    )
    dataset = SessionEventDataset.from_batch(
        batch,
        per_sample_session_dirs=manifest["session_dirs"],
    )

    train_indices = np.asarray(manifest["train_indices"], dtype=np.int64)
    if "val_indices" in manifest:
        val_indices = np.asarray(manifest["val_indices"], dtype=np.int64)
        test_indices = np.asarray(manifest["test_indices"], dtype=np.int64)
        val_sessions = tuple(Path(path) for path in manifest["val_sessions"])
    else:
        # v1 manifests combined val (extra-session) + test (intra-session) in test_indices
        combined_test = np.asarray(manifest["test_indices"], dtype=np.int64)
        val_session_set = {Path(path) for path in manifest["extra_test_sessions"]}
        per_sample_sessions = tuple(Path(path) for path in manifest["session_dirs"])
        val_mask = np.array(
            [per_sample_sessions[i] in val_session_set for i in combined_test],
            dtype=bool,
        )
        val_indices = combined_test[val_mask]
        test_indices = combined_test[~val_mask]
        val_sessions = tuple(Path(path) for path in manifest["extra_test_sessions"])

    return DatasetSplits(
        dataset=dataset,
        train=Subset(dataset, train_indices.tolist()),
        val=Subset(dataset, val_indices.tolist()),
        test=Subset(dataset, test_indices.tolist()),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_sessions=tuple(Path(path) for path in manifest["train_sessions"]),
        val_sessions=val_sessions,
        recordings_path=Path(manifest["recordings_path"]),
        pre_ms=float(manifest["pre_ms"]),
        post_ms=float(manifest["post_ms"]),
        extra_session_test_split=float(manifest["extra_session_test_split"]),
        intra_session_test_split=float(manifest["intra_session_test_split"]),
    )


def label_distribution(
    dataset: SessionEventDataset,
    indices: Sequence[int],
) -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for label in all_labels()}
    for index in indices:
        label = dataset[int(index)]["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def print_label_distribution(name: str, counts: dict[str, int], total: int) -> None:
    print(f"{name} label distribution ({total} samples):")
    for label in all_labels():
        count = counts.get(label, 0)
        pct = (100.0 * count / total) if total else 0.0
        print(f"  {label:14s} {count:5d}  ({pct:5.1f}%)")
    extra = sorted(label for label in counts if label not in all_labels() and counts[label] > 0)
    for label in extra:
        count = counts[label]
        pct = (100.0 * count / total) if total else 0.0
        print(f"  {label:14s} {count:5d}  ({pct:5.1f}%)")


def print_split_summary(splits: DatasetSplits) -> None:
    print("\n\n")
    print(f"splits: {len(splits.dataset)} total windows ({splits.dataset.skipped} skipped during build)")
    print(
        f"train: {len(splits.train)} samples from {len(splits.train_sessions)} sessions"
    )
    print(
        f"val:   {len(splits.val)} samples from {len(splits.val_sessions)} held-out sessions"
    )
    print(
        f"test:  {len(splits.test)} held-out intra-session events from train sessions"
    )
    print()
    for name, subset in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        counts = label_distribution(splits.dataset, subset.indices)
        print_label_distribution(name, counts, len(subset))
    print("\n\n")


# --- config (edit these)
SEED = 42
RECORDINGS_PATH = Path(__file__).resolve().parent.parent / "client" / "recordings"
SPLITS_OUTPUT_DIR = Path(__file__).resolve().parent / "splits"
PRE_MS = 0.0
POST_MS = 0.0
EXTRA_SESSION_TEST_SPLIT = 0.2
INTRA_SESSION_TEST_SPLIT = 0.15


if __name__ == "__main__":
    set_split_seed(SEED)
    seed_everything(SEED)

    total_started = time.perf_counter()
    build_started = time.perf_counter()
    splits = build_dataset_splits(
        RECORDINGS_PATH,
        pre_ms=PRE_MS,
        post_ms=POST_MS,
        extra_session_test_split=EXTRA_SESSION_TEST_SPLIT,
        intra_session_test_split=INTRA_SESSION_TEST_SPLIT,
        show_progress=True,
    )
    build_elapsed = time.perf_counter() - build_started

    save_started = time.perf_counter()
    save_dataset_splits(SPLITS_OUTPUT_DIR, splits)
    save_elapsed = time.perf_counter() - save_started
    total_elapsed = time.perf_counter() - total_started

    print(f"Saved splits to {SPLITS_OUTPUT_DIR.resolve()}")
    print(
        f"Timing: build={build_elapsed:.1f}s, save={save_elapsed:.1f}s, total={total_elapsed:.1f}s"
    )
    print_split_summary(splits)
