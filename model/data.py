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


# --- config

# what goes in - CHANGE HERE
INCLUDE_UNKNOWN_WORD_LABEL = False

INCLUDE_SILENCE_FROM_BREAKS = True
INCLUDE_SILENCE_FROM_OCCASIONAL_WORD = False

INCLUDE_TRANSITIONS_FROM_BREAKS = True
INCLUDE_TRANSITIONS_FROM_OCCASIONAL_WORD = False

# When True, transition windows use silence (not word starting/ending) as the
# non-word label, with the same soft word mass near the boundary.
MERGE_TRANSITIONS_INTO_SILENCE = True



TARGET_WORDS: tuple[str, ...] = (
    "highlight",
    "bullshit",
    "gogogo",
    "shitbull",
    "naan",
    # "halloween",
    # "glue",
)

# how it goes in - CHANGE HERE
EXTRA_SESSION_TEST_SPLIT = 0.15
INTRA_SESSION_TEST_SPLIT = 0.15
STRATIFIED_LABEL_SPLIT = True

# --- tech shit
SEED = 42
RECORDINGS_PATH = Path(__file__).resolve().parent.parent / "client" / "recordings"
SPLITS_OUTPUT_DIR = Path(__file__).resolve().parent / "splits"
PRE_MS = 0.0
POST_MS = 0.0

EEG_RECORD_FORMAT = "<QQ32f"
EEG_RECORD_FORMAT_CODES_LEGACY = "<QQ32i"

# --- scramble-breaks transition sampling (edit these)
# When True, transitions/silence gaps use only TARGET_WORDS spans (default).
# When False, every labeled word in a scramble-breaks block is used.
SCRAMBLE_BREAKS_SHIFTS_PER_TRANSITION = 3
SCRAMBLE_BREAKS_SHIFT_MIN_S = -1.2
SCRAMBLE_BREAKS_SHIFT_MAX_S = 1.2
# Silence/word mass stays zero until a phase exceeds this fraction; below it
# p_transition = 1.0. At 0.7 with COLLECTION_SAY_S=1.6s, shifts within ~±0.3s
# of the boundary remain pure transition.
TRANSITION_PURE_PHASE_FRAC = 0.7


SCRAMBLE_BREAKS_ONLY_TARGET_WORDS = True

# do not change, depends on app.py
COLLECTION_SAY_S = 1.6



# lable managment
UNKNOWN_WORD_LABEL = "unknown word"
SILENCE_LABEL = "silence"
WORD_STARTING_LABEL = "word starting"
WORD_ENDING_LABEL = "word ending"
SCRAMBLE_BREAKS_MODE = "scramble-breaks"
SCRAMBLE_BREAKS_BLOCK_START_EVENT = "silent_speech_scramble_start"




# --- helpers

def _validate_label_source_config() -> None:
    if INCLUDE_TRANSITIONS_FROM_BREAKS and not INCLUDE_SILENCE_FROM_BREAKS:
        raise ValueError(
            "INCLUDE_TRANSITIONS_FROM_BREAKS requires INCLUDE_SILENCE_FROM_BREAKS"
        )
    if INCLUDE_TRANSITIONS_FROM_OCCASIONAL_WORD and not INCLUDE_SILENCE_FROM_OCCASIONAL_WORD:
        raise ValueError(
            "INCLUDE_TRANSITIONS_FROM_OCCASIONAL_WORD requires INCLUDE_SILENCE_FROM_OCCASIONAL_WORD"
        )


_validate_label_source_config()


def _transitions_in_label_space() -> bool:
    return INCLUDE_TRANSITIONS_FROM_BREAKS or INCLUDE_TRANSITIONS_FROM_OCCASIONAL_WORD


def _silence_in_label_space() -> bool:
    if INCLUDE_SILENCE_FROM_BREAKS or INCLUDE_SILENCE_FROM_OCCASIONAL_WORD:
        return True
    return MERGE_TRANSITIONS_INTO_SILENCE and _transitions_in_label_space()


def all_labels() -> tuple[str, ...]:
    labels: list[str] = list(TARGET_WORDS)
    if _transitions_in_label_space() and not MERGE_TRANSITIONS_INTO_SILENCE:
        labels.extend([WORD_STARTING_LABEL, WORD_ENDING_LABEL])
    if INCLUDE_UNKNOWN_WORD_LABEL:
        labels.append(UNKNOWN_WORD_LABEL)
    if _silence_in_label_space():
        labels.append(SILENCE_LABEL)
    return tuple(labels)

def default_label_max_fractions(fraction: float = 0.20) -> dict[str, float]:
    return {label: fraction for label in all_labels()}
LABEL_MAX_FRACTIONS = default_label_max_fractions(0.25)

def _block_id_from_event_id(event_id: str) -> str:
    if event_id.startswith("silence:"):
        parts = event_id.split(":", 2)
        return parts[1] if len(parts) > 1 else ""
    if ":shift=" in event_id:
        return event_id.split(":", 1)[0]
    return ""


def _payload_block_id(payload: dict[str, Any]) -> str:
    return str(payload.get("collection_block_id", "")).strip()


ALL_LABELS: tuple[str, ...] = all_labels()
SILENT_SPEECH_WORD_EVENT = "silent_speech_word"
OCCASIONAL_WORD_MODE = "occasional_word"
LEGACY_OCCASIONAL_WORD_MODE = "negative_labels"
OCCASIONAL_WORD_BLOCK_START_EVENT = "silent_speech_block_start"
OCCASIONAL_WORD_BLOCK_END_EVENT = "silent_speech_block_end"
SCRAMBLE_BREAKS_TRANSITION_EVENT = "scramble_breaks_transition"
OCCASIONAL_WORD_TRANSITION_EVENT = "negative_labels_transition"
TRANSITION_EVENT_TYPES = frozenset(
    {SCRAMBLE_BREAKS_TRANSITION_EVENT, OCCASIONAL_WORD_TRANSITION_EVENT}
)


def _is_occasional_word_mode(mode: Any) -> bool:
    return mode in (OCCASIONAL_WORD_MODE, LEGACY_OCCASIONAL_WORD_MODE)

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


def _occasional_word_blocks(events: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
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
            event_type == OCCASIONAL_WORD_BLOCK_START_EVENT
            and _is_occasional_word_mode(payload.get("mode"))
        ):
            pending[block_id] = int(start_text)
            continue
        if (
            event_type == OCCASIONAL_WORD_BLOCK_END_EVENT
            and _is_occasional_word_mode(payload.get("mode"))
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


def _occasional_word_word_spans(
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


def _occasional_word_word_spans_labeled(
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
        word = normalize_word_label(event.get("label_text", ""))
        if word is None:
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


def _occasional_word_silence_spans(
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
    label_probs: list[Optional[dict[str, float]]],
    event_type_list: list[str],
    event_ids: list[str],
    collection_block_ids: list[str],
    center_samples: list[int],
    skipped: int,
) -> int:
    for block in _occasional_word_blocks(events):
        block_id = str(block["block_id"])
        word_spans_labeled = _occasional_word_word_spans_labeled(
            events,
            block_id,
            fs=fs,
            index_to_row=index_to_row,
        )
        span_pairs = [(start_idx, end_idx) for start_idx, end_idx, _word in word_spans_labeled]
        for gap_start, gap_end in _occasional_word_silence_spans(
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
            before_word, after_word = _bordering_words_for_gap(
                gap_start,
                gap_end,
                word_spans_labeled,
            )
            raw_windows.append(window)
            labels.append(SILENCE_LABEL)
            label_probs.append(None)
            event_type_list.append("occasional_word_silence")
            event_ids.append(
                f"silence:{block_id}:{gap_start}-{gap_end}:words={before_word}|{after_word}"
            )
            collection_block_ids.append(block_id)
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


def _transition_shift_label_probs(
    *,
    kind: str,
    word: str,
    silence_frac: float,
    word_frac: float,
    include_silence_label: bool,
) -> dict[str, float]:
    """Soft label distribution for a transition-centered window.

    Below TRANSITION_PURE_PHASE_FRAC in both phases, all mass stays on the
    transition label (or silence when MERGE_TRANSITIONS_INTO_SILENCE). Above
    that threshold, silence/word mass ramps linearly to 1.0 at a pure phase
    window.
    """
    threshold = TRANSITION_PURE_PHASE_FRAC
    ramp_span = max(1.0 - threshold, 1e-9)
    p_silence = max(0.0, (silence_frac - threshold) / ramp_span)
    p_word = max(0.0, (word_frac - threshold) / ramp_span)
    p_transition = max(0.0, 1.0 - p_silence - p_word)
    transition_label = WORD_STARTING_LABEL if kind == "silence_to_word" else WORD_ENDING_LABEL

    probs: dict[str, float] = {}
    if MERGE_TRANSITIONS_INTO_SILENCE:
        p_silence_merged = p_silence + p_transition
        if p_silence_merged > 0.0:
            probs[SILENCE_LABEL] = p_silence_merged
    else:
        if include_silence_label and p_silence > 0.0:
            probs[SILENCE_LABEL] = p_silence
        if p_transition > 0.0:
            probs[transition_label] = p_transition
    if p_word > 0.0 and (INCLUDE_UNKNOWN_WORD_LABEL or word != UNKNOWN_WORD_LABEL):
        probs[word] = p_word

    total = sum(probs.values())
    if total <= 0.0:
        if MERGE_TRANSITIONS_INTO_SILENCE:
            return {SILENCE_LABEL: 1.0}
        return {transition_label: 1.0}
    return {label: weight / total for label, weight in probs.items()}


def _transition_shift_label(
    *,
    kind: str,
    word: str,
    silence_frac: float,
    word_frac: float,
    include_silence_label: bool,
) -> str:
    probs = _transition_shift_label_probs(
        kind=kind,
        word=word,
        silence_frac=silence_frac,
        word_frac=word_frac,
        include_silence_label=include_silence_label,
    )
    return max(probs, key=probs.get)


def _parse_scramble_breaks_transition_event_id(event_id: str) -> Optional[tuple[str, str, float]]:
    if ":shift=" not in event_id:
        return None
    base, shift_text = event_id.rsplit(":shift=", 1)
    parts = base.split(":")
    if len(parts) < 4:
        return None
    kind = parts[1]
    if kind not in {"silence_to_word", "word_to_silence"}:
        return None
    word = parts[3]
    return kind, word, float(shift_text)


def _parse_silence_gap_words(event_id: str) -> tuple[str, ...]:
    if ":words=" not in event_id:
        return ()
    words_part = event_id.rsplit(":words=", 1)[1]
    return tuple(word for word in words_part.split("|") if word)


def _bordering_words_for_gap(
    gap_start: int,
    gap_end: int,
    word_spans: Sequence[tuple[int, int, str]],
) -> tuple[str, str]:
    before_word = ""
    after_word = ""
    for start_idx, end_idx, word in word_spans:
        if end_idx <= gap_start:
            before_word = word
        if start_idx >= gap_end and not after_word:
            after_word = word
    return before_word, after_word


def _sample_word_fraction(event_type: str, event_id: str) -> float:
    if event_type == SILENT_SPEECH_WORD_EVENT:
        return 1.0
    if event_type == SCRAMBLE_BREAKS_TRANSITION_EVENT:
        parsed = _parse_scramble_breaks_transition_event_id(event_id)
        if parsed is None:
            return 0.0
        _kind, _word, shift_s = parsed
        _silence_frac, word_frac = _transition_phase_fractions(
            shift_s=shift_s,
            window_s=COLLECTION_SAY_S,
            kind=_kind,
        )
        return word_frac
    if event_type == OCCASIONAL_WORD_TRANSITION_EVENT:
        parsed = _parse_scramble_breaks_transition_event_id(event_id)
        if parsed is None:
            return 0.0
        _kind, _word, shift_s = parsed
        _silence_frac, word_frac = _transition_phase_fractions(
            shift_s=shift_s,
            window_s=COLLECTION_SAY_S,
            kind=_kind,
        )
        return word_frac
    return 0.0


def _sample_silence_fraction(event_type: str, event_id: str) -> float:
    if event_type in {"scramble_breaks_silence", "occasional_word_silence", "negative_labels_silence"}:
        return 1.0
    if event_type == SILENT_SPEECH_WORD_EVENT:
        return 0.0
    if event_type in TRANSITION_EVENT_TYPES:
        parsed = _parse_scramble_breaks_transition_event_id(event_id)
        if parsed is None:
            return 0.0
        kind, _word, shift_s = parsed
        silence_frac, _word_frac = _transition_phase_fractions(
            shift_s=shift_s,
            window_s=COLLECTION_SAY_S,
            kind=kind,
        )
        return silence_frac
    return 0.0


def _is_full_word_label_sample(
    event_type: str,
    event_id: str,
    hard_label: str,
    word: str,
) -> bool:
    if hard_label != word:
        return False
    if event_type == SILENT_SPEECH_WORD_EVENT:
        return True
    if event_type in TRANSITION_EVENT_TYPES:
        return _sample_word_fraction(event_type, event_id) >= 1.0 - 1e-9
    return False


def _associated_words(event_type: str, event_id: str) -> tuple[str, ...]:
    if event_type in TRANSITION_EVENT_TYPES:
        parsed = _parse_scramble_breaks_transition_event_id(event_id)
        if parsed is not None:
            return (parsed[1],)
    gap_words = _parse_silence_gap_words(event_id)
    if gap_words:
        return gap_words
    return ()


def _label_prob_mass(
    label_probs: Optional[dict[str, float]],
    label: str,
    hard_label: str,
) -> float:
    if label_probs is None:
        return 1.0 if hard_label == label else 0.0
    return label_probs.get(label, 0.0)


def _label_max_prob(
    label_probs: Optional[dict[str, float]],
    hard_label: str,
) -> float:
    if label_probs is None:
        return 1.0
    if not label_probs:
        return 1.0
    return max(label_probs.values())


def _transition_event_include_silence(event_type: str) -> bool:
    if event_type == SCRAMBLE_BREAKS_TRANSITION_EVENT:
        return INCLUDE_SILENCE_FROM_BREAKS
    if event_type == OCCASIONAL_WORD_TRANSITION_EVENT:
        return INCLUDE_SILENCE_FROM_OCCASIONAL_WORD
    return False


def transition_label_probs_from_event_id(
    event_id: str,
    *,
    window_s: float = COLLECTION_SAY_S,
    event_type: str = SCRAMBLE_BREAKS_TRANSITION_EVENT,
) -> Optional[dict[str, float]]:
    parsed = _parse_scramble_breaks_transition_event_id(event_id)
    if parsed is None:
        return None
    kind, word, shift_s = parsed
    silence_frac, word_frac = _transition_phase_fractions(
        shift_s=shift_s,
        window_s=window_s,
        kind=kind,
    )
    return _transition_shift_label_probs(
        kind=kind,
        word=word,
        silence_frac=silence_frac,
        word_frac=word_frac,
        include_silence_label=_transition_event_include_silence(event_type),
    )


def label_probs_to_vector(
    label_probs: Optional[dict[str, float]],
    hard_label: str,
    label_to_idx: dict[str, int],
) -> np.ndarray:
    n = len(label_to_idx)
    vec = np.zeros(n, dtype=np.float32)
    if label_probs:
        for label, prob in label_probs.items():
            idx = label_to_idx.get(label)
            if idx is not None:
                vec[idx] = prob
        total = float(vec.sum())
        if total > 0.0:
            vec /= total
            return vec
    vec[label_to_idx[hard_label]] = 1.0
    return vec


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
            event_type == OCCASIONAL_WORD_BLOCK_END_EVENT
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
        if word is None:
            continue
        if SCRAMBLE_BREAKS_ONLY_TARGET_WORDS and word == UNKNOWN_WORD_LABEL:
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


def _append_block_transition_windows(
    *,
    blocks: Sequence[dict[str, Any]],
    word_spans_for_block: Any,
    channels: np.ndarray,
    index_to_row: dict[int, int],
    fs: float,
    pre_samples: int,
    post_samples: int,
    raw_windows: list[np.ndarray],
    labels: list[str],
    label_probs: list[Optional[dict[str, float]]],
    event_type_list: list[str],
    event_ids: list[str],
    collection_block_ids: list[str],
    center_samples: list[int],
    event_type: str,
    include_silence_label: bool,
) -> int:
    skipped = 0
    window_s = COLLECTION_SAY_S
    window_half_samples = max(1, int(round(window_s * fs / 2.0)))

    for block in blocks:
        block_id = str(block["block_id"])
        word_spans = word_spans_for_block(block_id)
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
                probs = _transition_shift_label_probs(
                    kind=kind,
                    word=word,
                    silence_frac=silence_frac,
                    word_frac=word_frac,
                    include_silence_label=include_silence_label,
                )
                label = max(probs, key=probs.get)
                center = boundary_idx + shift_samples
                raw_windows.append(window)
                labels.append(label)
                label_probs.append(probs)
                event_type_list.append(event_type)
                event_ids.append(f"{boundary_key}:shift={shift_s:+.3f}")
                collection_block_ids.append(block_id)
                center_samples.append(center)
    return skipped


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
    label_probs: list[Optional[dict[str, float]]],
    event_type_list: list[str],
    event_ids: list[str],
    collection_block_ids: list[str],
    center_samples: list[int],
) -> int:
    return _append_block_transition_windows(
        blocks=_scramble_breaks_blocks(events),
        word_spans_for_block=lambda block_id: _scramble_breaks_word_spans(
            events,
            block_id,
            fs=fs,
            index_to_row=index_to_row,
        ),
        channels=channels,
        index_to_row=index_to_row,
        fs=fs,
        pre_samples=pre_samples,
        post_samples=post_samples,
        raw_windows=raw_windows,
        labels=labels,
        label_probs=label_probs,
        event_type_list=event_type_list,
        event_ids=event_ids,
        collection_block_ids=collection_block_ids,
        center_samples=center_samples,
        event_type=SCRAMBLE_BREAKS_TRANSITION_EVENT,
        include_silence_label=INCLUDE_SILENCE_FROM_BREAKS,
    )


def _append_occasional_word_transition_windows(
    *,
    channels: np.ndarray,
    events: Sequence[dict[str, str]],
    index_to_row: dict[int, int],
    fs: float,
    pre_samples: int,
    post_samples: int,
    raw_windows: list[np.ndarray],
    labels: list[str],
    label_probs: list[Optional[dict[str, float]]],
    event_type_list: list[str],
    event_ids: list[str],
    collection_block_ids: list[str],
    center_samples: list[int],
) -> int:
    return _append_block_transition_windows(
        blocks=_occasional_word_blocks(events),
        word_spans_for_block=lambda block_id: _occasional_word_word_spans_labeled(
            events,
            block_id,
            fs=fs,
            index_to_row=index_to_row,
        ),
        channels=channels,
        index_to_row=index_to_row,
        fs=fs,
        pre_samples=pre_samples,
        post_samples=post_samples,
        raw_windows=raw_windows,
        labels=labels,
        label_probs=label_probs,
        event_type_list=event_type_list,
        event_ids=event_ids,
        collection_block_ids=collection_block_ids,
        center_samples=center_samples,
        event_type=OCCASIONAL_WORD_TRANSITION_EVENT,
        include_silence_label=INCLUDE_SILENCE_FROM_OCCASIONAL_WORD,
    )


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
    label_probs: list[Optional[dict[str, float]]],
    event_type_list: list[str],
    event_ids: list[str],
    collection_block_ids: list[str],
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
        for gap_start, gap_end in _occasional_word_silence_spans(
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
            before_word, after_word = _bordering_words_for_gap(
                gap_start,
                gap_end,
                word_spans,
            )
            raw_windows.append(window)
            labels.append(SILENCE_LABEL)
            label_probs.append(None)
            event_type_list.append("scramble_breaks_silence")
            event_ids.append(
                f"silence:{block_id}:{gap_start}-{gap_end}:words={before_word}|{after_word}"
            )
            collection_block_ids.append(block_id)
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
    collection_block_ids: tuple[str, ...]
    center_sample_index: np.ndarray
    session_dirs: tuple[Path, ...]
    sample_rate_hz: float
    pre_samples: int
    post_samples: int
    skipped: int
    label_probs: tuple[Optional[dict[str, float]], ...] = ()

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
    label_probs: list[Optional[dict[str, float]]] = []
    event_type_list: list[str] = []
    event_ids: list[str] = []
    collection_block_ids: list[str] = []
    center_samples: list[int] = []
    skipped = 0

    for event in events:
        event_type = event.get("event_type", "")
        if event_types is not None and event_type not in event_types:
            continue
        if event_type != SILENT_SPEECH_WORD_EVENT:
            continue

        payload = parse_event_payload(event)
        mode = payload.get("mode")
        if mode == SCRAMBLE_BREAKS_MODE and INCLUDE_TRANSITIONS_FROM_BREAKS:
            continue
        if _is_occasional_word_mode(mode) and INCLUDE_TRANSITIONS_FROM_OCCASIONAL_WORD:
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
        label_probs.append(None)
        event_type_list.append(event_type)
        event_ids.append(event.get("event_id", ""))
        collection_block_ids.append(_payload_block_id(payload))
        center_samples.append(start_idx)

    if INCLUDE_TRANSITIONS_FROM_BREAKS:
        skipped += _append_scramble_breaks_transition_windows(
            channels=channels,
            events=events,
            index_to_row=index_to_row,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            raw_windows=raw_windows,
            labels=labels,
            label_probs=label_probs,
            event_type_list=event_type_list,
            event_ids=event_ids,
            collection_block_ids=collection_block_ids,
            center_samples=center_samples,
        )
    elif INCLUDE_SILENCE_FROM_BREAKS:
        skipped += _append_scramble_breaks_silence_windows(
            channels=channels,
            events=events,
            index_to_row=index_to_row,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            raw_windows=raw_windows,
            labels=labels,
            label_probs=label_probs,
            event_type_list=event_type_list,
            event_ids=event_ids,
            collection_block_ids=collection_block_ids,
            center_samples=center_samples,
        )

    if INCLUDE_TRANSITIONS_FROM_OCCASIONAL_WORD:
        skipped += _append_occasional_word_transition_windows(
            channels=channels,
            events=events,
            index_to_row=index_to_row,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            raw_windows=raw_windows,
            labels=labels,
            label_probs=label_probs,
            event_type_list=event_type_list,
            event_ids=event_ids,
            collection_block_ids=collection_block_ids,
            center_samples=center_samples,
        )
    elif INCLUDE_SILENCE_FROM_OCCASIONAL_WORD:
        skipped = _append_silence_windows(
            channels=channels,
            events=events,
            index_to_row=index_to_row,
            fs=fs,
            pre_samples=pre_samples,
            post_samples=post_samples,
            raw_windows=raw_windows,
            labels=labels,
            label_probs=label_probs,
            event_type_list=event_type_list,
            event_ids=event_ids,
            collection_block_ids=collection_block_ids,
            center_samples=center_samples,
            skipped=skipped,
        )

    if raw_windows and len(label_probs) != len(raw_windows):
        raise RuntimeError(
            f"label_probs length ({len(label_probs)}) != windows ({len(raw_windows)})"
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
        collection_block_ids=tuple(collection_block_ids),
        center_sample_index=np.asarray(center_samples, dtype=np.int64),
        session_dirs=(session_dir,),
        sample_rate_hz=fs,
        pre_samples=pre_samples,
        post_samples=post_samples,
        skipped=skipped,
        label_probs=tuple(label_probs),
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

    def _batch_label_probs(batch: EventWindowBatch) -> tuple[Optional[dict[str, float]], ...]:
        if batch.label_probs:
            return batch.label_probs
        return tuple(None for _ in batch.labels)

    return EventWindowBatch(
        x=np.concatenate(aligned_x, axis=0),
        labels=tuple(label for batch in batches for label in batch.labels),
        event_types=tuple(event_type for batch in batches for event_type in batch.event_types),
        event_ids=tuple(event_id for batch in batches for event_id in batch.event_ids),
        collection_block_ids=tuple(
            block_id for batch in batches for block_id in batch.collection_block_ids
        ),
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
        label_probs=tuple(
            probs
            for batch in batches
            for probs in _batch_label_probs(batch)
        ),
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
        label_probs = None
        if self._batch.label_probs:
            label_probs = self._batch.label_probs[index]
        item: dict[str, Any] = {
            "x": self._window_tensor(index),
            "label": label,
            "label_probs": label_probs,
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
    stratified_label_split: bool = False
    label_max_fractions: Optional[dict[str, float]] = None
    label_cap_dropped: tuple[int, int, int] = (0, 0, 0)


def _pick_redundant_index_to_drop(
    candidates: Sequence[int],
    *,
    per_sample_sessions: Sequence[Path],
    per_sample_block_ids: Sequence[str],
    per_sample_event_ids: Sequence[str],
) -> int:
    """Pick the most redundant sample within a label (same session/block cluster)."""
    session_counts: dict[str, int] = defaultdict(int)
    block_counts: dict[tuple[str, str], int] = defaultdict(int)
    for index in candidates:
        session_key = str(per_sample_sessions[index])
        block_key = per_sample_block_ids[index] or per_sample_event_ids[index]
        session_counts[session_key] += 1
        block_counts[(session_key, block_key)] += 1

    def redundancy(index: int) -> tuple[int, int, int]:
        session_key = str(per_sample_sessions[index])
        block_key = per_sample_block_ids[index] or per_sample_event_ids[index]
        return (
            block_counts[(session_key, block_key)],
            session_counts[session_key],
            _stable_event_seed(per_sample_event_ids[index]),
        )

    return max(candidates, key=redundancy)


def apply_label_fraction_caps(
    indices: Sequence[int],
    per_sample_labels: Sequence[str],
    per_sample_sessions: Sequence[Path],
    per_sample_block_ids: Sequence[str],
    per_sample_event_ids: Sequence[str],
    label_max_fractions: dict[str, float],
) -> np.ndarray:
    """Drop samples until each capped label is at or below its max fraction."""
    kept = [int(index) for index in indices]
    if not kept or not label_max_fractions:
        return np.asarray(kept, dtype=np.int64)

    capped_labels = {
        label: fraction
        for label, fraction in label_max_fractions.items()
        if 0.0 < fraction < 1.0
    }
    if not capped_labels:
        return np.asarray(kept, dtype=np.int64)

    max_drops = len(kept)
    drops = 0
    while drops < max_drops:
        total = len(kept)
        if total == 0:
            break

        worst_label: Optional[str] = None
        worst_excess = 0.0
        for label, cap in capped_labels.items():
            count = sum(1 for index in kept if per_sample_labels[index] == label)
            if count == 0:
                continue
            excess = (count / total) - cap
            if excess > worst_excess + 1e-12:
                worst_excess = excess
                worst_label = label

        if worst_label is None:
            break

        candidates = [index for index in kept if per_sample_labels[index] == worst_label]
        drop_index = _pick_redundant_index_to_drop(
            candidates,
            per_sample_sessions=per_sample_sessions,
            per_sample_block_ids=per_sample_block_ids,
            per_sample_event_ids=per_sample_event_ids,
        )
        kept.remove(drop_index)
        drops += 1

    return np.asarray(kept, dtype=np.int64)


def _apply_split_label_fraction_caps(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    per_sample_labels: Sequence[str],
    per_sample_sessions: Sequence[Path],
    per_sample_block_ids: Sequence[str],
    per_sample_event_ids: Sequence[str],
    label_max_fractions: Optional[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    if not label_max_fractions:
        return train_indices, val_indices, test_indices, (0, 0, 0)

    capped_train = apply_label_fraction_caps(
        train_indices,
        per_sample_labels,
        per_sample_sessions,
        per_sample_block_ids,
        per_sample_event_ids,
        label_max_fractions,
    )
    capped_val = apply_label_fraction_caps(
        val_indices,
        per_sample_labels,
        per_sample_sessions,
        per_sample_block_ids,
        per_sample_event_ids,
        label_max_fractions,
    )
    capped_test = apply_label_fraction_caps(
        test_indices,
        per_sample_labels,
        per_sample_sessions,
        per_sample_block_ids,
        per_sample_event_ids,
        label_max_fractions,
    )
    dropped = (
        len(train_indices) - len(capped_train),
        len(val_indices) - len(capped_val),
        len(test_indices) - len(capped_test),
    )
    return capped_train, capped_val, capped_test, dropped


def _label_counts_for_indices(
    indices: Sequence[int],
    per_sample_labels: Sequence[str],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for index in indices:
        counts[per_sample_labels[int(index)]] += 1
    return dict(counts)


def _label_distribution_l1_distance(
    counts: dict[str, int],
    target_fractions: dict[str, float],
) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("inf")
    return sum(
        abs(counts.get(label, 0) / total - fraction)
        for label, fraction in target_fractions.items()
    )


def _pick_val_sessions(
    sessions: list[Path],
    session_to_indices: dict[Path, list[int]],
    per_sample_labels: Sequence[str],
    n_val_sessions: int,
    *,
    stratified: bool,
    n_trials: int = 1000,
) -> set[Path]:
    if n_val_sessions <= 0:
        return set()

    if not stratified:
        session_order = _rng.permutation(len(sessions))
        return {sessions[int(i)] for i in session_order[:n_val_sessions]}

    all_indices = [index for session in sessions for index in session_to_indices[session]]
    global_counts = _label_counts_for_indices(all_indices, per_sample_labels)
    global_total = sum(global_counts.values())
    if global_total == 0:
        session_order = _rng.permutation(len(sessions))
        return {sessions[int(i)] for i in session_order[:n_val_sessions]}

    target_fractions = {
        label: count / global_total for label, count in global_counts.items()
    }

    best_score = float("inf")
    best_val_sessions: set[Path] = set()
    for _ in range(n_trials):
        session_order = _rng.permutation(len(sessions))
        val_sessions = {sessions[int(i)] for i in session_order[:n_val_sessions]}
        val_indices = [
            index for session in val_sessions for index in session_to_indices[session]
        ]
        score = _label_distribution_l1_distance(
            _label_counts_for_indices(val_indices, per_sample_labels),
            target_fractions,
        )
        if score < best_score:
            best_score = score
            best_val_sessions = val_sessions

    if best_val_sessions:
        return best_val_sessions

    session_order = _rng.permutation(len(sessions))
    return {sessions[int(i)] for i in session_order[:n_val_sessions]}


def _split_session_train_test(
    indices: Sequence[int],
    per_sample_labels: Sequence[str],
    *,
    intra_session_test_split: float,
    stratified: bool,
) -> tuple[list[int], list[int]]:
    indices = list(indices)
    if len(indices) <= 1 or intra_session_test_split == 0.0:
        return indices, []

    if not stratified:
        event_order = _rng.permutation(len(indices))
        n_test = max(1, int(round(len(indices) * intra_session_test_split)))
        n_test = min(n_test, len(indices) - 1)
        test_indices = [indices[int(i)] for i in event_order[:n_test]]
        train_indices = [indices[int(i)] for i in event_order[n_test:]]
        return train_indices, test_indices

    by_label: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_label[per_sample_labels[int(index)]].append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for label_indices in by_label.values():
        label_order = _rng.permutation(len(label_indices))
        ordered = [label_indices[int(i)] for i in label_order]
        if len(ordered) == 1:
            train_indices.extend(ordered)
            continue
        n_test = max(1, int(round(len(ordered) * intra_session_test_split)))
        n_test = min(n_test, len(ordered) - 1)
        test_indices.extend(ordered[:n_test])
        train_indices.extend(ordered[n_test:])

    return train_indices, test_indices


def split_sample_indices(
    sessions: list[Path],
    per_sample_sessions: Sequence[Path],
    per_sample_labels: Sequence[str],
    *,
    extra_session_test_split: float = 0.2,
    intra_session_test_split: float = 0.15,
    stratified_label_split: bool = False,
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

    n_val_sessions = int(round(len(sessions) * extra_session_test_split))
    if len(sessions) > 1:
        n_val_sessions = min(n_val_sessions, len(sessions) - 1)
    else:
        n_val_sessions = 0

    val_sessions = _pick_val_sessions(
        sessions,
        session_to_indices,
        per_sample_labels,
        n_val_sessions,
        stratified=stratified_label_split,
    )
    remaining_sessions = [session for session in sessions if session not in val_sessions]

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for session_dir in sorted(val_sessions):
        val_indices.extend(session_to_indices[session_dir])

    for session_dir in remaining_sessions:
        session_train, session_test = _split_session_train_test(
            session_to_indices.get(session_dir, []),
            per_sample_labels,
            intra_session_test_split=intra_session_test_split,
            stratified=stratified_label_split,
        )
        train_indices.extend(session_train)
        test_indices.extend(session_test)

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
    stratified_label_split: bool = False,
    label_max_fractions: Optional[dict[str, float]] = None,
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
        dataset.batch.labels,
        extra_session_test_split=extra_session_test_split,
        intra_session_test_split=intra_session_test_split,
        stratified_label_split=stratified_label_split,
    )
    block_ids = dataset.batch.collection_block_ids or tuple(
        _block_id_from_event_id(event_id) for event_id in dataset.batch.event_ids
    )
    (
        train_indices,
        val_indices,
        test_indices,
        label_cap_dropped,
    ) = _apply_split_label_fraction_caps(
        train_indices,
        val_indices,
        test_indices,
        per_sample_labels=dataset.batch.labels,
        per_sample_sessions=dataset._session_dirs,
        per_sample_block_ids=block_ids,
        per_sample_event_ids=dataset.batch.event_ids,
        label_max_fractions=label_max_fractions,
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
        stratified_label_split=stratified_label_split,
        label_max_fractions=label_max_fractions,
        label_cap_dropped=label_cap_dropped,
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

    label_probs_manifest: list[Optional[dict[str, float]]] = []
    for label, event_type, event_id in zip(
        batch.labels,
        batch.event_types,
        batch.event_ids,
        strict=True,
    ):
        if event_type in TRANSITION_EVENT_TYPES:
            probs = transition_label_probs_from_event_id(
                event_id,
                event_type=event_type,
            )
            label_probs_manifest.append(probs)
        else:
            label_probs_manifest.append(None)

    manifest = {
        "version": 3,
        "seed": SPLIT_SEED,
        "recordings_path": str(splits.recordings_path.resolve()),
        "pre_ms": splits.pre_ms,
        "post_ms": splits.post_ms,
        "extra_session_test_split": splits.extra_session_test_split,
        "intra_session_test_split": splits.intra_session_test_split,
        "stratified_label_split": splits.stratified_label_split,
        "label_max_fractions": splits.label_max_fractions,
        "label_cap_dropped": list(splits.label_cap_dropped),
        "sample_rate_hz": batch.sample_rate_hz,
        "pre_samples": batch.pre_samples,
        "post_samples": batch.post_samples,
        "skipped": batch.skipped,
        "labels": list(batch.labels),
        "label_probs": label_probs_manifest,
        "event_types": list(batch.event_types),
        "event_ids": list(batch.event_ids),
        "collection_block_ids": list(batch.collection_block_ids),
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

    if "collection_block_ids" in manifest:
        collection_block_ids = tuple(manifest["collection_block_ids"])
    else:
        collection_block_ids = tuple(
            _block_id_from_event_id(event_id) for event_id in manifest["event_ids"]
        )

    if "label_probs" in manifest:
        raw_label_probs = manifest["label_probs"]
        label_probs = tuple(
            dict(probs) if probs is not None else None for probs in raw_label_probs
        )
    else:
        label_probs = tuple(
            transition_label_probs_from_event_id(
                event_id,
                event_type=event_type,
            )
            if event_type in TRANSITION_EVENT_TYPES
            else None
            for event_type, event_id in zip(
                manifest["event_types"],
                manifest["event_ids"],
                strict=True,
            )
        )

    batch = EventWindowBatch(
        x=windows["x"],
        labels=tuple(manifest["labels"]),
        event_types=tuple(manifest["event_types"]),
        event_ids=tuple(manifest["event_ids"]),
        collection_block_ids=collection_block_ids,
        center_sample_index=windows["center_sample_index"],
        session_dirs=tuple(Path(path) for path in manifest["session_dirs"]),
        sample_rate_hz=float(manifest["sample_rate_hz"]),
        pre_samples=int(manifest["pre_samples"]),
        post_samples=int(manifest["post_samples"]),
        skipped=int(manifest.get("skipped", 0)),
        label_probs=label_probs,
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
        stratified_label_split=bool(manifest.get("stratified_label_split", False)),
        label_max_fractions=manifest.get("label_max_fractions"),
        label_cap_dropped=tuple(manifest.get("label_cap_dropped", (0, 0, 0))),
    )


def _label_probs_for_sample(
    batch: EventWindowBatch,
    index: int,
) -> Optional[dict[str, float]]:
    if batch.label_probs:
        probs = batch.label_probs[index]
        if probs is not None:
            return probs
    if batch.event_types[index] in TRANSITION_EVENT_TYPES:
        return transition_label_probs_from_event_id(
            batch.event_ids[index],
            event_type=batch.event_types[index],
        )
    return None


def _format_word_context(words: tuple[str, ...]) -> str:
    if not words:
        return "(none)"
    if len(words) == 1:
        return words[0]
    return " | ".join(words)


def print_label_coverage_summary(
    dataset: SessionEventDataset,
    indices: Optional[Sequence[int]] = None,
    *,
    name: str = "all",
) -> None:
    batch = dataset.batch
    if indices is None:
        indices = range(len(dataset))

    word_stats: dict[str, dict[str, list[float] | int]] = {
        word: {
            "total": 0,
            "full": 0,
            "partial_word_fracs": [],
            "word_probs": [],
            "max_probs": [],
        }
        for word in TARGET_WORDS
    }
    transition_stats: dict[str, dict[str, dict[str, list[float] | int]]] = {
        WORD_STARTING_LABEL: defaultdict(lambda: {"count": 0, "word_fracs": [], "label_probs": [], "max_probs": []}),
        WORD_ENDING_LABEL: defaultdict(lambda: {"count": 0, "word_fracs": [], "label_probs": [], "max_probs": []}),
        SILENCE_LABEL: defaultdict(lambda: {"count": 0, "word_fracs": [], "label_probs": [], "max_probs": []}),
    }

    for index in indices:
        hard_label = batch.labels[index]
        event_type = batch.event_types[index]
        event_id = batch.event_ids[index]
        label_probs = _label_probs_for_sample(batch, index)
        word_frac = _sample_word_fraction(event_type, event_id)
        max_prob = _label_max_prob(label_probs, hard_label)

        if hard_label in TARGET_WORDS:
            stats = word_stats[hard_label]
            stats["total"] = int(stats["total"]) + 1
            stats["word_probs"].append(_label_prob_mass(label_probs, hard_label, hard_label))
            stats["max_probs"].append(max_prob)
            if _is_full_word_label_sample(event_type, event_id, hard_label, hard_label):
                stats["full"] = int(stats["full"]) + 1
            else:
                stats["partial_word_fracs"].append(word_frac)

        if hard_label in transition_stats:
            context = _format_word_context(_associated_words(event_type, event_id))
            group = transition_stats[hard_label][context]
            group["count"] = int(group["count"]) + 1
            group["word_fracs"].append(word_frac)
            group["label_probs"].append(_label_prob_mass(label_probs, hard_label, hard_label))
            group["max_probs"].append(max_prob)

    print(f"{name} word label coverage (hard label = target word):")
    for word in TARGET_WORDS:
        stats = word_stats[word]
        total = int(stats["total"])
        if total == 0:
            print(f"  {word:12s}  no samples")
            continue
        full = int(stats["full"])
        partial = total - full
        partial_fracs = stats["partial_word_fracs"]
        avg_partial = (
            100.0 * sum(partial_fracs) / len(partial_fracs)
            if partial_fracs
            else 0.0
        )
        avg_word_prob = 100.0 * sum(stats["word_probs"]) / total
        avg_max_prob = 100.0 * sum(stats["max_probs"]) / total
        print(
            f"  {word:12s}  n={total:4d}  "
            f"full={full:4d} ({100.0 * full / total:5.1f}%)  "
            f"partial={partial:4d} ({100.0 * partial / total:5.1f}%)  "
            f"partial_avg_word={avg_partial:5.1f}%  "
            f"avg_P({word})={avg_word_prob:5.1f}%  "
            f"avg_max_prob={avg_max_prob:5.1f}%"
        )

    print()
    print(f"{name} transition / silence labels — word content in window:")
    for label in (WORD_STARTING_LABEL, WORD_ENDING_LABEL, SILENCE_LABEL):
        groups = transition_stats[label]
        total = sum(int(group["count"]) for group in groups.values())
        print(f"  {label} ({total} samples):")
        if total == 0:
            print("    (none)")
            continue
        for context in sorted(groups, key=lambda key: (-int(groups[key]["count"]), key)):
            group = groups[context]
            count = int(group["count"])
            avg_word = 100.0 * sum(group["word_fracs"]) / count
            avg_label = 100.0 * sum(group["label_probs"]) / count
            avg_max = 100.0 * sum(group["max_probs"]) / count
            print(
                f"    {context:24s}  n={count:4d} ({100.0 * count / total:5.1f}%)  "
                f"avg_word_in_window={avg_word:5.1f}%  "
                f"avg_P({label})={avg_label:5.1f}%  "
                f"avg_max_prob={avg_max:5.1f}%"
            )
    print()


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
    print(
        f"split balancing: {'stratified by label' if splits.stratified_label_split else 'random'}"
    )
    if splits.label_max_fractions:
        dropped = splits.label_cap_dropped
        print(
            "label caps: "
            + ", ".join(f"{label}={fraction:.0%}" for label, fraction in sorted(splits.label_max_fractions.items()))
        )
        print(
            f"dropped by caps: train={dropped[0]}, val={dropped[1]}, test={dropped[2]}"
        )
    print()
    print_label_coverage_summary(splits.dataset, name="all")
    for name, subset in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        counts = label_distribution(splits.dataset, subset.indices)
        print_label_distribution(name, counts, len(subset))
    print("\n\n")


# --- config (edit these)
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
        stratified_label_split=STRATIFIED_LABEL_SPLIT,
        label_max_fractions=LABEL_MAX_FRACTIONS,
        show_progress=True,
    )


    build_elapsed = time.perf_counter() - build_started
    save_started = time.perf_counter()

    save_dataset_splits(SPLITS_OUTPUT_DIR, splits)

    save_elapsed = time.perf_counter() - save_started
    total_elapsed = time.perf_counter() - total_started

    print(f"Saved splits to {SPLITS_OUTPUT_DIR.resolve()}")
    print(f"Timing: build={build_elapsed:.1f}s, save={save_elapsed:.1f}s, total={total_elapsed:.1f}s")
    print_split_summary(splits)