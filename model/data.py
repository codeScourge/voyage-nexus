from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from _preprocessors import (
    BandpassConfig,
    DEFAULT_EEG_BANDPASS_CONFIG,
    DEFAULT_EMG_BANDPASS_CONFIG,
    DEFAULT_LINE_NOISE_CONFIG,
    LineNoiseConfig,
    preprocess_session_channels,
)

EEG_RECORD_FORMAT = "<QQ32f"
EEG_RECORD_FORMAT_CODES_LEGACY = "<QQ32i"

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
        raise RuntimeError("No EEG frames in session")
    return frames


def load_events(session_dir: Path) -> list[dict[str, str]]:
    events_path = session_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    with events_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def build_event_windows(
    session_dir: Path,
    *,
    pre_ms: float = 300.0,
    post_ms: float = 700.0,
    event_types: Optional[set[str]] = None,
    line_noise: Optional[LineNoiseConfig] = DEFAULT_LINE_NOISE_CONFIG,
    eeg_bandpass: Optional[BandpassConfig] = DEFAULT_EEG_BANDPASS_CONFIG,
    emg_bandpass: Optional[BandpassConfig] = DEFAULT_EMG_BANDPASS_CONFIG,
    filter_order: int = 4,
) -> EventWindowBatch:
    """Cut fixed-length windows from session EEG aligned to event start samples."""
    session_dir = Path(session_dir)
    meta = load_session_meta(session_dir)
    frames = load_eeg_frames(session_dir)
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
    pre_samples, post_samples, window_len = window_sample_counts(fs, pre_ms, post_ms)

    index_to_row = {int(idx): row for row, idx in enumerate(sample_indices)}

    x_windows: list[np.ndarray] = []
    labels: list[str] = []
    event_type_list: list[str] = []
    event_ids: list[str] = []
    center_samples: list[int] = []
    skipped = 0

    for event in events:
        event_type = event.get("event_type", "")
        if event_types is not None and event_type not in event_types:
            continue

        sample_text = event.get("sample_index_start", "")
        if not sample_text:
            continue
        sample_idx = int(sample_text)
        row_idx = index_to_row.get(sample_idx)
        if row_idx is None:
            skipped += 1
            continue
        start = row_idx - pre_samples
        end = row_idx + post_samples + 1
        if start < 0 or end > channels.shape[0]:
            skipped += 1
            continue

        window = channels[start:end, :]
        if window.shape[0] != window_len:
            skipped += 1
            continue

        x_windows.append(window)
        labels.append(event.get("label_text", ""))
        event_type_list.append(event_type)
        event_ids.append(event.get("event_id", ""))
        center_samples.append(sample_idx)

    if x_windows:
        x = np.stack(x_windows, axis=0).astype(np.float32)
    else:
        x = np.zeros((0, window_len, channels.shape[1]), dtype=np.float32)

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

    return EventWindowBatch(
        x=np.concatenate([batch.x for batch in batches], axis=0),
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
    ) -> None:
        if isinstance(session_dirs, (str, Path)):
            dirs = [Path(session_dirs)]
        else:
            dirs = [Path(path) for path in session_dirs]
        if not dirs:
            raise ValueError("session_dirs must not be empty")

        batches = [
            build_event_windows(
                session_dir,
                pre_ms=pre_ms,
                post_ms=post_ms,
                event_types=event_types,
                line_noise=line_noise,
                eeg_bandpass=eeg_bandpass,
                emg_bandpass=emg_bandpass,
                filter_order=filter_order,
            )
            for session_dir in dirs
        ]
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
