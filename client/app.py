from __future__ import annotations

import argparse
import csv
import json
import queue
import random
import signal
import struct
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import serial
from flask import Flask, jsonify, render_template, request
from serial.tools import list_ports

_MODEL_ROOT = Path(__file__).resolve().parent.parent / "model"
if _MODEL_ROOT.is_dir() and str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from train import default_checkpoint_path, get_device
from val import load_checkpoint
from infer import fix_window_length, predict_fusion_batch
from _preprocessors import (
    DEFAULT_EEG_BANDPASS_CONFIG,
    DEFAULT_EMG_BANDPASS_CONFIG,
    DEFAULT_LINE_NOISE_CONFIG,
    preprocess_session_channels,
)
from _protocol import (
    CHANNEL_COUNT,
    CHANNEL_UNITS,
    EEG_LSB_UV,
    EEG_RECORD_FORMAT,
    EMG_LSB_UV,
    FRAME_SIZE,
    FrameParser,
    SampleFrame,
    ads_calibration_metadata,
)

import sounddevice as sd
import torch

try:
    import whisper
except:
    whiper = None

SAMPLE_RATE_HZ = 1000
WINDOW_SECONDS = 2
FILTER_BUFFER_SECONDS = 4
UPDATE_HZ = 30
CHANNELS_PER_PAGE = 8
PLOT_POINTS = 400
DEFAULT_BAUD = 2_000_000
DEFAULT_RECORDINGS_DIR = "recordings"
ALIGNMENT_WINDOW_FRAMES = 512
THINKING_COUNTDOWN_SECONDS = 3
COLLECTION_WORDS = ("highlight", "bullshit", "gogogo")
COLLECTION_REPETITIONS = 7
# Collection timing hyperparameters (seconds) — edit these to tune the protocol
COLLECTION_BEFORE_S = 1.0
COLLECTION_BETWEEN_S = 0.4
COLLECTION_SAY_S = 1.6
ALIGNMENT_TEST_PAD_S = 0.3
ALIGNMENT_TEST_DISPLAY_CHANNELS = tuple(range(8))
MODEL_TEST_TRIALS = 3
DEFAULT_SCRAMBLE_SET = 5
DEFAULT_SCRAMBLE_REP = 5
SCRAMBLE_SET_MIN = 1
SCRAMBLE_SET_MAX = 20
SCRAMBLE_REP_MIN = 1
SCRAMBLE_REP_MAX = 20
NEGATIVE_LABELS_STILL_FRACTION = 0.65
NEGATIVE_LABELS_NEG_WORD_FRACTION = 0.25
NEGATIVE_LABELS_POS_WORD_FRACTION = 0.10
NEGATIVE_LABELS_STILL_MIN_S = 1.0
NEGATIVE_LABELS_STILL_MAX_S = 20.0
NEGATIVE_LABEL_WORD_COUNT = 15
NEGATIVE_LABEL_WORD_POOL = (
    "sidewalk",
    "umbrella",
    "keyboard",
    "backward",
    "sunshine",
    "notebook",
    "sandwich",
    "football",
    "standard",
    "platform",
    "wildcard",
    "backpack",
    "handbook",
    "doorstep",
    "headline",
    "goldmine",
    "basement",
    "countdown",
    "bookmark",
    "lukewarm",
    "turnover",
    "backyard",
    "doorbell",
    "flashbulb",
    "handrail",
)
SPEECH_AUDIO_RATE_HZ = 16000
SPEECH_AUDIO_CHANNELS = 1
RAIL_CODE_WARN = 0x7FFF00
RAIL_WINDOW_FRAMES = 1000
RAIL_WARN_PERCENT = 10.0
DEFAULT_FIXED_SCALE_UV = 150.0
DEFAULT_HTTP_PORT = 5050
EEG_SWAP_HALVES = False
BAND_FRAME_SAMPLES = 512
BAND_HOP_SAMPLES = 128
BAND_PLOT_POINTS = 80

EEG_BANDS: tuple[dict[str, Any], ...] = (
    {"id": "delta", "name": "Delta", "range": "0.5–4 Hz", "low_hz": 0.5, "high_hz": 4.0, "color": "#9b59b6"},
    {"id": "theta", "name": "Theta", "range": "4–8 Hz", "low_hz": 4.0, "high_hz": 8.0, "color": "#3498db"},
    {"id": "mu", "name": "Mu", "range": "8–12 Hz", "low_hz": 8.0, "high_hz": 12.0, "color": "#1abc9c"},
    {"id": "alpha", "name": "Alpha", "range": "8–13 Hz", "low_hz": 8.0, "high_hz": 13.0, "color": "#2ecc71"},
    {"id": "beta", "name": "Beta", "range": "13–30 Hz", "low_hz": 13.0, "high_hz": 30.0, "color": "#f1c40f"},
    {"id": "gamma", "name": "Gamma", "range": "30–100 Hz", "low_hz": 30.0, "high_hz": 100.0, "color": "#e67e22"},
    {"id": "high_gamma", "name": "High Gamma", "range": ">100 Hz", "low_hz": 100.0, "high_hz": None, "color": "#e74c3c"},
)

EMG_BANDS: tuple[dict[str, Any], ...] = (
    {"id": "low", "name": "Low", "range": "20–60 Hz", "low_hz": 20.0, "high_hz": 60.0, "color": "#5dade2"},
    {"id": "mid", "name": "Mid", "range": "60–150 Hz", "low_hz": 60.0, "high_hz": 150.0, "color": "#58d68d"},
    {"id": "high", "name": "High", "range": "150–500 Hz", "low_hz": 150.0, "high_hz": 500.0, "color": "#f5b041"},
)


def channel_name(channel_idx: int) -> str:
    if channel_idx < 16:
        return f"EEG{channel_idx + 1}"
    return f"EMG{channel_idx - 15}"


def channel_order_dict(channel_count: int) -> dict[str, str]:
    return {str(i): channel_name(i) for i in range(channel_count)}


def new_session_dir_name(now: datetime | None = None) -> str:
    """e.g. 2026-06-15_21-27-46_session_a3f8b2c1"""
    when = now or datetime.now()
    uid = uuid.uuid4().hex[:8]
    stamp = when.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_session_{uid}"


def bands_for_channel(channel_idx: int) -> tuple[dict[str, Any], ...]:
    return EEG_BANDS if channel_idx < 16 else EMG_BANDS


def _ordered_buffer_view(buffer_uv: np.ndarray, write_idx: int) -> np.ndarray:
    """Return ring buffer contents in chronological order (oldest sample first)."""
    if write_idx == 0:
        return buffer_uv
    return np.concatenate(
        (buffer_uv[:, write_idx:], buffer_uv[:, :write_idx]),
        axis=1,
    )


def _preprocess_monitor_window(view: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Apply the same preprocessing as model/data.load_session_channels (line notch + band-pass)."""
    block = np.asarray(view, dtype=np.float32).T  # (time, channels)
    filtered = preprocess_session_channels(
        block,
        sample_rate_hz,
        line_noise=DEFAULT_LINE_NOISE_CONFIG,
        eeg=DEFAULT_EEG_BANDPASS_CONFIG,
        emg=DEFAULT_EMG_BANDPASS_CONFIG,
    )
    return filtered.T


def _band_power_series(
    signal: np.ndarray,
    sample_rate_hz: float,
    bands: tuple[dict[str, Any], ...],
    *,
    frame_samples: int = BAND_FRAME_SAMPLES,
    hop_samples: int = BAND_HOP_SAMPLES,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Short-time band power over the trailing window (one value per hop)."""
    n = int(signal.shape[0])
    if n < frame_samples:
        empty_t = np.linspace(-WINDOW_SECONDS, 0, 1, dtype=np.float32)
        empty_bands = [
            {"id": b["id"], "name": b["name"], "range": b["range"], "color": b["color"], "y": [0.0]}
            for b in bands
        ]
        return empty_t, empty_bands

    window = np.hanning(frame_samples).astype(np.float64)
    freqs = np.fft.rfftfreq(frame_samples, d=1.0 / sample_rate_hz)
    nyquist = sample_rate_hz / 2.0
    starts = np.arange(0, n - frame_samples + 1, hop_samples, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)

    band_powers: list[list[float]] = [[] for _ in bands]
    for start in starts:
        segment = signal[start : start + frame_samples].astype(np.float64)
        spectrum = np.fft.rfft(segment * window)
        power = (np.abs(spectrum) ** 2) / frame_samples
        for band_idx, band in enumerate(bands):
            low_hz = float(band["low_hz"])
            high_hz = band.get("high_hz")
            high_cut = nyquist if high_hz is None else min(float(high_hz), nyquist)
            mask = (freqs >= low_hz) & (freqs < high_cut)
            if not np.any(mask):
                band_powers[band_idx].append(0.0)
            else:
                band_powers[band_idx].append(float(power[mask].sum()))

    end_sample = int(starts[-1] + frame_samples)
    time_s = (starts.astype(np.float64) + frame_samples * 0.5 - end_sample) / sample_rate_hz
    series = [
        {
            "id": band["id"],
            "name": band["name"],
            "range": band["range"],
            "color": band["color"],
            "y": powers,
        }
        for band, powers in zip(bands, band_powers, strict=True)
    ]
    return time_s.astype(np.float32), series


def _format_band_debug(channel_idx: int, signal: np.ndarray, series: list[dict[str, Any]]) -> str:
    sig = np.asarray(signal, dtype=np.float64)
    lines = [
        (
            f"[BANDS] {channel_name(channel_idx)} "
            f"(preprocessed) mean={sig.mean():.1f} std={sig.std():.1f} "
            f"min={sig.min():.1f} max={sig.max():.1f} µV "
            f"({sig.size} samples)"
        )
    ]
    for band in series:
        y = np.asarray(band["y"], dtype=np.float64)
        latest = float(y[-1]) if y.size else 0.0
        lines.append(
            f"  {band['name']:11s} min={y.min():.3g} max={y.max():.3g} latest={latest:.3g}"
        )
    return "\n".join(lines)


def _downsample_band_series(
    time_s: np.ndarray,
    series: list[dict[str, Any]],
    max_points: int = BAND_PLOT_POINTS,
) -> tuple[list[float], list[dict[str, Any]]]:
    n = int(time_s.shape[0])
    if n <= max_points:
        return time_s.astype(np.float32).tolist(), [
            {
                "id": s["id"],
                "name": s["name"],
                "range": s["range"],
                "color": s["color"],
                "y": [float(v) for v in s["y"]],
            }
            for s in series
        ]
    step = max(1, n // max_points)
    idx = np.arange(0, n, step, dtype=np.int64)
    return time_s[idx].astype(np.float32).tolist(), [
        {
            "id": s["id"],
            "name": s["name"],
            "range": s["range"],
            "color": s["color"],
            "y": [float(s["y"][i]) for i in idx],
        }
        for s in series
    ]


class SerialReader(threading.Thread):
    def __init__(self, port: str, baudrate: int, out_queue: queue.Queue["RxFrame"]) -> None:
        super().__init__(daemon=True)
        self._port = port
        self._baudrate = baudrate
        self._out_queue = out_queue
        self._stop_event = threading.Event()
        self._parser = FrameParser()
        self.last_error: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            with serial.Serial(self._port, self._baudrate, timeout=0.05) as ser:
                while not self._stop_event.is_set():
                    chunk = ser.read(max(FRAME_SIZE * 8, 256))
                    if not chunk:
                        continue
                    for frame in self._parser.feed(chunk):
                        self._out_queue.put(RxFrame(frame=frame, host_rx_time_ns=time.perf_counter_ns()))
        except Exception as exc:  # pylint: disable=broad-except
            self.last_error = str(exc)


class TestReader(threading.Thread):
    EEG_AMPLITUDE_UV = 50.0
    EMG_AMPLITUDE_UV = 120.0

    def __init__(self, out_queue: queue.Queue["RxFrame"]) -> None:
        super().__init__(daemon=True)
        self._out_queue = out_queue
        self._stop_event = threading.Event()
        self.last_error: Optional[str] = None
        self._sample_index = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            interval_s = 1.0 / SAMPLE_RATE_HZ
            next_tick = time.perf_counter()
            while not self._stop_event.is_set():
                frame = self._make_frame(self._sample_index)
                self._out_queue.put(RxFrame(frame=frame, host_rx_time_ns=time.perf_counter_ns()))
                self._sample_index += 1
                next_tick += interval_s
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
        except Exception as exc:  # pylint: disable=broad-except
            self.last_error = str(exc)

    def _make_frame(self, sample_index: int) -> SampleFrame:
        t = sample_index / SAMPLE_RATE_HZ
        uv = np.zeros(CHANNEL_COUNT, dtype=np.float64)
        for ch in range(CHANNEL_COUNT):
            freq_hz = 2.0 + ch * 0.4
            amp = self.EEG_AMPLITUDE_UV if ch < 16 else self.EMG_AMPLITUDE_UV
            phase = ch * (np.pi / 10.0)
            uv[ch] = amp * np.sin(2.0 * np.pi * freq_hz * t + phase)

        codes = np.zeros(CHANNEL_COUNT, dtype=np.int32)
        codes[:16] = np.round(uv[:16] / EEG_LSB_UV).astype(np.int32)
        codes[16:] = np.round(uv[16:] / EMG_LSB_UV).astype(np.int32)
        return SampleFrame(
            mcu_time_us=sample_index * 1000,
            sample_index=sample_index,
            channels_i32=codes,
        )


@dataclass(slots=True)
class RxFrame:
    frame: SampleFrame
    host_rx_time_ns: int


class TimeAligner:
    def __init__(self, sample_rate_hz: int, max_points: int = ALIGNMENT_WINDOW_FRAMES) -> None:
        self._sample_rate_hz = float(sample_rate_hz)
        self._host_ns: deque[int] = deque(maxlen=max_points)
        self._sample_idx: deque[int] = deque(maxlen=max_points)
        self._latest_host_ns: Optional[int] = None
        self._latest_sample_idx: Optional[int] = None

    def update(self, sample_index: int, host_rx_time_ns: int) -> None:
        self._host_ns.append(host_rx_time_ns)
        self._sample_idx.append(sample_index)
        self._latest_host_ns = host_rx_time_ns
        self._latest_sample_idx = sample_index

    def estimate_sample_float(self, event_host_ns: int) -> Optional[float]:
        if len(self._host_ns) >= 2:
            x = np.asarray(self._host_ns, dtype=np.float64)
            y = np.asarray(self._sample_idx, dtype=np.float64)
            x0 = x[0]
            x_shift = x - x0
            vx = np.var(x_shift)
            if vx > 0.0:
                cov = np.cov(x_shift, y, bias=True)[0, 1]
                slope = cov / vx
                intercept = float(y.mean() - slope * x_shift.mean())
                return slope * (float(event_host_ns) - float(x0)) + intercept

        if self._latest_host_ns is None or self._latest_sample_idx is None:
            return None
        dt_s = (event_host_ns - self._latest_host_ns) / 1e9
        return float(self._latest_sample_idx) + dt_s * self._sample_rate_hz

    @staticmethod
    def quantize_sample_index(sample_float: Optional[float]) -> Optional[int]:
        if sample_float is None:
            return None
        return int(round(sample_float))


def say_window_sample_end(start_idx: int, start_float: float) -> tuple[int, float]:
    """End sample for a collection say window (COLLECTION_SAY_S at SAMPLE_RATE_HZ)."""
    end_float = start_float + COLLECTION_SAY_S * SAMPLE_RATE_HZ
    end_idx = TimeAligner.quantize_sample_index(end_float) or int(round(end_float))
    return end_idx, end_float


class SessionRecorder:
    def __init__(self, sample_rate_hz: int, channel_count: int) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._channel_count = channel_count
        self._session_dir: Optional[Path] = None
        self._eeg_handle = None
        self._events_handle = None
        self._events_writer: Optional[csv.writer] = None
        self._frames_since_flush = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._eeg_handle is not None and self._events_writer is not None

    @property
    def session_dir(self) -> Optional[Path]:
        with self._lock:
            return self._session_dir

    def start(self, base_dir: Path) -> Path:
        with self._lock:
            if self._eeg_handle is not None and self._events_writer is not None:
                if not self._session_dir:
                    raise RuntimeError("Recorder state invalid while enabled")
                return self._session_dir

            base_dir.mkdir(parents=True, exist_ok=True)
            base_name = new_session_dir_name()
            session_dir = base_dir / base_name
            suffix = 1
            while session_dir.exists():
                session_dir = base_dir / f"{base_name}_{suffix:02d}"
                suffix += 1
            session_dir.mkdir(parents=True, exist_ok=False)

            meta = {
                "created_at_iso": datetime.now(timezone.utc).isoformat(),
                "sample_rate_hz": self._sample_rate_hz,
                "channel_count": self._channel_count,
                "channel_units": CHANNEL_UNITS,
                "channel_order": channel_order_dict(self._channel_count),
                "ads_calibration": ads_calibration_metadata(),
                "files": {"eeg_frames": "eeg_frames.bin", "events": "events.csv"},
                "eeg_record_format": EEG_RECORD_FORMAT,
            }
            (session_dir / "session_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            self._session_dir = session_dir
            self._eeg_handle = (session_dir / "eeg_frames.bin").open("wb")
            self._events_handle = (session_dir / "events.csv").open("w", newline="", encoding="utf-8")
            self._events_writer = csv.writer(self._events_handle)
            self._events_writer.writerow(
                [
                    "event_id",
                    "event_type",
                    "label_text",
                    "host_time_iso",
                    "host_time_ns",
                    "sample_index_start",
                    "sample_index_start_float",
                    "sample_index_end",
                    "sample_index_end_float",
                    "confidence",
                    "alignment_method",
                    "payload_json",
                ]
            )
            self._events_handle.flush()
            return session_dir

    def stop(self) -> None:
        with self._lock:
            if self._eeg_handle is not None:
                self._eeg_handle.flush()
                self._eeg_handle.close()
            if self._events_handle is not None:
                self._events_handle.flush()
                self._events_handle.close()
            self._session_dir = None
            self._eeg_handle = None
            self._events_handle = None
            self._events_writer = None
            self._frames_since_flush = 0

    def append_frame(self, frame: SampleFrame) -> None:
        with self._lock:
            if not self._eeg_handle:
                return
            channels_uv = frame.channels_uv().astype(np.float32, copy=False).tolist()
            payload = struct.pack(
                EEG_RECORD_FORMAT,
                int(frame.sample_index),
                int(frame.mcu_time_us),
                *channels_uv,
            )
            self._eeg_handle.write(payload)
            self._frames_since_flush += 1
            if self._frames_since_flush >= 250:
                self._eeg_handle.flush()
                self._frames_since_flush = 0

    def log_event(
        self,
        event_type: str,
        label_text: str = "",
        *,
        sample_index_start: Optional[int] = None,
        sample_index_start_float: Optional[float] = None,
        sample_index_end: Optional[int] = None,
        sample_index_end_float: Optional[float] = None,
        confidence: Optional[float] = None,
        alignment_method: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if not self._events_writer or not self._events_handle:
                return
            host_iso = datetime.now(timezone.utc).isoformat()
            host_ns = time.time_ns()
            self._events_writer.writerow(
                [
                    uuid.uuid4().hex,
                    event_type,
                    label_text,
                    host_iso,
                    host_ns,
                    "" if sample_index_start is None else sample_index_start,
                    "" if sample_index_start_float is None else f"{sample_index_start_float:.6f}",
                    "" if sample_index_end is None else sample_index_end,
                    "" if sample_index_end_float is None else f"{sample_index_end_float:.6f}",
                    "" if confidence is None else f"{confidence:.6f}",
                    alignment_method,
                    json.dumps(payload or {}, separators=(",", ":")),
                ]
            )
            self._events_handle.flush()


class SpeechAudioRecorder:
    def __init__(self, samplerate: int, channels: int) -> None:
        self._samplerate = samplerate
        self._channels = channels
        self._stream = None
        self._file_handle = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self, wav_path: Path) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")
        if self._stream is not None:
            return

        import wave

        wav_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = wave.open(str(wav_path), "wb")
        self._file_handle.setnchannels(self._channels)
        self._file_handle.setsampwidth(2)
        self._file_handle.setframerate(self._samplerate)

        def _on_audio(indata: np.ndarray, frames: int, _time_info: Any, _status: Any) -> None:
            if frames <= 0:
                return
            with self._lock:
                if self._file_handle is not None:
                    self._file_handle.writeframes(indata.tobytes())

        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            channels=self._channels,
            dtype="int16",
            callback=_on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None


class TrialState(Enum):
    IDLE = "idle"
    THINKING_COUNTDOWN = "thinking_countdown"
    THINKING_ACTIVE = "thinking_active"
    SPEECH_ACTIVE = "speech_active"


class AcquisitionService:
    def __init__(
        self,
        *,
        serial_port: Optional[str] = None,
        baudrate: int = DEFAULT_BAUD,
        test_mode: bool = False,
        debug_bands: bool = False,
    ) -> None:
        self._test_mode = test_mode
        self._debug_bands = debug_bands
        self._band_debug_last_print_ns = 0
        self._queue: queue.Queue[RxFrame] = queue.Queue(maxsize=20_000)
        if test_mode:
            self._reader: SerialReader | TestReader = TestReader(out_queue=self._queue)
        else:
            if not serial_port:
                raise ValueError("serial_port is required unless test_mode is enabled")
            self._reader = SerialReader(port=serial_port, baudrate=baudrate, out_queue=self._queue)
        self._reader.start()
        self._pump_stop = threading.Event()

        self._recorder = SessionRecorder(sample_rate_hz=SAMPLE_RATE_HZ, channel_count=CHANNEL_COUNT)
        self._aligner = TimeAligner(sample_rate_hz=SAMPLE_RATE_HZ)
        self._speech_recorder = SpeechAudioRecorder(SPEECH_AUDIO_RATE_HZ, SPEECH_AUDIO_CHANNELS)
        self._asr_results_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._whisper_model = None

        self._trial_state = TrialState.IDLE
        self._current_trial_id: Optional[str] = None
        self._current_trial_word = ""
        self._speech_block_id: Optional[str] = None
        self._speech_block_start_sample_float: Optional[float] = None
        self._speech_audio_path: Optional[Path] = None
        self._last_session_dir: Optional[Path] = None
        self._thinking_countdown_deadline_ns: Optional[int] = None
        self._pending_asr_jobs = 0

        self._collect_phase = "disabled"
        self._collect_mode = "single"
        self._collect_word = ""
        self._collect_rep = 0
        self._collect_reps_total = COLLECTION_REPETITIONS
        self._collect_set_idx = 0
        self._collect_set_total = 0
        self._collect_prev_word = ""
        self._collect_block_id: Optional[str] = None
        self._collect_deadline_ns: Optional[int] = None
        self._collect_countdown_word_switch = False
        self._collect_neg_words: list[str] = []
        self._collect_neg_segment_kind = ""
        self._collect_neg_still_time_s = 0.0
        self._collect_neg_negative_word_time_s = 0.0
        self._collect_neg_positive_word_time_s = 0.0
        self._collect_neg_segment_started_ns: Optional[int] = None

        self._align_test_phase = "idle"
        self._align_test_deadline_ns: Optional[int] = None
        self._align_test_say_start_idx: Optional[int] = None
        self._align_test_say_start_float: Optional[float] = None
        self._align_test_say_end_idx: Optional[int] = None
        self._align_test_say_end_float: Optional[float] = None
        self._align_test_method = ""
        self._align_test_result: Optional[dict[str, Any]] = None

        self._model_test_phase = "idle"
        self._model_test_deadline_ns: Optional[int] = None
        self._model_test_trial = 0
        self._model_test_trials_total = MODEL_TEST_TRIALS
        self._model_test_captures: list[dict[str, Any]] = []
        self._model_test_result: Optional[dict[str, Any]] = None
        self._model_test_say_start_idx: Optional[int] = None
        self._model_test_say_start_float: Optional[float] = None
        self._model_test_nn = None
        self._model_test_label_to_idx: Optional[dict[str, int]] = None
        self._model_test_idx_to_label: Optional[dict[int, str]] = None
        self._model_test_checkpoint: Optional[str] = None
        self._model_test_model_config: Optional[dict[str, Any]] = None
        self._model_test_device = None

        self._latest_sample_index = -1
        self._total_frames = 0
        self._buffer_samples = FILTER_BUFFER_SECONDS * SAMPLE_RATE_HZ
        self._display_samples = WINDOW_SECONDS * SAMPLE_RATE_HZ
        self._buffer_uv = np.zeros((CHANNEL_COUNT, self._buffer_samples), dtype=np.float32)
        self._write_idx = 0
        self._rail_history = np.zeros((CHANNEL_COUNT, RAIL_WINDOW_FRAMES), dtype=np.uint8)
        self._rail_counts = np.zeros(CHANNEL_COUNT, dtype=np.int32)
        self._rail_history_idx = 0
        self._rail_history_filled = 0
        self._lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False

        self._pump_thread = threading.Thread(target=self._pump, daemon=True)
        self._pump_thread.start()

    def _pump(self) -> None:
        while not self._pump_stop.is_set():
            self._drain_asr_results()
            try:
                rx_frame = self._queue.get(timeout=0.05)
            except queue.Empty:
                self._tick_countdown()
                self._tick_collect()
                self._tick_alignment_test()
                self._tick_model_test()
                continue
            self._push_frame(rx_frame)
            self._tick_countdown()
            self._tick_collect()
            self._tick_alignment_test()
            self._tick_model_test()

        while True:
            try:
                rx_frame = self._queue.get_nowait()
            except queue.Empty:
                break
            self._push_frame(rx_frame)

    def _finalize_recording_on_shutdown(self, reason: str) -> None:
        if not self._recorder.enabled:
            return
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        self._recorder.log_event(
            event_type="forced_shutdown",
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={"source": "host", "reason": reason},
        )
        session_dir = self._recorder.session_dir
        self._recorder.stop()
        self._reset_collect()
        if session_dir:
            self._last_session_dir = session_dir

    def shutdown(self, *, recording_shutdown_reason: str = "shutdown") -> None:
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

        self._reader.stop()
        self._pump_stop.set()
        self._pump_thread.join(timeout=2.0)
        self._speech_recorder.stop()
        if self._recorder.enabled:
            self._finalize_recording_on_shutdown(recording_shutdown_reason)
        self._reader.join(timeout=1.0)

    def _aligned_sample_now(self) -> tuple[int, Optional[float], Optional[int], str]:
        event_host_ns = time.perf_counter_ns()
        aligned_float = self._aligner.estimate_sample_float(event_host_ns)
        aligned_idx = self._aligner.quantize_sample_index(aligned_float)
        method = "host_time_regression" if aligned_float is not None else ""
        return event_host_ns, aligned_float, aligned_idx, method

    def _push_frame(self, rx_frame: RxFrame) -> None:
        frame = self._remap_frame_channels(rx_frame.frame)
        uv = frame.channels_uv()
        railed = (np.abs(frame.channels_i32) >= RAIL_CODE_WARN).astype(np.uint8)
        with self._lock:
            self._buffer_uv[:, self._write_idx] = uv
            self._write_idx = (self._write_idx + 1) % self._buffer_samples
            self._latest_sample_index = frame.sample_index
            self._total_frames += 1
            old = self._rail_history[:, self._rail_history_idx]
            self._rail_counts += railed.astype(np.int32) - old.astype(np.int32)
            self._rail_history[:, self._rail_history_idx] = railed
            self._rail_history_idx = (self._rail_history_idx + 1) % RAIL_WINDOW_FRAMES
            self._rail_history_filled = min(self._rail_history_filled + 1, RAIL_WINDOW_FRAMES)
        self._aligner.update(sample_index=frame.sample_index, host_rx_time_ns=rx_frame.host_rx_time_ns)
        self._recorder.append_frame(frame)

    def _remap_frame_channels(self, frame: SampleFrame) -> SampleFrame:
        if self._test_mode or not EEG_SWAP_HALVES:
            return frame
        remapped = frame.channels_i32.copy()
        remapped[0:8] = frame.channels_i32[8:16]
        remapped[8:16] = frame.channels_i32[0:8]
        return SampleFrame(
            mcu_time_us=frame.mcu_time_us,
            sample_index=frame.sample_index,
            channels_i32=remapped,
        )

    def _tick_countdown(self) -> None:
        if self._trial_state != TrialState.THINKING_COUNTDOWN:
            return
        deadline = self._thinking_countdown_deadline_ns
        if deadline is None or time.perf_counter_ns() < deadline:
            return
        _event_host_ns, start_float, start_idx, start_method = self._aligned_sample_now()
        if start_idx is None or not self._current_trial_id:
            self._trial_state = TrialState.IDLE
            self._thinking_countdown_deadline_ns = None
            return
        self._trial_state = TrialState.THINKING_ACTIVE
        self._thinking_countdown_deadline_ns = None
        self._recorder.log_event(
            event_type="thinking_trial_start",
            label_text=self._current_trial_word,
            sample_index_start=start_idx,
            sample_index_start_float=start_float,
            alignment_method=start_method,
            payload={"trial_id": self._current_trial_id},
        )

    def _rail_warning(self) -> dict[str, Any]:
        with self._lock:
            filled = self._rail_history_filled
            counts = self._rail_counts.copy()
        if filled == 0:
            return {"level": "waiting", "text": "Rail warning: waiting for data"}
        denom = float(filled)
        rail_pct = (counts.astype(np.float32) / denom) * 100.0
        flagged = np.where(rail_pct >= RAIL_WARN_PERCENT)[0]
        if flagged.size == 0:
            return {"level": "ok", "text": "Rail warning: none"}
        labels: list[str] = []
        for idx in flagged[:8]:
            group = "EEG" if idx < 16 else "EMG"
            ch = (idx + 1) if idx < 16 else (idx - 15)
            labels.append(f"{group}{ch} {rail_pct[idx]:.0f}%")
        extra = "" if flagged.size <= 8 else f" +{flagged.size - 8} more"
        return {"level": "warn", "text": "Rail warning: " + ", ".join(labels) + extra}

    def _thinking_countdown_remaining(self) -> Optional[int]:
        if self._trial_state != TrialState.THINKING_COUNTDOWN:
            return None
        deadline = self._thinking_countdown_deadline_ns
        if deadline is None:
            return None
        remaining_s = (deadline - time.perf_counter_ns()) / 1e9
        return max(0, int(remaining_s + 0.999))

    def _collect_phase_remaining_s(self) -> Optional[float]:
        if self._collect_phase not in ("countdown", "say", "still"):
            return None
        if self._collect_deadline_ns is None:
            return None
        return max(0.0, (self._collect_deadline_ns - time.perf_counter_ns()) / 1e9)

    @staticmethod
    def _countdown_seconds(*, new_label: bool, word_switch: bool = False) -> float:
        if not new_label:
            return COLLECTION_BETWEEN_S
        if word_switch:
            return COLLECTION_BEFORE_S
        return COLLECTION_BEFORE_S

    def _begin_collect_countdown(self, *, new_label: bool, word_switch: bool = False) -> None:
        self._collect_phase = "countdown"
        self._collect_countdown_word_switch = word_switch
        countdown_s = self._countdown_seconds(new_label=new_label, word_switch=word_switch)
        self._collect_deadline_ns = time.perf_counter_ns() + int(countdown_s * 1_000_000_000)

    def _pick_scramble_word(self) -> str:
        choices = [word for word in COLLECTION_WORDS if word != self._collect_prev_word]
        if not choices:
            choices = list(COLLECTION_WORDS)
        return random.choice(choices)

    def _pick_negative_label_words(self) -> list[str]:
        pool = list(NEGATIVE_LABEL_WORD_POOL)
        random.shuffle(pool)
        return pool[:NEGATIVE_LABEL_WORD_COUNT]

    def _pick_negative_labels_segment(self) -> str:
        targets = {
            "still": NEGATIVE_LABELS_STILL_FRACTION,
            "negative_word": NEGATIVE_LABELS_NEG_WORD_FRACTION,
            "positive_word": NEGATIVE_LABELS_POS_WORD_FRACTION,
        }
        elapsed = {
            "still": self._collect_neg_still_time_s,
            "negative_word": self._collect_neg_negative_word_time_s,
            "positive_word": self._collect_neg_positive_word_time_s,
        }
        total_elapsed = sum(elapsed.values())
        if total_elapsed <= 0:
            roll = random.random()
            if roll < targets["still"]:
                return "still"
            if roll < targets["still"] + targets["negative_word"]:
                return "negative_word"
            return "positive_word"

        deficits = {kind: targets[kind] - (elapsed[kind] / total_elapsed) for kind in targets}
        weights = {kind: max(deficits[kind], 0.01) for kind in targets}
        kinds = list(weights.keys())
        return random.choices(kinds, weights=[weights[kind] for kind in kinds], k=1)[0]

    def _mark_negative_labels_segment_start(self) -> None:
        self._collect_neg_segment_started_ns = time.perf_counter_ns()

    def _record_negative_labels_segment_elapsed(self, segment_kind: str) -> None:
        started_ns = self._collect_neg_segment_started_ns
        if started_ns is None:
            return
        elapsed_s = max(0.0, (time.perf_counter_ns() - started_ns) / 1e9)
        self._collect_neg_segment_started_ns = None
        if segment_kind == "still":
            self._collect_neg_still_time_s += elapsed_s
        elif segment_kind == "negative_word":
            self._collect_neg_negative_word_time_s += elapsed_s
        elif segment_kind == "positive_word":
            self._collect_neg_positive_word_time_s += elapsed_s

    def _begin_negative_labels_still(self) -> None:
        duration_s = random.uniform(NEGATIVE_LABELS_STILL_MIN_S, NEGATIVE_LABELS_STILL_MAX_S)
        self._collect_neg_segment_kind = "still"
        self._collect_word = ""
        self._collect_phase = "still"
        self._collect_deadline_ns = time.perf_counter_ns() + int(duration_s * 1_000_000_000)
        self._mark_negative_labels_segment_start()

    def _begin_negative_labels_word_segment(self, segment_kind: str) -> None:
        if segment_kind == "negative_word":
            self._collect_word = random.choice(self._collect_neg_words)
        else:
            self._collect_word = random.choice(COLLECTION_WORDS)
        self._collect_neg_segment_kind = segment_kind
        self._collect_rep = 1
        self._collect_reps_total = 1
        self._begin_collect_countdown(new_label=True)
        self._mark_negative_labels_segment_start()

    def _begin_negative_labels_segment(self) -> None:
        segment_kind = self._pick_negative_labels_segment()
        if segment_kind == "still":
            self._begin_negative_labels_still()
        else:
            self._begin_negative_labels_word_segment(segment_kind)

    def _collect_status(self) -> dict[str, Any]:
        busy = self._collect_phase in ("countdown", "say", "still")
        negative_labels_active = self._collect_mode == "negative_labels" and busy
        return {
            "phase": self._collect_phase,
            "mode": self._collect_mode if busy else None,
            "word": self._collect_word or None,
            "repetition": self._collect_rep if busy else None,
            "repetitions_total": self._collect_reps_total if busy else COLLECTION_REPETITIONS,
            "set_index": self._collect_set_idx if busy and self._collect_mode == "scramble" else None,
            "sets_total": self._collect_set_total if busy and self._collect_mode == "scramble" else None,
            "neg_segment_kind": self._collect_neg_segment_kind if negative_labels_active else None,
            "negative_labels_active": negative_labels_active,
            "phase_remaining_s": self._collect_phase_remaining_s(),
            "words": list(COLLECTION_WORDS),
            "before_s": COLLECTION_BEFORE_S,
            "between_s": COLLECTION_BETWEEN_S,
            "word_switch_s": COLLECTION_BEFORE_S,
            "say_s": COLLECTION_SAY_S,
            "word_switch": self._collect_countdown_word_switch if self._collect_phase == "countdown" else False,
            "default_scramble_set": DEFAULT_SCRAMBLE_SET,
            "default_scramble_rep": DEFAULT_SCRAMBLE_REP,
            "scramble_set_min": SCRAMBLE_SET_MIN,
            "scramble_set_max": SCRAMBLE_SET_MAX,
            "scramble_rep_min": SCRAMBLE_REP_MIN,
            "scramble_rep_max": SCRAMBLE_REP_MAX,
        }

    def _reset_collect(self) -> None:
        self._collect_phase = "disabled"
        self._collect_mode = "single"
        self._collect_word = ""
        self._collect_rep = 0
        self._collect_reps_total = COLLECTION_REPETITIONS
        self._collect_set_idx = 0
        self._collect_set_total = 0
        self._collect_prev_word = ""
        self._collect_block_id = None
        self._collect_deadline_ns = None
        self._collect_countdown_word_switch = False
        self._collect_neg_words = []
        self._collect_neg_segment_kind = ""
        self._collect_neg_still_time_s = 0.0
        self._collect_neg_negative_word_time_s = 0.0
        self._collect_neg_positive_word_time_s = 0.0
        self._collect_neg_segment_started_ns = None

    def _finish_collect_block(self) -> None:
        finished_word = self._collect_word
        finished_block = self._collect_block_id
        finished_mode = self._collect_mode
        self._collect_phase = "pick_word"
        self._collect_word = ""
        self._collect_rep = 0
        self._collect_reps_total = COLLECTION_REPETITIONS
        self._collect_set_idx = 0
        self._collect_set_total = 0
        self._collect_prev_word = ""
        self._collect_block_id = None
        self._collect_deadline_ns = None
        self._collect_countdown_word_switch = False
        self._collect_mode = "single"
        if finished_block and self._recorder.enabled:
            _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
            if aligned_idx is not None:
                self._recorder.log_event(
                    event_type="silent_speech_block_end",
                    label_text=finished_word,
                    sample_index_start=aligned_idx,
                    sample_index_start_float=aligned_float,
                    alignment_method=method,
                    payload={
                        "collection_block_id": finished_block,
                        "word": finished_word,
                        "mode": finished_mode,
                    },
                )

    def _tick_collect(self) -> None:
        if self._collect_phase not in ("countdown", "say", "still"):
            return
        deadline = self._collect_deadline_ns
        if deadline is None or time.perf_counter_ns() < deadline:
            return

        if self._collect_phase == "still":
            self._record_negative_labels_segment_elapsed("still")
            if self._collect_mode == "negative_labels":
                self._begin_negative_labels_segment()
            return

        if self._collect_phase == "countdown":
            self._collect_phase = "say"
            self._collect_countdown_word_switch = False
            self._collect_deadline_ns = time.perf_counter_ns() + int(COLLECTION_SAY_S * 1_000_000_000)
            self._log_silent_speech_rep()
            return

        if self._collect_mode == "negative_labels":
            self._record_negative_labels_segment_elapsed(self._collect_neg_segment_kind)
            self._begin_negative_labels_segment()
            return

        if self._collect_rep < self._collect_reps_total:
            self._collect_rep += 1
            self._begin_collect_countdown(new_label=False)
            return

        self._collect_prev_word = self._collect_word
        if self._collect_mode == "scramble" and self._collect_set_idx < self._collect_set_total:
            self._collect_set_idx += 1
            self._collect_word = self._pick_scramble_word()
            self._collect_rep = 1
            self._begin_collect_countdown(new_label=True, word_switch=True)
            self._log_scramble_word()
            return

        self._finish_collect_block()

    def _log_silent_speech_rep(self) -> None:
        if not self._recorder.enabled or not self._collect_block_id:
            return
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            return
        end_idx, end_float = say_window_sample_end(aligned_idx, aligned_float)
        self._recorder.log_event(
            event_type="silent_speech_word",
            label_text=self._collect_word,
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            sample_index_end=end_idx,
            sample_index_end_float=end_float,
            alignment_method=method,
            payload={
                "collection_block_id": self._collect_block_id,
                "repetition": self._collect_rep,
                "word": self._collect_word,
                "mode": self._collect_mode,
                "set_index": self._collect_set_idx if self._collect_mode == "scramble" else None,
                "sets_total": self._collect_set_total if self._collect_mode == "scramble" else None,
            },
        )

    def _log_scramble_word(self) -> None:
        if not self._recorder.enabled or not self._collect_block_id:
            return
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            return
        self._recorder.log_event(
            event_type="silent_speech_scramble_word",
            label_text=self._collect_word,
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={
                "collection_block_id": self._collect_block_id,
                "word": self._collect_word,
                "set_index": self._collect_set_idx,
                "sets_total": self._collect_set_total,
                "repetitions_planned": self._collect_reps_total,
            },
        )

    def start_collect_word(self, word: str) -> None:
        if not self._recorder.enabled:
            raise RuntimeError("Start session recording first")
        if self._collect_phase != "pick_word":
            raise RuntimeError("Finish the current collection before choosing another")
        if self._trial_state != TrialState.IDLE:
            raise RuntimeError("Another trial is already active")
        word = word.strip().lower()
        if word not in COLLECTION_WORDS:
            raise RuntimeError(f"Word must be one of: {', '.join(COLLECTION_WORDS)}")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet; cannot start collection")

        block_id = uuid.uuid4().hex
        self._collect_block_id = block_id
        self._collect_mode = "single"
        self._collect_reps_total = COLLECTION_REPETITIONS
        self._collect_set_idx = 0
        self._collect_set_total = 0
        self._collect_prev_word = ""
        self._collect_word = word
        self._collect_rep = 1
        self._begin_collect_countdown(new_label=True)
        self._recorder.log_event(
            event_type="silent_speech_block_start",
            label_text=word,
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={
                "collection_block_id": block_id,
                "word": word,
                "mode": "single",
                "repetitions_planned": COLLECTION_REPETITIONS,
            },
        )

    def start_collect_scramble(self, set_count: int, rep_count: int) -> None:
        if not self._recorder.enabled:
            raise RuntimeError("Start session recording first")
        if self._collect_phase != "pick_word":
            raise RuntimeError("Finish the current collection before starting another")
        if self._trial_state != TrialState.IDLE:
            raise RuntimeError("Another trial is already active")
        if not SCRAMBLE_SET_MIN <= set_count <= SCRAMBLE_SET_MAX:
            raise RuntimeError(f"set must be between {SCRAMBLE_SET_MIN} and {SCRAMBLE_SET_MAX}")
        if not SCRAMBLE_REP_MIN <= rep_count <= SCRAMBLE_REP_MAX:
            raise RuntimeError(f"rep must be between {SCRAMBLE_REP_MIN} and {SCRAMBLE_REP_MAX}")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet; cannot start collection")

        block_id = uuid.uuid4().hex
        self._collect_prev_word = ""
        word = self._pick_scramble_word()
        self._collect_block_id = block_id
        self._collect_mode = "scramble"
        self._collect_reps_total = rep_count
        self._collect_set_total = set_count
        self._collect_set_idx = 1
        self._collect_word = word
        self._collect_rep = 1
        self._begin_collect_countdown(new_label=True)
        self._recorder.log_event(
            event_type="silent_speech_scramble_start",
            label_text=word,
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={
                "collection_block_id": block_id,
                "word": word,
                "mode": "scramble",
                "sets_planned": set_count,
                "repetitions_per_set": rep_count,
            },
        )
        self._log_scramble_word()

    def start_collect_negative_labels(self) -> None:
        if not self._recorder.enabled:
            raise RuntimeError("Start session recording first")
        if self._collect_phase != "pick_word":
            raise RuntimeError("Finish the current collection before starting another")
        if self._trial_state != TrialState.IDLE:
            raise RuntimeError("Another trial is already active")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet; cannot start collection")

        block_id = uuid.uuid4().hex
        self._collect_block_id = block_id
        self._collect_mode = "negative_labels"
        self._collect_neg_words = self._pick_negative_label_words()
        self._collect_neg_segment_kind = ""
        self._collect_neg_still_time_s = 0.0
        self._collect_neg_negative_word_time_s = 0.0
        self._collect_neg_positive_word_time_s = 0.0
        self._collect_neg_segment_started_ns = None
        self._recorder.log_event(
            event_type="silent_speech_block_start",
            label_text="",
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={
                "collection_block_id": block_id,
                "mode": "negative_labels",
                "negative_words": list(self._collect_neg_words),
                "still_fraction": NEGATIVE_LABELS_STILL_FRACTION,
                "negative_word_fraction": NEGATIVE_LABELS_NEG_WORD_FRACTION,
                "positive_word_fraction": NEGATIVE_LABELS_POS_WORD_FRACTION,
            },
        )
        self._begin_negative_labels_segment()

    def stop_collect_negative_labels(self) -> None:
        if self._collect_mode != "negative_labels":
            raise RuntimeError("Negative labels mode is not active")
        if self._collect_phase == "still":
            self._record_negative_labels_segment_elapsed("still")
        elif self._collect_phase == "say" and self._collect_neg_segment_kind in ("negative_word", "positive_word"):
            self._record_negative_labels_segment_elapsed(self._collect_neg_segment_kind)
        block_id = self._collect_block_id
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is not None and block_id and self._recorder.enabled:
            self._recorder.log_event(
                event_type="silent_speech_block_end",
                label_text="",
                sample_index_start=aligned_idx,
                sample_index_start_float=aligned_float,
                alignment_method=method,
                payload={
                    "collection_block_id": block_id,
                    "mode": "negative_labels",
                },
            )
        self._collect_phase = "pick_word"
        self._collect_mode = "single"
        self._collect_word = ""
        self._collect_rep = 0
        self._collect_reps_total = COLLECTION_REPETITIONS
        self._collect_set_idx = 0
        self._collect_set_total = 0
        self._collect_prev_word = ""
        self._collect_block_id = None
        self._collect_deadline_ns = None
        self._collect_countdown_word_switch = False
        self._collect_neg_words = []
        self._collect_neg_segment_kind = ""
        self._collect_neg_still_time_s = 0.0
        self._collect_neg_negative_word_time_s = 0.0
        self._collect_neg_positive_word_time_s = 0.0
        self._collect_neg_segment_started_ns = None

    def toggle_collect_negative_labels(self) -> None:
        if self._collect_mode == "negative_labels":
            self.stop_collect_negative_labels()
        else:
            self.start_collect_negative_labels()

    def _alignment_test_busy(self) -> bool:
        return self._align_test_phase in ("countdown", "blink")

    def _reset_alignment_test(self) -> None:
        self._align_test_phase = "idle"
        self._align_test_deadline_ns = None
        self._align_test_say_start_idx = None
        self._align_test_say_start_float = None
        self._align_test_say_end_idx = None
        self._align_test_say_end_float = None
        self._align_test_method = ""
        self._align_test_result = None

    def _alignment_test_phase_remaining_s(self) -> Optional[float]:
        if self._align_test_phase not in ("countdown", "blink"):
            return None
        if self._align_test_deadline_ns is None:
            return None
        return max(0.0, (self._align_test_deadline_ns - time.perf_counter_ns()) / 1e9)

    def _alignment_test_status(self) -> dict[str, Any]:
        return {
            "phase": self._align_test_phase,
            "phase_remaining_s": self._alignment_test_phase_remaining_s(),
            "before_s": COLLECTION_BEFORE_S,
            "say_s": COLLECTION_SAY_S,
            "has_result": self._align_test_result is not None,
        }

    def _extract_samples_by_index_range(
        self,
        start_idx: int,
        end_idx: int,
    ) -> Optional[tuple[np.ndarray, int]]:
        with self._lock:
            latest = self._latest_sample_index
            filled = min(self._total_frames, self._buffer_samples)
            if filled <= 0 or latest < 0:
                return None
            oldest_idx = latest - filled + 1
            if end_idx < oldest_idx or start_idx > latest:
                return None
            clip_start = max(start_idx, oldest_idx)
            clip_end = min(end_idx, latest)
            view = _ordered_buffer_view(self._buffer_uv, self._write_idx)
            i0 = clip_start - oldest_idx
            i1 = clip_end - oldest_idx + 1
            return view[:, i0:i1].copy(), clip_start

    def _build_alignment_test_result(self) -> dict[str, Any]:
        if self._align_test_say_start_float is None or self._align_test_say_start_idx is None:
            return {"error": "alignment test missing label start"}

        say_start_f = float(self._align_test_say_start_float)
        say_start_i = int(self._align_test_say_start_idx)
        say_end_f = say_start_f + COLLECTION_SAY_S * SAMPLE_RATE_HZ
        say_end_i = TimeAligner.quantize_sample_index(say_end_f) or int(round(say_end_f))

        pad_samples = int(round(ALIGNMENT_TEST_PAD_S * SAMPLE_RATE_HZ))
        display_start = say_start_i - pad_samples
        display_end = say_end_i + pad_samples
        extracted = self._extract_samples_by_index_range(display_start, display_end)
        if extracted is None:
            return {
                "error": "labeled window is no longer in the ring buffer",
                "sample_index_start": say_start_i,
                "sample_index_end": say_end_i,
                "sample_index_start_float": say_start_f,
                "sample_index_end_float": say_end_f,
                "alignment_method": self._align_test_method,
            }

        raw, clip_start = extracted
        filtered = _preprocess_monitor_window(raw, SAMPLE_RATE_HZ)
        n = int(raw.shape[1])
        # Seconds ago relative to the newest sample in the clip (0 = now), same as live waveform plots.
        time_s = (np.arange(n, dtype=np.float32) + (clip_start - display_end)) / SAMPLE_RATE_HZ
        label_t0_s = (say_start_i - display_end) / SAMPLE_RATE_HZ
        label_t1_s = (say_end_i - display_end) / SAMPLE_RATE_HZ

        step = max(1, n // PLOT_POINTS)
        time_ds = time_s[::step]
        traces: list[dict[str, Any]] = []
        for channel_idx in ALIGNMENT_TEST_DISPLAY_CHANNELS:
            traces.append(
                {
                    "index": channel_idx,
                    "name": channel_name(channel_idx),
                    "y": raw[channel_idx, ::step].astype(np.float32).tolist(),
                    "y_filtered": filtered[channel_idx, ::step].astype(np.float32).tolist(),
                }
            )

        return {
            "sample_index_start": say_start_i,
            "sample_index_end": say_end_i,
            "sample_index_start_float": say_start_f,
            "sample_index_end_float": say_end_f,
            "alignment_method": self._align_test_method,
            "label_t0_s": float(label_t0_s),
            "label_t1_s": float(label_t1_s),
            "time_s": time_ds.astype(np.float32).tolist(),
            "traces": traces,
            "before_s": COLLECTION_BEFORE_S,
            "say_s": COLLECTION_SAY_S,
            "pad_s": ALIGNMENT_TEST_PAD_S,
        }

    def _tick_alignment_test(self) -> None:
        if self._align_test_phase not in ("countdown", "blink"):
            return
        deadline = self._align_test_deadline_ns
        if deadline is None or time.perf_counter_ns() < deadline:
            return

        if self._align_test_phase == "countdown":
            _event_host_ns, start_float, start_idx, method = self._aligned_sample_now()
            if start_idx is None:
                self._reset_alignment_test()
                return
            self._align_test_say_start_float = start_float
            self._align_test_say_start_idx = start_idx
            self._align_test_method = method
            self._align_test_phase = "blink"
            self._align_test_deadline_ns = time.perf_counter_ns() + int(COLLECTION_SAY_S * 1_000_000_000)
            return

        _event_host_ns, end_float, end_idx, method = self._aligned_sample_now()
        if end_idx is not None:
            self._align_test_say_end_float = end_float
            self._align_test_say_end_idx = end_idx
            if method:
                self._align_test_method = method
        self._align_test_deadline_ns = None
        self._align_test_phase = "result"
        self._align_test_result = self._build_alignment_test_result()

    def start_alignment_test(self) -> None:
        if self._recorder.enabled:
            raise RuntimeError("Stop session recording before running alignment test")
        if self._trial_state != TrialState.IDLE:
            raise RuntimeError("Finish the active trial before running alignment test")
        if self._collect_phase not in ("disabled", "pick_word"):
            raise RuntimeError("Finish collection before running alignment test")
        if self._alignment_test_busy():
            raise RuntimeError("Alignment test already running")
        if self._model_test_busy():
            raise RuntimeError("Finish model test before running alignment test")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet; wait for EEG stream")

        self._align_test_result = None
        self._align_test_say_start_idx = None
        self._align_test_say_start_float = None
        self._align_test_say_end_idx = None
        self._align_test_say_end_float = None
        self._align_test_method = method
        self._align_test_phase = "countdown"
        self._align_test_deadline_ns = time.perf_counter_ns() + int(COLLECTION_BEFORE_S * 1_000_000_000)

    def clear_alignment_test(self) -> None:
        self._reset_alignment_test()

    def alignment_test_result(self) -> dict[str, Any]:
        if self._align_test_phase != "result" or self._align_test_result is None:
            raise RuntimeError("No alignment test result available")
        return self._align_test_result

    def _model_test_busy(self) -> bool:
        return self._model_test_phase in ("countdown", "say")

    def _reset_model_test(self) -> None:
        self._model_test_phase = "idle"
        self._model_test_deadline_ns = None
        self._model_test_trial = 0
        self._model_test_captures = []
        self._model_test_result = None
        self._model_test_say_start_idx = None
        self._model_test_say_start_float = None

    def _model_test_phase_remaining_s(self) -> Optional[float]:
        if self._model_test_phase not in ("countdown", "say"):
            return None
        if self._model_test_deadline_ns is None:
            return None
        return max(0.0, (self._model_test_deadline_ns - time.perf_counter_ns()) / 1e9)

    def _model_test_status(self) -> dict[str, Any]:
        return {
            "phase": self._model_test_phase,
            "trial": self._model_test_trial if self._model_test_busy() else None,
            "trials_total": self._model_test_trials_total,
            "phase_remaining_s": self._model_test_phase_remaining_s(),
            "before_s": COLLECTION_BEFORE_S,
            "say_s": COLLECTION_SAY_S,
            "has_result": self._model_test_result is not None,
            "checkpoint": self._model_test_checkpoint,
        }

    def _extract_training_aligned_window(self, start_idx: int, end_idx: int) -> Optional[np.ndarray]:
        """Filter the full ring buffer, then crop the event span — same order as data.build_event_windows."""
        if self._model_test_model_config is None:
            return None
        target_len = int(self._model_test_model_config["T"])
        with self._lock:
            latest = self._latest_sample_index
            filled = min(self._total_frames, self._buffer_samples)
            if filled <= 0 or latest < 0:
                return None
            oldest_idx = latest - filled + 1
            if end_idx < oldest_idx or start_idx > latest:
                return None
            view = _ordered_buffer_view(self._buffer_uv, self._write_idx)

        block_tc = np.asarray(view.T, dtype=np.float32)
        filtered_tc = preprocess_session_channels(
            block_tc,
            float(SAMPLE_RATE_HZ),
            line_noise=DEFAULT_LINE_NOISE_CONFIG,
            eeg=DEFAULT_EEG_BANDPASS_CONFIG,
            emg=DEFAULT_EMG_BANDPASS_CONFIG,
        )
        i0 = max(0, start_idx - oldest_idx)
        i1 = min(filled, end_idx - oldest_idx + 1)
        if i1 <= i0:
            return None
        return fix_window_length(filtered_tc[i0:i1, :], target_len)

    def _load_model_for_test(self) -> None:
        if self._model_test_nn is not None:
            return
        if torch is None:
            raise RuntimeError("PyTorch is not installed")

        path = default_checkpoint_path("best")
        if not path.exists():
            raise RuntimeError(f"No checkpoint found at {path}")
        device = get_device()
        model, label_to_idx, meta = load_checkpoint(path, device)
        self._model_test_nn = model
        self._model_test_label_to_idx = label_to_idx
        self._model_test_idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        self._model_test_checkpoint = str(path.resolve())
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._model_test_model_config = dict(ckpt["model_config"])
        self._model_test_device = device

    def _capture_model_test_window(self) -> dict[str, Any]:
        trial = self._model_test_trial
        if self._model_test_say_start_idx is None or self._model_test_say_start_float is None:
            return {"trial": trial, "error": "missing say window start"}

        say_start_i = int(self._model_test_say_start_idx)
        say_start_f = float(self._model_test_say_start_float)
        say_end_f = say_start_f + COLLECTION_SAY_S * SAMPLE_RATE_HZ
        say_end_i = TimeAligner.quantize_sample_index(say_end_f) or int(round(say_end_f))
        window_tc = self._extract_training_aligned_window(say_start_i, say_end_i)
        if window_tc is None:
            return {
                "trial": trial,
                "error": "labeled window is no longer in the ring buffer",
                "sample_index_start": say_start_i,
                "sample_index_end": say_end_i,
            }

        return {
            "trial": trial,
            "sample_index_start": say_start_i,
            "sample_index_end": say_end_i,
            "window_tc": window_tc,
            "window_len": int(window_tc.shape[0]),
        }

    def _predict_model_test_batch(self, captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._model_test_nn is None:
            raise RuntimeError("Model not loaded")
        idx_to_label = self._model_test_idx_to_label or {}

        trial_rows: list[dict[str, Any]] = []
        windows_tc: list[np.ndarray] = []
        batch_row_indices: list[int] = []

        for capture in captures:
            row: dict[str, Any] = {
                "trial": capture.get("trial"),
                "sample_index_start": capture.get("sample_index_start"),
                "sample_index_end": capture.get("sample_index_end"),
                "window_len": capture.get("window_len"),
            }
            if capture.get("error"):
                row["error"] = capture["error"]
                trial_rows.append(row)
                continue
            window_tc = capture.get("window_tc")
            if window_tc is None:
                row["error"] = "missing window"
                trial_rows.append(row)
                continue
            batch_row_indices.append(len(trial_rows))
            trial_rows.append(row)
            windows_tc.append(np.asarray(window_tc, dtype=np.float32))

        if windows_tc:
            preds = predict_fusion_batch(
                self._model_test_nn,
                windows_tc,
                device=self._model_test_device,
                idx_to_label=idx_to_label,
            )
            for batch_idx, row_idx in enumerate(batch_row_indices):
                pred = preds[batch_idx]
                trial_rows[row_idx].update(pred)
                labeled_probs = {
                    idx_to_label[i]: round(float(pred["probs"][i]), 4)
                    for i in range(len(pred["probs"]))
                }
                print(
                    f"[model-test] trial {trial_rows[row_idx]['trial']}: "
                    f"logits={[round(v, 4) for v in pred['logits']]} "
                    f"probs={labeled_probs} "
                    f"pred={pred['predicted_label']} ({pred['predicted_probability']:.4f})",
                    flush=True,
                )

        return trial_rows

    def _finalize_model_test(self) -> None:
        try:
            trial_results = self._predict_model_test_batch(self._model_test_captures)
        except Exception as exc:  # pylint: disable=broad-except
            trial_results = [
                {
                    "trial": capture.get("trial"),
                    "sample_index_start": capture.get("sample_index_start"),
                    "sample_index_end": capture.get("sample_index_end"),
                    "error": str(exc),
                }
                for capture in self._model_test_captures
            ]

        label_order = (
            sorted(self._model_test_label_to_idx.keys())
            if self._model_test_label_to_idx
            else []
        )
        self._model_test_result = {
            "checkpoint": self._model_test_checkpoint,
            "trials_total": self._model_test_trials_total,
            "label_order": label_order,
            "trials": trial_results,
        }
        self._model_test_captures = []
        self._model_test_phase = "result"

    def _tick_model_test(self) -> None:
        if self._model_test_phase not in ("countdown", "say"):
            return
        deadline = self._model_test_deadline_ns
        if deadline is None or time.perf_counter_ns() < deadline:
            return

        if self._model_test_phase == "countdown":
            _event_host_ns, start_float, start_idx, _method = self._aligned_sample_now()
            if start_idx is None:
                self._reset_model_test()
                return
            self._model_test_say_start_float = start_float
            self._model_test_say_start_idx = start_idx
            self._model_test_phase = "say"
            self._model_test_deadline_ns = time.perf_counter_ns() + int(COLLECTION_SAY_S * 1_000_000_000)
            return

        capture = self._capture_model_test_window()
        self._model_test_captures.append(capture)
        self._model_test_say_start_idx = None
        self._model_test_say_start_float = None

        if self._model_test_trial < self._model_test_trials_total:
            self._model_test_trial += 1
            self._model_test_phase = "countdown"
            self._model_test_deadline_ns = time.perf_counter_ns() + int(COLLECTION_BEFORE_S * 1_000_000_000)
            return

        self._model_test_deadline_ns = None
        self._finalize_model_test()

    def start_model_test(self) -> None:
        if self._recorder.enabled:
            raise RuntimeError("Stop session recording before running model test")
        if self._trial_state != TrialState.IDLE:
            raise RuntimeError("Finish the active trial before running model test")
        if self._collect_phase not in ("disabled", "pick_word"):
            raise RuntimeError("Finish collection before running model test")
        if self._alignment_test_busy():
            raise RuntimeError("Finish alignment test before running model test")
        if self._model_test_busy():
            raise RuntimeError("Model test already running")
        _event_host_ns, aligned_float, aligned_idx, _method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet; wait for EEG stream")

        self._load_model_for_test()
        self._reset_model_test()
        self._model_test_trials_total = MODEL_TEST_TRIALS
        self._model_test_trial = 1
        self._model_test_phase = "countdown"
        self._model_test_deadline_ns = time.perf_counter_ns() + int(COLLECTION_BEFORE_S * 1_000_000_000)

    def clear_model_test(self) -> None:
        self._reset_model_test()

    def model_test_result(self) -> dict[str, Any]:
        if self._model_test_phase != "result" or self._model_test_result is None:
            raise RuntimeError("No model test result available")
        return self._model_test_result

    def status(self) -> dict[str, Any]:
        with self._lock:
            latest_sample = self._latest_sample_index
            total_frames = self._total_frames
        rail = self._rail_warning()
        return {
            "test_mode": self._test_mode,
            "serial_error": self._reader.last_error,
            "trial_state": self._trial_state.value,
            "trial_word": self._current_trial_word or None,
            "thinking_countdown": self._thinking_countdown_remaining(),
            "recording_enabled": self._recorder.enabled,
            "session_dir": str(self._recorder.session_dir) if self._recorder.session_dir else None,
            "last_session_dir": str(self._last_session_dir) if self._last_session_dir else None,
            "pending_asr_jobs": self._pending_asr_jobs,
            "latest_sample_index": latest_sample,
            "total_frames": total_frames,
            "rail_warning": rail,
            "channel_count": CHANNEL_COUNT,
            "channels_per_page": CHANNELS_PER_PAGE,
            "window_seconds": WINDOW_SECONDS,
            "update_hz": UPDATE_HZ,
            "default_fixed_scale_uv": DEFAULT_FIXED_SCALE_UV,
            "collect": self._collect_status(),
            "alignment_test": self._alignment_test_status(),
            "model_test": self._model_test_status(),
        }

    def waveform(
        self,
        *,
        mode: str = "paged",
        page_start: int = 0,
        single_channel: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            raw_view = _ordered_buffer_view(self._buffer_uv, self._write_idx)

        raw_display = raw_view
        if raw_display.shape[1] > self._display_samples:
            raw_display = raw_view[:, -self._display_samples :]

        # Filter the full look-back buffer, then keep only the trailing display window so
        # edge transients stay in the discarded prefix (filtered waveform + band power only).
        filtered_view = _preprocess_monitor_window(raw_view, SAMPLE_RATE_HZ)
        if filtered_view.shape[1] > self._display_samples:
            filtered_view = filtered_view[:, -self._display_samples :]

        step = max(1, raw_display.shape[1] // PLOT_POINTS)
        raw_ds = raw_display[:, ::step]
        filtered_ds = filtered_view[:, ::step]
        time_s = np.linspace(-WINDOW_SECONDS, 0, raw_ds.shape[1], dtype=np.float32)

        if mode == "single":
            indices = [single_channel % CHANNEL_COUNT]
        elif mode == "all":
            indices = list(range(CHANNEL_COUNT))
        else:
            indices = [page_start + i for i in range(CHANNELS_PER_PAGE) if page_start + i < CHANNEL_COUNT]

        traces: list[dict[str, Any]] = []
        debug_lines: list[str] = []
        for channel_idx in indices:
            channel_signal = filtered_view[channel_idx]
            band_time_s, band_series = _band_power_series(
                channel_signal,
                SAMPLE_RATE_HZ,
                bands_for_channel(channel_idx),
            )
            if self._debug_bands and channel_idx in (0, 9, 15, 16):
                debug_lines.append(_format_band_debug(channel_idx, channel_signal, band_series))
            band_time_ds, band_series_ds = _downsample_band_series(band_time_s, band_series)
            traces.append(
                {
                    "index": channel_idx,
                    "name": channel_name(channel_idx),
                    "y": raw_ds[channel_idx].astype(np.float32).tolist(),
                    "y_filtered": filtered_ds[channel_idx].astype(np.float32).tolist(),
                    "bands": band_series_ds,
                    "band_time_s": band_time_ds,
                }
            )
        if self._debug_bands and debug_lines:
            now_ns = time.perf_counter_ns()
            if now_ns - self._band_debug_last_print_ns >= 1_000_000_000:
                self._band_debug_last_print_ns = now_ns
                print("\n".join(debug_lines), flush=True)
        return {
            "time_s": time_s.astype(np.float32).tolist(),
            "traces": traces,
            "eeg_bands": list(EEG_BANDS),
            "emg_bands": list(EMG_BANDS),
        }

    def dashboard(
        self,
        *,
        mode: str = "paged",
        page_start: int = 0,
        single_channel: int = 0,
    ) -> dict[str, Any]:
        return {
            "status": self.status(),
            "waveform": self.waveform(
                mode=mode,
                page_start=page_start,
                single_channel=single_channel,
            ),
        }

    def start_recording(self, base_dir: Path) -> Path:
        if self._alignment_test_busy():
            raise RuntimeError("Wait for alignment test to finish before recording")
        if self._model_test_busy():
            raise RuntimeError("Wait for model test to finish before recording")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        session_dir = self._recorder.start(base_dir)
        self._last_session_dir = session_dir
        if aligned_idx is not None:
            self._recorder.log_event(
                event_type="session_recording_started",
                sample_index_start=aligned_idx,
                sample_index_start_float=aligned_float,
                alignment_method=method,
                payload={"source": "flask"},
            )
        self._collect_phase = "pick_word"
        return session_dir

    def stop_recording(self) -> Optional[Path]:
        if self._trial_state != TrialState.IDLE:
            raise RuntimeError("End active trial before stopping session recording")
        if self._collect_phase not in ("disabled", "pick_word"):
            raise RuntimeError("Finish the current collection before stopping recording")
        if self._pending_asr_jobs > 0:
            raise RuntimeError("Wait for speech transcription to finish before stopping")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is not None:
            self._recorder.log_event(
                event_type="session_recording_stopped",
                sample_index_start=aligned_idx,
                sample_index_start_float=aligned_float,
                alignment_method=method,
                payload={"source": "flask"},
            )
        session_dir = self._recorder.session_dir
        self._recorder.stop()
        self._reset_collect()
        if session_dir:
            self._last_session_dir = session_dir
        return session_dir

    def log_marker(self, label: str, source: str = "api") -> int:
        if not self._recorder.enabled:
            raise RuntimeError("Recording disabled")
        event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet")
        self._recorder.log_event(
            event_type="manual_marker",
            label_text=label,
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={"source": source, "event_host_perf_counter_ns": event_host_ns},
        )
        return aligned_idx

    def start_thinking_trial(self, word: str) -> None:
        if not self._recorder.enabled:
            raise RuntimeError("Start session recording first")
        if self._trial_state != TrialState.IDLE:
            raise RuntimeError("Another trial is already active")
        word = word.strip()
        if not word:
            raise RuntimeError("word is required")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet")
        trial_id = uuid.uuid4().hex
        self._recorder.log_event(
            event_type="thinking_trial_armed",
            label_text=word,
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={"trial_id": trial_id, "countdown_seconds": THINKING_COUNTDOWN_SECONDS},
        )
        self._trial_state = TrialState.THINKING_COUNTDOWN
        self._current_trial_id = trial_id
        self._current_trial_word = word
        self._thinking_countdown_deadline_ns = (
            time.perf_counter_ns() + (THINKING_COUNTDOWN_SECONDS * 1_000_000_000)
        )

    def end_thinking_trial(self) -> None:
        if self._trial_state != TrialState.THINKING_ACTIVE or not self._current_trial_id:
            raise RuntimeError("No active thinking trial")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet")
        self._recorder.log_event(
            event_type="thinking_trial_end",
            label_text=self._current_trial_word,
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={"trial_id": self._current_trial_id},
        )
        self._trial_state = TrialState.IDLE
        self._current_trial_id = None
        self._current_trial_word = ""

    def start_speech_block(self) -> str:
        if not self._recorder.enabled:
            raise RuntimeError("Start session recording first")
        if self._trial_state != TrialState.IDLE:
            raise RuntimeError("Another trial is already active")
        if not self._speech_recorder.available:
            raise RuntimeError("Install sounddevice to record speech audio")
        session_dir = self._recorder.session_dir
        if session_dir is None:
            raise RuntimeError("Session folder unavailable")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet")
        block_id = uuid.uuid4().hex
        wav_path = session_dir / f"speech_block_{block_id}.wav"
        self._speech_recorder.start(wav_path)
        self._trial_state = TrialState.SPEECH_ACTIVE
        self._speech_block_id = block_id
        self._speech_audio_path = wav_path
        self._speech_block_start_sample_float = aligned_float
        self._recorder.log_event(
            event_type="speech_block_start",
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={"speech_block_id": block_id, "audio_file": wav_path.name},
        )
        return block_id

    def stop_speech_block(self) -> str:
        if self._trial_state != TrialState.SPEECH_ACTIVE or not self._speech_block_id:
            raise RuntimeError("No active speech block")
        _event_host_ns, aligned_float, aligned_idx, method = self._aligned_sample_now()
        if aligned_idx is None:
            raise RuntimeError("No samples yet")
        block_id = self._speech_block_id
        wav_path = self._speech_audio_path
        start_float = self._speech_block_start_sample_float
        self._speech_recorder.stop()
        self._recorder.log_event(
            event_type="speech_block_end",
            sample_index_start=aligned_idx,
            sample_index_start_float=aligned_float,
            alignment_method=method,
            payload={"speech_block_id": block_id, "audio_file": wav_path.name if wav_path else ""},
        )
        if wav_path is not None and start_float is not None:
            self._pending_asr_jobs += 1
            worker = threading.Thread(
                target=self._run_asr_in_background,
                args=(wav_path, block_id, start_float),
                daemon=True,
            )
            worker.start()
        self._trial_state = TrialState.IDLE
        self._speech_block_id = None
        self._speech_audio_path = None
        self._speech_block_start_sample_float = None
        return block_id

    def _run_asr_in_background(self, wav_path: Path, block_id: str, block_start_sample_float: float) -> None:
        try:
            if whisper is None:
                self._asr_results_queue.put(
                    {"type": "asr_error", "speech_block_id": block_id, "error": "whisper not installed"}
                )
                return
            if self._whisper_model is None:
                self._whisper_model = whisper.load_model("base")
            result = self._whisper_model.transcribe(str(wav_path), word_timestamps=True, fp16=False)
            words: list[dict[str, Any]] = []
            for segment in result.get("segments", []):
                for word in segment.get("words", []) or []:
                    words.append(
                        {
                            "text": str(word.get("word", "")).strip(),
                            "start_s": float(word.get("start", 0.0)),
                            "end_s": float(word.get("end", 0.0)),
                            "confidence": float(word.get("probability", 0.0)),
                        }
                    )
            self._asr_results_queue.put(
                {
                    "type": "asr_words",
                    "speech_block_id": block_id,
                    "base_sample_float": block_start_sample_float,
                    "word_count": len(words),
                    "words": words,
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._asr_results_queue.put(
                {"type": "asr_error", "speech_block_id": block_id, "error": str(exc)}
            )

    def _drain_asr_results(self) -> None:
        while True:
            try:
                result = self._asr_results_queue.get_nowait()
            except queue.Empty:
                break
            if self._pending_asr_jobs > 0:
                self._pending_asr_jobs -= 1
            if result.get("type") == "asr_error":
                self._recorder.log_event(
                    event_type="speech_asr_error",
                    payload={
                        "speech_block_id": result.get("speech_block_id", ""),
                        "error": result.get("error", "unknown"),
                    },
                )
                continue
            if result.get("type") != "asr_words":
                continue
            base_sample_float = float(result.get("base_sample_float", 0.0))
            block_id = str(result.get("speech_block_id", ""))
            for word_info in result.get("words", []):
                text = str(word_info.get("text", "")).strip()
                if not text:
                    continue
                start_s = float(word_info.get("start_s", 0.0))
                end_s = float(word_info.get("end_s", start_s))
                confidence = float(word_info.get("confidence", 0.0))
                sample_start_float = base_sample_float + (start_s * SAMPLE_RATE_HZ)
                sample_end_float = base_sample_float + (end_s * SAMPLE_RATE_HZ)
                sample_start = TimeAligner.quantize_sample_index(sample_start_float)
                sample_end = TimeAligner.quantize_sample_index(sample_end_float)
                self._recorder.log_event(
                    event_type="speech_word",
                    label_text=text,
                    sample_index_start=sample_start,
                    sample_index_start_float=sample_start_float,
                    sample_index_end=sample_end,
                    sample_index_end_float=sample_end_float,
                    confidence=confidence,
                    alignment_method="speech_offset_from_block_start",
                    payload={
                        "speech_block_id": block_id,
                        "word_start_s": start_s,
                        "word_end_s": end_s,
                    },
                )

def create_app(service: AcquisitionService) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/dashboard")
    def dashboard() -> Any:
        mode = request.args.get("mode", "paged")
        page_start = int(request.args.get("page_start", 0))
        single_channel = int(request.args.get("single_channel", 0))
        return jsonify(
            service.dashboard(
                mode=mode,
                page_start=page_start,
                single_channel=single_channel,
            )
        )

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True})

    @app.get("/status")
    def status() -> Any:
        return jsonify(service.status())

    @app.post("/recording/start")
    def start_recording() -> Any:
        payload = request.get_json(silent=True) or {}
        base_dir = Path(payload.get("base_dir", Path.cwd() / DEFAULT_RECORDINGS_DIR))
        session_dir = service.start_recording(base_dir)
        return jsonify({"session_dir": str(session_dir)})

    @app.post("/recording/stop")
    def stop_recording() -> Any:
        session_dir = service.stop_recording()
        return jsonify({"session_dir": str(session_dir) if session_dir else None})

    @app.post("/collect/word")
    def collect_word() -> Any:
        payload = request.get_json(silent=True) or {}
        word = str(payload.get("word", "")).strip()
        service.start_collect_word(word)
        return jsonify(service.status()["collect"])

    @app.post("/collect/scramble")
    def collect_scramble() -> Any:
        payload = request.get_json(silent=True) or {}
        set_count = int(payload.get("set", DEFAULT_SCRAMBLE_SET))
        rep_count = int(payload.get("rep", DEFAULT_SCRAMBLE_REP))
        service.start_collect_scramble(set_count=set_count, rep_count=rep_count)
        return jsonify(service.status()["collect"])

    @app.post("/collect/negative-labels/toggle")
    def collect_negative_labels_toggle() -> Any:
        service.toggle_collect_negative_labels()
        return jsonify(service.status()["collect"])

    @app.post("/alignment-test/start")
    def start_alignment_test() -> Any:
        service.start_alignment_test()
        return jsonify(service.status()["alignment_test"])

    @app.post("/alignment-test/clear")
    def clear_alignment_test() -> Any:
        service.clear_alignment_test()
        return jsonify(service.status()["alignment_test"])

    @app.get("/alignment-test/result")
    def alignment_test_result() -> Any:
        return jsonify(service.alignment_test_result())

    @app.post("/model-test/start")
    def start_model_test() -> Any:
        service.start_model_test()
        return jsonify(service.status()["model_test"])

    @app.post("/model-test/clear")
    def clear_model_test() -> Any:
        service.clear_model_test()
        return jsonify(service.status()["model_test"])

    @app.get("/model-test/result")
    def model_test_result() -> Any:
        return jsonify(service.model_test_result())

    @app.post("/events/marker")
    def marker() -> Any:
        payload = request.get_json(silent=True) or {}
        label = str(payload.get("label", "")).strip()
        if not label:
            return jsonify({"error": "label is required"}), 400
        source = str(payload.get("source", "api"))
        sample_index = service.log_marker(label=label, source=source)
        return jsonify({"sample_index": sample_index, "label": label})

    @app.post("/trials/thinking/start")
    def start_thinking() -> Any:
        payload = request.get_json(silent=True) or {}
        word = str(payload.get("word", "")).strip()
        service.start_thinking_trial(word)
        return jsonify({"state": "thinking_countdown"})

    @app.post("/trials/thinking/end")
    def end_thinking() -> Any:
        service.end_thinking_trial()
        return jsonify({"state": "idle"})

    @app.post("/trials/speech/start")
    def start_speech() -> Any:
        block_id = service.start_speech_block()
        return jsonify({"speech_block_id": block_id})

    @app.post("/trials/speech/stop")
    def stop_speech() -> Any:
        block_id = service.stop_speech_block()
        return jsonify({"speech_block_id": block_id, "state": "idle"})

    @app.errorhandler(RuntimeError)
    def handle_runtime_error(exc: RuntimeError) -> Any:
        return jsonify({"error": str(exc)}), 400

    return app


def pick_default_port() -> Optional[str]:
    ports = list(list_ports.comports())
    if not ports:
        return None
    usb_first = sorted(ports, key=lambda p: ("usb" not in p.description.lower(), p.device))
    return usb_first[0].device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Woodside Flask host service")
    parser.add_argument("--test", action="store_true", help="Stream synthetic sine waves instead of USB serial")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. /dev/tty.usbmodemXXXX)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask bind host")
    parser.add_argument(
        "--http-port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help="Flask bind port (avoid 5000 on macOS — often used by AirPlay)",
    )
    parser.add_argument(
        "--debug-bands",
        action="store_true",
        help="Print EEG/EMG band power diagnostics to the console once per second",
    )
    return parser.parse_args()


def _install_shutdown_handlers(service: AcquisitionService) -> None:
    def _on_signal(signum: int, _frame: Any) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = str(signum)
        service.shutdown(recording_shutdown_reason=sig_name)
        raise SystemExit(128 + signum if signum != signal.SIGINT else 130)

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, _on_signal)


def main() -> int:
    args = parse_args()
    if args.test:
        service = AcquisitionService(test_mode=True, debug_bands=args.debug_bands)
    else:
        serial_port = args.port or pick_default_port() or "/dev/tty.usbmodem1101"
        if not serial_port:
            print("No serial ports found. Pass --port explicitly or use --test.")
            return 1
        service = AcquisitionService(
            serial_port=serial_port,
            baudrate=args.baud,
            debug_bands=args.debug_bands,
        )
    app = create_app(service)
    _install_shutdown_handlers(service)
    url = f"http://127.0.0.1:{args.http_port}/"
    print(f"Woodside monitor GUI: {url}")
    if args.test:
        print("Test mode: streaming synthetic sine waves (no USB).")
    if args.debug_bands:
        print("Band debug: printing power values for EEG1, EEG10, EEG16, EMG1 once per second.")
    print("(On macOS, port 5000 is often AirPlay and returns HTTP 403 — use the URL above.)")
    try:
        app.run(host=args.host, port=args.http_port, threaded=True, use_reloader=False)
    finally:
        service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
