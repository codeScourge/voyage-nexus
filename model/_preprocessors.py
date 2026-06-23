"""Lightweight EEG/EMG band-pass filters for inspection and windowing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos

EEG_CHANNELS = slice(0, 16)
EMG_CHANNELS = slice(16, 32)

# Typical inspection bands at 1 kHz (high must stay below Nyquist).
DEFAULT_EEG_BANDPASS = (1.0, 40.0)   # highpass_hz, lowpass_hz
DEFAULT_EMG_BANDPASS = (20.0, 450.0)


@dataclass(frozen=True, slots=True)
class BandpassConfig:
    highpass_hz: float
    lowpass_hz: float

    def __post_init__(self) -> None:
        if self.highpass_hz <= 0 or self.lowpass_hz <= 0:
            raise ValueError("bandpass cutoffs must be positive")
        if self.highpass_hz >= self.lowpass_hz:
            raise ValueError(
                f"highpass_hz ({self.highpass_hz}) must be below lowpass_hz ({self.lowpass_hz})"
            )


DEFAULT_EEG_BANDPASS_CONFIG = BandpassConfig(*DEFAULT_EEG_BANDPASS)
DEFAULT_EMG_BANDPASS_CONFIG = BandpassConfig(*DEFAULT_EMG_BANDPASS)


@dataclass(frozen=True, slots=True)
class LineNoiseConfig:
    """Notch mains frequencies (50 / 60 Hz) and optional harmonics."""

    frequencies_hz: tuple[float, ...] = (50.0, 60.0)
    q: float = 30.0
    harmonics: bool = False
    max_frequency_hz: float = 480.0

    def __post_init__(self) -> None:
        if self.q <= 0:
            raise ValueError(f"q must be positive, got {self.q}")
        if not self.frequencies_hz:
            raise ValueError("frequencies_hz must not be empty")
        if any(f <= 0 for f in self.frequencies_hz):
            raise ValueError("line-noise frequencies must be positive")


DEFAULT_LINE_NOISE_CONFIG = LineNoiseConfig()


def _notch_frequencies(
    sample_rate_hz: float,
    config: LineNoiseConfig,
) -> tuple[float, ...]:
    nyquist = sample_rate_hz / 2.0
    out: list[float] = []
    for base in config.frequencies_hz:
        _validate_cutoff(sample_rate_hz, base, label="line_noise_hz")
        out.append(base)
        if not config.harmonics:
            continue
        harmonic = 2.0 * base
        while harmonic < nyquist and harmonic <= config.max_frequency_hz:
            out.append(harmonic)
            harmonic += base
    return tuple(sorted(set(out)))


def notch_filter(
    x: np.ndarray,
    sample_rate_hz: float,
    notch_hz: float,
    *,
    q: float = 30.0,
    axis: int = -1,
    zero_phase: bool = True,
) -> np.ndarray:
    """IIR notch at ``notch_hz``.

    Uses zero-phase forward-backward filtering by default (offline batch data).
    Set ``zero_phase=False`` for causal single-pass filtering (live monitor windows).
    """
    _validate_cutoff(sample_rate_hz, notch_hz, label="notch_hz")
    b, a = iirnotch(notch_hz, q, sample_rate_hz)
    sos = tf2sos(b, a)
    x32 = np.asarray(x, dtype=np.float32)
    if zero_phase:
        return sosfiltfilt(sos, x32, axis=axis).astype(np.float32)
    return sosfilt(sos, x32, axis=axis).astype(np.float32)


def apply_line_noise_notch(
    channels: np.ndarray,
    sample_rate_hz: float,
    config: LineNoiseConfig = DEFAULT_LINE_NOISE_CONFIG,
    *,
    zero_phase: bool = True,
) -> np.ndarray:
    """Cascade notches for 50/60 Hz (and optional harmonics) on all channels."""
    out = np.asarray(channels, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Expected (time, channels) array, got shape {out.shape}")
    for notch_hz in _notch_frequencies(sample_rate_hz, config):
        out = notch_filter(
            out,
            sample_rate_hz,
            notch_hz,
            q=config.q,
            axis=0,
            zero_phase=zero_phase,
        )
    return out


def preprocess_session_channels(
    channels: np.ndarray,
    sample_rate_hz: float,
    *,
    line_noise: LineNoiseConfig | None = DEFAULT_LINE_NOISE_CONFIG,
    eeg: BandpassConfig | None = DEFAULT_EEG_BANDPASS_CONFIG,
    emg: BandpassConfig | None = DEFAULT_EMG_BANDPASS_CONFIG,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Line-noise notches first, then per-modality band-pass."""
    out = np.asarray(channels, dtype=np.float32)
    if line_noise is not None:
        out = apply_line_noise_notch(out, sample_rate_hz, line_noise, zero_phase=zero_phase)
    if eeg is not None or emg is not None:
        out = apply_session_bandpass(out, sample_rate_hz, eeg=eeg, emg=emg, order=order)
    return out


def _validate_cutoff(sample_rate_hz: float, cutoff_hz: float, *, label: str) -> None:
    if cutoff_hz <= 0:
        raise ValueError(f"{label} must be positive, got {cutoff_hz}")
    nyquist = sample_rate_hz / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"{label} ({cutoff_hz}) must be below Nyquist ({nyquist:.3g} Hz)"
        )


def _validate_bandpass_for_rate(sample_rate_hz: float, config: BandpassConfig) -> None:
    _validate_cutoff(sample_rate_hz, config.highpass_hz, label="highpass_hz")
    _validate_cutoff(sample_rate_hz, config.lowpass_hz, label="lowpass_hz")


def bandpass_butter(
    x: np.ndarray,
    sample_rate_hz: float,
    config: BandpassConfig,
    *,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass via forward-backward SOS filtering."""
    _validate_bandpass_for_rate(sample_rate_hz, config)
    sos = butter(
        order,
        [config.highpass_hz, config.lowpass_hz],
        btype="band",
        fs=sample_rate_hz,
        output="sos",
    )
    x32 = np.asarray(x, dtype=np.float32)
    return sosfiltfilt(sos, x32, axis=axis).astype(np.float32)


def apply_channel_bandpass(
    channels: np.ndarray,
    sample_rate_hz: float,
    config: BandpassConfig,
    *,
    channel_slice: slice,
    order: int = 4,
) -> np.ndarray:
    """Band-pass a channel subset; other columns are copied unchanged."""
    out = np.asarray(channels, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Expected (time, channels) array, got shape {out.shape}")
    if out.shape[1] <= channel_slice.start:
        return out

    block = out[:, channel_slice]
    filtered = bandpass_butter(block, sample_rate_hz, config, order=order, axis=0)
    result = out.copy()
    result[:, channel_slice] = filtered
    return result


def apply_session_bandpass(
    channels: np.ndarray,
    sample_rate_hz: float,
    *,
    eeg: BandpassConfig | None = DEFAULT_EEG_BANDPASS_CONFIG,
    emg: BandpassConfig | None = DEFAULT_EMG_BANDPASS_CONFIG,
    order: int = 4,
) -> np.ndarray:
    """Apply EEG and EMG band-pass filters independently."""
    out = np.asarray(channels, dtype=np.float32)
    if eeg is not None:
        out = apply_channel_bandpass(out, sample_rate_hz, eeg, channel_slice=EEG_CHANNELS, order=order)
    if emg is not None:
        out = apply_channel_bandpass(out, sample_rate_hz, emg, channel_slice=EMG_CHANNELS, order=order)
    return out
