from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

import numpy as np

MAGIC = 0x574F4F44
VERSION = 1
CHANNEL_COUNT = 32
FRAME_FORMAT = "<IHHQQ32ii"
FRAME_SIZE = struct.calcsize(FRAME_FORMAT)
CHECKSUM_SEED = 0xA5A55A5A

EEG_GAIN = 6.0
EMG_GAIN = 1.0
ADS_VREF_VOLTS = 4.5
SWAP_EEG_DAISY_HALVES = True


def xor_checksum(payload: bytes) -> int:
    checksum = CHECKSUM_SEED
    for i, byte in enumerate(payload):
        checksum ^= byte << ((i % 4) * 8)
    return checksum & 0xFFFFFFFF


def lsb_volts(gain: float) -> float:
    return (2.0 * ADS_VREF_VOLTS / gain) / ((2**24) - 1)


EEG_LSB_UV = lsb_volts(EEG_GAIN) * 1e6
EMG_LSB_UV = lsb_volts(EMG_GAIN) * 1e6

# Session file: sample_index, mcu_time_us, 32 channels in microvolts (float32).
EEG_RECORD_FORMAT = "<QQ32f"
EEG_RECORD_FORMAT_CODES_LEGACY = "<QQ32i"
CHANNEL_UNITS = "uV"


def ads_calibration_metadata() -> dict[str, float | str]:
    """Calibration used by codes_to_uv(); stored in session_meta.json."""
    return {
        "units": CHANNEL_UNITS,
        "ads_vref_volts": ADS_VREF_VOLTS,
        "eeg_gain": EEG_GAIN,
        "emg_gain": EMG_GAIN,
        "eeg_lsb_uv": EEG_LSB_UV,
        "emg_lsb_uv": EMG_LSB_UV,
        "formula": "uv = code * (2 * Vref / gain) / (2**24 - 1) * 1e6",
        "eeg_channels": "0..15",
        "emg_channels": "16..31",
    }


def codes_to_uv(channels_i32: np.ndarray) -> np.ndarray:
    """Convert ADS1299 signed codes to microvolts (single conversion path)."""
    out = np.asarray(channels_i32, dtype=np.float64)   # was float32
    if out.shape != (CHANNEL_COUNT,):
        raise ValueError(f"Expected {CHANNEL_COUNT} channels, got shape {out.shape}")
    out[:16] *= EEG_LSB_UV          # plain Python floats are float64
    out[16:] *= EMG_LSB_UV
    return out


def apply_channel_remap(channels: np.ndarray) -> np.ndarray:
    """Apply host-side channel ordering fixes.

    In current hardware wiring, the two 8-channel EEG daisy blocks arrive reversed
    relative to board labeling (9..16 then 1..8). Swap them back so host uses
    board order EEG1..EEG16 consistently.
    """
    if not SWAP_EEG_DAISY_HALVES:
        return channels
    out = channels.copy()
    out[:8] = channels[8:16]
    out[8:16] = channels[:8]
    return out


@dataclass(slots=True)
class SampleFrame:
    mcu_time_us: int
    sample_index: int
    channels_i32: np.ndarray  # shape=(32,), int32

    def channels_uv(self) -> np.ndarray:
        return codes_to_uv(self.channels_i32)


class FrameParser:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self.desync_count = 0
        self.bad_checksum_count = 0

    def feed(self, data: bytes) -> List[SampleFrame]:
        self._buffer.extend(data)
        frames: List[SampleFrame] = []

        while True:
            if len(self._buffer) < FRAME_SIZE:
                break

            magic_idx = self._buffer.find(struct.pack("<I", MAGIC))
            if magic_idx < 0:
                self.desync_count += 1
                # Keep tail in case the next chunk starts with partial magic.
                del self._buffer[:-3]
                break

            if magic_idx > 0:
                self.desync_count += 1
                del self._buffer[:magic_idx]

            if len(self._buffer) < FRAME_SIZE:
                break

            raw = bytes(self._buffer[:FRAME_SIZE])
            packet_checksum = struct.unpack_from("<I", raw, FRAME_SIZE - 4)[0]
            calc_checksum = xor_checksum(raw[:-4])
            if packet_checksum != calc_checksum:
                self.bad_checksum_count += 1
                self.desync_count += 1
                del self._buffer[0]
                continue

            unpacked = struct.unpack(FRAME_FORMAT, raw)
            _, version, frame_bytes, mcu_time_us, sample_index, *rest = unpacked
            channels = np.asarray(rest[:-1], dtype=np.int32)
            channels = apply_channel_remap(channels)
            if version != VERSION or frame_bytes != FRAME_SIZE:
                self.desync_count += 1
                del self._buffer[0]
                continue

            # Sanity bounds for ADS1299 24-bit signed codes.
            if np.any(channels > 0x7FFFFF) or np.any(channels < -0x800000):
                self.desync_count += 1
                del self._buffer[0]
                continue

            frames.append(
                SampleFrame(
                    mcu_time_us=mcu_time_us,
                    sample_index=sample_index,
                    channels_i32=channels,
                )
            )
            del self._buffer[:FRAME_SIZE]

        return frames
