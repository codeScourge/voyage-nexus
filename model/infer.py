"""Live inference helpers — mirror FusionDataset + build_event_windows from train/data."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from _preprocessors import EEG_CHANNELS, EMG_CHANNELS


def fix_window_length(
    window: np.ndarray,
    target_len: int,
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center-crop or center-pad (T, C) to target_len — same as data.build_event_windows."""
    if window.ndim != 2:
        raise ValueError(f"expected (time, channels), got {window.shape}")
    length, channels = window.shape
    if length == target_len:
        return window.astype(np.float32, copy=False)
    if length >= target_len:
        off = (length - target_len) // 2
        return window[off : off + target_len, :].astype(np.float32, copy=True)
    out = np.full((target_len, channels), pad_value, dtype=np.float32)
    off = (target_len - length) // 2
    out[off : off + length, :] = window
    return out


def prepare_fusion_sample(window_tc: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eeg, emg each (1, C, T) — identical to FusionDataset.__getitem__."""
    x = torch.from_numpy(np.asarray(window_tc, dtype=np.float32))
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    eeg = x[:, EEG_CHANNELS].T.unsqueeze(0)
    emg = x[:, EMG_CHANNELS].T.unsqueeze(0)
    return eeg, emg


def prediction_from_logits(
    logits: np.ndarray,
    idx_to_label: dict[int, str],
) -> dict[str, Any]:
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=0).numpy()
    by_class_idx = [
        {"label": idx_to_label[i], "probability": float(probs[i])}
        for i in sorted(idx_to_label.keys())
    ]
    predictions = sorted(by_class_idx, key=lambda row: row["probability"], reverse=True)
    top = predictions[0]
    return {
        "predicted_label": top["label"],
        "predicted_probability": top["probability"],
        "predictions": predictions,
        "predictions_by_label": by_class_idx,
        "probs_by_label": {row["label"]: row["probability"] for row in by_class_idx},
        "logits": [float(v) for v in logits],
        "probs": [float(v) for v in probs],
    }


@torch.no_grad()
def predict_fusion_batch(
    model: nn.Module,
    windows_tc: list[np.ndarray],
    *,
    device: torch.device,
    idx_to_label: dict[int, str],
) -> list[dict[str, Any]]:
    if not windows_tc:
        return []
    eeg_batch: list[torch.Tensor] = []
    emg_batch: list[torch.Tensor] = []
    for window in windows_tc:
        eeg, emg = prepare_fusion_sample(window)
        eeg_batch.append(eeg)
        emg_batch.append(emg)
    eeg = torch.stack(eeg_batch, dim=0).to(device)
    emg = torch.stack(emg_batch, dim=0).to(device)
    logits = model(eeg, emg).cpu().numpy()
    return [prediction_from_logits(logits[i], idx_to_label) for i in range(logits.shape[0])]
