"""Session discovery and EEG window extraction for the web viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data import (
    channel_names_from_meta,
    load_events,
    load_session_channels,
)

REFERENCE_WINDOW_MS = 200.0
REFERENCE_WIDTH_CM = 7.0
CHANNEL_COUNT = 32


def discover_sessions(path: Path) -> list[Path]:
    path = Path(path)
    if (path / "session_meta.json").exists():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Not a session or recordings directory: {path}")
    sessions = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_dir() and (candidate / "session_meta.json").exists()
    )
    if not sessions:
        raise FileNotFoundError(f"No sessions under {path}")
    return sessions


def parse_positive_ms(value: float | str, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def ms_to_samples(sample_rate_hz: float, ms: float) -> int:
    return max(1, int(round((ms / 1000.0) * sample_rate_hz)))


def sample_period_ms(sample_rate_hz: float) -> float:
    return 1000.0 / sample_rate_hz


def target_span_ms(
    sample_rate_hz: float,
    target_start_row: int,
    target_end_row: int,
) -> float:
    span_samples = target_end_row - target_start_row + 1
    return span_samples * sample_period_ms(sample_rate_hz)


def time_axis_ms_from_slice_start(sample_rate_hz: float, window_len: int) -> np.ndarray:
    return np.arange(window_len, dtype=np.float64) / sample_rate_hz * 1000.0


def slice_rows(channels: np.ndarray, start_row: int, end_row: int) -> np.ndarray:
    if start_row < 0 or end_row < start_row or end_row >= channels.shape[0]:
        return np.zeros((0, channels.shape[1]), dtype=np.float32)
    return channels[start_row : end_row + 1, :]


def marker_times_ms(pre_ms: float, target_span_ms_value: float) -> tuple[float, float]:
    """Red lines at target sample start/end: pre ms into the window, then + span."""
    return pre_ms, pre_ms + target_span_ms_value


@dataclass(frozen=True, slots=True)
class ResolvedEvent:
    row_idx: int
    end_row_idx: int | None
    sample_index: int
    sample_index_end: int | None
    label: str
    event_type: str
    event_id: str


def event_row_span(row_idx: int, end_row_idx: int | None) -> tuple[int, int]:
    end_row = end_row_idx if end_row_idx is not None else row_idx
    if end_row < row_idx:
        end_row = row_idx
    return row_idx, end_row


def resolve_plottable_events(
    channels: np.ndarray,
    sample_indices: np.ndarray,
    events: list[dict[str, str]],
) -> list[ResolvedEvent]:
    index_to_row = {int(idx): row for row, idx in enumerate(sample_indices)}
    resolved: list[ResolvedEvent] = []

    for event in events:
        event_type = event.get("event_type", "")
        sample_text = event.get("sample_index_start", "")
        if not sample_text:
            continue
        sample_idx = int(sample_text)
        row_idx = index_to_row.get(sample_idx)
        if row_idx is None:
            continue

        end_row_idx: int | None = None
        sample_index_end: int | None = None
        end_text = event.get("sample_index_end", "").strip()
        if end_text:
            sample_index_end = int(end_text)
            end_row_idx = index_to_row.get(sample_index_end)

        start_row, end_row = event_row_span(row_idx, end_row_idx)
        if start_row < 0 or end_row >= channels.shape[0]:
            continue
        resolved.append(
            ResolvedEvent(
                row_idx=row_idx,
                end_row_idx=end_row_idx,
                sample_index=sample_idx,
                sample_index_end=sample_index_end,
                label=event.get("label_text", "") or "(no label)",
                event_type=event_type or "?",
                event_id=event.get("event_id", "") or "?",
            )
        )
    return resolved


@dataclass
class LoadedSession:
    session_dir: Path
    channels: np.ndarray
    sample_indices: np.ndarray
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    events: list[dict[str, str]] | None
    duration_s: float

    @property
    def frame_count(self) -> int:
        return int(self.channels.shape[0])


def window_payload(
    window: np.ndarray,
    time_ms: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    title: str,
    total_ms: float,
    window_ms: float,
    pre_ms: float,
    post_ms: float,
    info: str,
    marker_start_ms: float,
    marker_end_ms: float,
) -> dict:
    if window.size == 0:
        traces: list[list[float]] = [[] for _ in range(CHANNEL_COUNT)]
    else:
        traces = [window[:, ch].astype(float).tolist() for ch in range(window.shape[1])]
        while len(traces) < CHANNEL_COUNT:
            traces.append([])

    return {
        "title": title,
        "info": info,
        "total_ms": total_ms,
        "window_ms": window_ms,
        "pre_ms": pre_ms,
        "post_ms": post_ms,
        "plot_width_cm": REFERENCE_WIDTH_CM * max(total_ms, 1.0) / REFERENCE_WINDOW_MS,
        "time_ms": time_ms.astype(float).tolist(),
        "channel_names": list(channel_names),
        "channels": traces,
        "marker_start_ms": marker_start_ms,
        "marker_end_ms": marker_end_ms,
    }


class SessionStore:
    def __init__(self, session_dirs: list[Path]) -> None:
        self._session_dirs = session_dirs
        self._loaded: dict[Path, LoadedSession] = {}

    def _load(self, index: int) -> LoadedSession:
        if index < 0 or index >= len(self._session_dirs):
            raise IndexError("session index out of range")
        session_dir = self._session_dirs[index]
        if session_dir in self._loaded:
            return self._loaded[session_dir]

        session = load_session_channels(session_dir)
        events_path = session_dir / "events.csv"
        events = load_events(session_dir) if events_path.exists() else None

        if session.frame_count > 1:
            dt = np.diff(session.sample_indices.astype(np.float64)) / session.sample_rate_hz
            duration_s = float(np.sum(dt))
        else:
            duration_s = 0.0

        loaded = LoadedSession(
            session_dir=session_dir,
            channels=session.channels,
            sample_indices=session.sample_indices,
            sample_rate_hz=session.sample_rate_hz,
            channel_names=channel_names_from_meta(session.meta),
            events=events,
            duration_s=duration_s,
        )
        self._loaded[session_dir] = loaded
        return loaded

    def list_sessions(self) -> list[dict]:
        items: list[dict] = []
        for index, path in enumerate(self._session_dirs):
            try:
                session = self._load(index)
                items.append(
                    {
                        "index": index,
                        "name": path.name,
                        "frame_count": session.frame_count,
                        "duration_s": round(session.duration_s, 2),
                        "sample_rate_hz": session.sample_rate_hz,
                        "has_events": session.events is not None,
                    }
                )
            except Exception as exc:
                items.append({"index": index, "name": path.name, "error": str(exc)})
        return items

    def timeline_range(
        self,
        index: int,
        *,
        window_ms: float,
        pre_ms: float,
    ) -> dict:
        session = self._load(index)
        fs = session.sample_rate_hz
        pre_samples = ms_to_samples(fs, pre_ms)
        window_samples = ms_to_samples(fs, window_ms)
        min_row = pre_samples
        max_row = max(pre_samples, session.frame_count - window_samples + pre_samples)
        if max_row < min_row:
            return {"min_row": 0, "max_row": 0, "default_row": 0}
        return {"min_row": min_row, "max_row": max_row, "default_row": min_row}

    def list_events(self, index: int) -> dict:
        session = self._load(index)
        if session.events is None:
            return {"events": [], "skipped": 0, "message": "No events.csv in this session"}
        resolved = resolve_plottable_events(
            session.channels,
            session.sample_indices,
            session.events,
        )
        skipped = len(session.events) - len(resolved)
        fs = session.sample_rate_hz
        event_items: list[dict] = []
        for i, ev in enumerate(resolved):
            start_row, end_row = event_row_span(ev.row_idx, ev.end_row_idx)
            span_ms = target_span_ms(fs, start_row, end_row)
            event_items.append(
                {
                    "index": i,
                    "event_type": ev.event_type,
                    "label": ev.label,
                    "event_id": ev.event_id,
                    "sample_index": ev.sample_index,
                    "sample_index_end": ev.sample_index_end,
                    "window_ms": round(span_ms, 3),
                }
            )
        return {"events": event_items, "skipped": skipped}

    def timeline_window(
        self,
        index: int,
        *,
        target_row: int,
        window_ms: float,
        pre_ms: float,
        post_ms: float,
    ) -> dict:
        session = self._load(index)
        fs = session.sample_rate_hz
        pre_samples = ms_to_samples(fs, pre_ms)
        window_samples = ms_to_samples(fs, window_ms)

        target_start = target_row
        target_end = target_row
        span_ms = target_span_ms(fs, target_start, target_end)

        start_row = target_start - pre_samples
        end_row = start_row + window_samples - 1
        window = slice_rows(session.channels, start_row, end_row)
        if window.size == 0:
            return window_payload(
                window,
                np.array([]),
                session.channel_names,
                title=f"{session.session_dir.name} — timeline",
                total_ms=0.0,
                window_ms=window_ms,
                pre_ms=pre_ms,
                post_ms=post_ms,
                info="Window out of range",
                marker_start_ms=pre_ms,
                marker_end_ms=pre_ms + span_ms,
            )

        total_ms = window.shape[0] * 1000.0 / fs
        time_ms = time_axis_ms_from_slice_start(fs, window.shape[0])
        marker_start_ms, marker_end_ms = marker_times_ms(pre_ms, span_ms)

        target_sample = int(session.sample_indices[target_row])
        t_target_s = target_row / fs
        info = (
            f"target row {target_row} | sample {target_sample} | t ≈ {t_target_s:.3f} s | "
            f"window {window_ms:g} ms | pre {pre_ms:g} ms | post {post_ms:g} ms"
        )
        title = f"{session.session_dir.name} — timeline @ sample {target_sample}"
        return window_payload(
            window,
            time_ms,
            session.channel_names,
            title=title,
            total_ms=total_ms,
            window_ms=window_ms,
            pre_ms=pre_ms,
            post_ms=post_ms,
            info=info,
            marker_start_ms=marker_start_ms,
            marker_end_ms=marker_end_ms,
        )

    def event_window(
        self,
        index: int,
        event_index: int,
        *,
        pre_ms: float,
        post_ms: float,
    ) -> dict:
        session = self._load(index)
        if session.events is None:
            raise ValueError("No events in session")

        resolved = resolve_plottable_events(
            session.channels,
            session.sample_indices,
            session.events,
        )
        if event_index < 0 or event_index >= len(resolved):
            raise IndexError("event index out of range")

        ev = resolved[event_index]
        target_start, target_end = event_row_span(ev.row_idx, ev.end_row_idx)
        fs = session.sample_rate_hz
        span_ms = target_span_ms(fs, target_start, target_end)

        pre_samples = ms_to_samples(fs, pre_ms)
        post_samples = ms_to_samples(fs, post_ms)
        start_row = target_start - pre_samples
        end_row = target_end + post_samples
        window = slice_rows(session.channels, start_row, end_row)
        total_ms = window.shape[0] * 1000.0 / fs if window.size else 0.0
        time_ms = time_axis_ms_from_slice_start(fs, window.shape[0])
        marker_start_ms, marker_end_ms = marker_times_ms(pre_ms, span_ms)

        end_note = f" → {ev.sample_index_end}" if ev.sample_index_end is not None else ""
        info = (
            f"Event {event_index + 1}/{len(resolved)} | {ev.event_type} | {ev.label} | "
            f"id {ev.event_id} | sample {ev.sample_index}{end_note} | "
            f"window {span_ms:g} ms | pre {pre_ms:g} ms | post {post_ms:g} ms"
        )
        title = f"{session.session_dir.name} — event {event_index} | {ev.event_type} | {ev.label}"
        return window_payload(
            window,
            time_ms,
            session.channel_names,
            title=title,
            total_ms=total_ms,
            window_ms=span_ms,
            pre_ms=pre_ms,
            post_ms=post_ms,
            info=info,
            marker_start_ms=marker_start_ms,
            marker_end_ms=marker_end_ms,
        )
