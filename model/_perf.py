"""Lightweight stage timing for dataset build and training."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, TextIO

import torch


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True, slots=True)
class PerfEntry:
    name: str
    seconds: float
    where: str


def _write(text: str, *, file: TextIO | None = None) -> None:
    out = file or sys.stdout
    print(text, file=out, flush=True)


class PerfReport:
    """Accumulates named stage timings and prints a readable breakdown."""

    def __init__(self, title: str = "", *, live: bool = False) -> None:
        self.title = title
        self.live = live
        self._totals: dict[tuple[str, str], float] = {}

    @contextmanager
    def section(self, name: str, *, where: str = "CPU") -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started, where=where)

    def add(self, name: str, seconds: float, *, where: str = "CPU") -> None:
        if seconds < 0.0:
            seconds = 0.0
        key = (name, where)
        self._totals[key] = self._totals.get(key, 0.0) + seconds
        if self.live:
            self.print_block()

    @property
    def entries(self) -> list[PerfEntry]:
        return [
            PerfEntry(name=name, seconds=seconds, where=where)
            for (name, where), seconds in self._totals.items()
        ]

    def total_seconds(self) -> float:
        return sum(self._totals.values())

    def sorted_entries(
        self,
        *,
        min_seconds: float = 0.005,
        min_pct: float = 0.5,
        max_rows: int = 12,
    ) -> list[PerfEntry]:
        entries = sorted(self.entries, key=lambda item: item.seconds, reverse=True)
        total = self.total_seconds()
        if not entries:
            return []

        visible: list[PerfEntry] = []
        for entry in entries:
            pct = (100.0 * entry.seconds / total) if total > 0.0 else 0.0
            if entry.seconds >= min_seconds and pct >= min_pct:
                visible.append(entry)
        if not visible:
            visible = entries[: min(max_rows, len(entries))]
        else:
            visible = visible[:max_rows]
        return visible

    def format_block(
        self,
        *,
        prefix: str = "",
        min_seconds: float = 0.005,
        min_pct: float = 0.5,
        max_rows: int = 12,
        wall_seconds: float | None = None,
    ) -> str:
        visible = self.sorted_entries(
            min_seconds=min_seconds,
            min_pct=min_pct,
            max_rows=max_rows,
        )
        total = wall_seconds if wall_seconds is not None and wall_seconds > 0.0 else self.total_seconds()
        if not visible:
            heading = f"{prefix}perf"
            if self.title:
                heading = f"{prefix}{self.title}"
            return f"{heading}: (no timings)"

        title = self.title or "perf"
        if prefix:
            title = f"{prefix}{title}"
        lines = [f"=== {title} ==="]
        name_width = max(len(entry.name) for entry in visible)
        name_width = max(name_width, 5)

        for entry in visible:
            pct = (100.0 * entry.seconds / total) if total > 0.0 else 0.0
            lines.append(
                f"  {entry.name:<{name_width}}  {entry.seconds:>7.2f}s  {pct:>4.0f}%  {entry.where}"
            )

        accounted_total = self.total_seconds()
        if wall_seconds is not None:
            lines.append(f"  {'total (wall)':<{name_width}}  {wall_seconds:>7.2f}s       —")
            if abs(wall_seconds - accounted_total) > 0.05:
                lines.append(
                    f"  (accounted {accounted_total:.2f}s; {wall_seconds - accounted_total:+.2f}s unaccounted)"
                )
        else:
            lines.append(f"  {'total':<{name_width}}  {accounted_total:>7.2f}s       —")

        hidden = len(self.entries) - len(visible)
        if hidden > 0:
            lines.append(f"  ({hidden} smaller stage(s) omitted)")
        return "\n".join(lines)

    def print_block(
        self,
        *,
        prefix: str = "",
        file: TextIO | None = None,
        write: Callable[..., None] | None = None,
        min_seconds: float = 0.005,
        min_pct: float = 0.5,
        max_rows: int = 12,
        wall_seconds: float | None = None,
    ) -> None:
        text = self.format_block(
            prefix=prefix,
            min_seconds=min_seconds,
            min_pct=min_pct,
            max_rows=max_rows,
            wall_seconds=wall_seconds,
        )
        emit = write or _write
        emit(text, file=file)
        if self.entries:
            emit("", file=file)

    def print_line(self, *, prefix: str = "", file: TextIO | None = None) -> None:
        self.print_block(prefix=prefix, file=file, min_pct=0.0, max_rows=20)

    def print_summary(
        self,
        *,
        prefix: str = "",
        file: TextIO | None = None,
        write: Callable[..., None] | None = None,
        wall_seconds: float | None = None,
    ) -> None:
        self.print_block(prefix=prefix, file=file, write=write, wall_seconds=wall_seconds)
