from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from collections import defaultdict

from data import (
    COLLECTION_SAY_S,
    SILENT_SPEECH_WORD_EVENT,
    SPLIT_KIND,
    TARGET_WORDS,
    TRANSITION_EVENT_TYPES,
    TRANSITION_PURE_PHASE_FRAC,
    _parse_scramble_breaks_transition_event_id,
    _sample_word_fraction,
    format_split_name,
    load_dataset_splits,
)
from models import get_embedding_taps
from train import (
    CHECKPOINT_DIR,
    FusionDataset,
    _per_label_colors,
    build_model_from_config,
    fusion_dataset_kwargs,
    get_device,
    list_run_dirs,
    run_dir_by_offset,
    seed_everything,
    soft_cross_entropy,
)

# --- report options (edit here; CLI / meta_val can override)
COMPARE_BEST_VS_LAST = True
# Base checkpoints to evaluate. When COMPARE_BEST_VS_LAST is True, last is added automatically.
EVAL_CHECKPOINTS = ("best",)
REVIEW_SPLITS = ("train", "val", "test")

# Per-split sections
SHOW_SPLIT_PER_CLASS = True
SHOW_SPLIT_CONFUSION = True
SHOW_SPLIT_SESSIONS = True

# Summary section
SHOW_SUMMARY = True
SHOW_SUMMARY_PER_CLASS = True
SHOW_SUMMARY_SESSIONS = True

# Best-vs-last comparison detail (only when COMPARE_BEST_VS_LAST and both exist)
SHOW_COMPARE_PER_CLASS = True
SHOW_COMPARE_SESSIONS = True

# Embeddings / UMAP
RUN_EMBEDDINGS = True
EMBEDDING_SPLITS = ("val", "test")
EMBEDDING_MAX_PER_LABEL = 200
EMBEDDINGS_N_NEIGHBORS = 15
EMBEDDINGS_MIN_DIST = 0.1
EMBEDDINGS_OUTPUT_NAME = "embeddings_umap.png"

# Session ranking thresholds
SESSION_TOP_K = 7
SESSION_MIN_SAMPLES = 3
BATCH_SIZE = 32

# Word-coverage detection (transition / partial-word windows)
SHOW_WORD_COVERAGE_DETECTION = True
WORD_COVERAGE_BIN_NAMES = ("full", "partial_word", "transition", "silence_side")

SPLIT_DISPLAY_WIDTH = 14
SESSION_DISPLAY_WIDTH = 22
VAL_REPORT_NAME = "validation_report.md"
VAL_METRICS_NAME = "validation_metrics.json"

ALL_SPLITS = ("train", "val", "test")
ALL_CHECKPOINTS = ("best", "last")


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    """Controls what validate_run_dir evaluates and prints."""

    compare_best_vs_last: bool = COMPARE_BEST_VS_LAST
    checkpoints: tuple[str, ...] = EVAL_CHECKPOINTS
    splits: tuple[str, ...] = REVIEW_SPLITS
    show_split_per_class: bool = SHOW_SPLIT_PER_CLASS
    show_split_confusion: bool = SHOW_SPLIT_CONFUSION
    show_split_sessions: bool = SHOW_SPLIT_SESSIONS
    show_summary: bool = SHOW_SUMMARY
    show_summary_per_class: bool = SHOW_SUMMARY_PER_CLASS
    show_summary_sessions: bool = SHOW_SUMMARY_SESSIONS
    show_compare_per_class: bool = SHOW_COMPARE_PER_CLASS
    show_compare_sessions: bool = SHOW_COMPARE_SESSIONS
    run_embeddings: bool = RUN_EMBEDDINGS
    embedding_splits: tuple[str, ...] = EMBEDDING_SPLITS
    embedding_max_per_label: int = EMBEDDING_MAX_PER_LABEL
    embeddings_n_neighbors: int = EMBEDDINGS_N_NEIGHBORS
    embeddings_min_dist: float = EMBEDDINGS_MIN_DIST
    session_top_k: int = SESSION_TOP_K
    session_min_samples: int = SESSION_MIN_SAMPLES
    batch_size: int = BATCH_SIZE
    show_word_coverage_detection: bool = SHOW_WORD_COVERAGE_DETECTION
    save_report: bool = True

    def resolved_checkpoints(self) -> tuple[str, ...]:
        kinds = [kind for kind in ALL_CHECKPOINTS if kind in self.checkpoints]
        if self.compare_best_vs_last:
            for kind in ALL_CHECKPOINTS:
                if kind not in kinds:
                    kinds.append(kind)
        if not kinds:
            raise ValueError(
                f"checkpoints must include at least one of {ALL_CHECKPOINTS}; got {self.checkpoints!r}"
            )
        return tuple(kinds)

    def resolved_splits(self) -> tuple[str, ...]:
        splits = tuple(name for name in ALL_SPLITS if name in self.splits)
        if not splits:
            raise ValueError(
                f"splits must include at least one of {ALL_SPLITS}; got {self.splits!r}"
            )
        return splits


def validation_options_from_args(args: argparse.Namespace) -> ValidationOptions:
    """Build options from an argparse namespace (val.py / meta_val.py CLIs)."""
    return ValidationOptions(
        compare_best_vs_last=args.compare_best_vs_last,
        checkpoints=tuple(args.checkpoints),
        splits=tuple(args.splits),
        show_split_per_class=args.split_per_class,
        show_split_confusion=args.split_confusion,
        show_split_sessions=args.split_sessions,
        show_summary=args.summary,
        show_summary_per_class=args.summary_per_class,
        show_summary_sessions=args.summary_sessions,
        show_compare_per_class=args.compare_per_class,
        show_compare_sessions=args.compare_sessions,
        run_embeddings=args.embeddings,
        embedding_splits=tuple(args.embedding_splits),
        session_top_k=args.session_top_k,
        session_min_samples=args.session_min_samples,
        batch_size=args.batch_size,
        show_word_coverage_detection=args.word_coverage_detection,
    )


def add_validation_option_args(parser: argparse.ArgumentParser) -> None:
    """Shared CLI flags for val.py and meta_val.py."""
    parser.add_argument(
        "--compare-best-vs-last",
        action=argparse.BooleanOptionalAction,
        default=COMPARE_BEST_VS_LAST,
        help="evaluate last.pt and print best-vs-last comparison",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        choices=list(ALL_CHECKPOINTS),
        default=list(EVAL_CHECKPOINTS),
        help="which checkpoints to evaluate (default from val.py)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(ALL_SPLITS),
        default=list(REVIEW_SPLITS),
        help="which splits to evaluate and print",
    )
    parser.add_argument(
        "--split-per-class",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SPLIT_PER_CLASS,
        help="per-class table inside each split section",
    )
    parser.add_argument(
        "--split-confusion",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SPLIT_CONFUSION,
        help="confusion matrix inside each split section",
    )
    parser.add_argument(
        "--split-sessions",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SPLIT_SESSIONS,
        help="per-session rankings inside each split section",
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SUMMARY,
        help="print the cross-split summary block",
    )
    parser.add_argument(
        "--summary-per-class",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SUMMARY_PER_CLASS,
        help="per-class recall table in the summary",
    )
    parser.add_argument(
        "--summary-sessions",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SUMMARY_SESSIONS,
        help="cross-split session rankings in the summary",
    )
    parser.add_argument(
        "--compare-per-class",
        action=argparse.BooleanOptionalAction,
        default=SHOW_COMPARE_PER_CLASS,
        help="per-class recall deltas in best-vs-last comparison",
    )
    parser.add_argument(
        "--compare-sessions",
        action=argparse.BooleanOptionalAction,
        default=SHOW_COMPARE_SESSIONS,
        help="session accuracy deltas in best-vs-last comparison",
    )
    parser.add_argument(
        "--embeddings",
        action=argparse.BooleanOptionalAction,
        default=RUN_EMBEDDINGS,
        help="run UMAP embedding plots after metrics",
    )
    parser.add_argument(
        "--embedding-splits",
        nargs="+",
        choices=["val", "test"],
        default=list(EMBEDDING_SPLITS),
        help="splits used for embedding UMAP",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--session-top-k",
        type=int,
        default=SESSION_TOP_K,
        help="number of best/worst sessions to print",
    )
    parser.add_argument(
        "--session-min-samples",
        type=int,
        default=SESSION_MIN_SAMPLES,
        help="minimum samples required to include a session in rankings",
    )
    parser.add_argument(
        "--word-coverage-detection",
        action=argparse.BooleanOptionalAction,
        default=SHOW_WORD_COVERAGE_DETECTION,
        help="word detection rate by window word-coverage bin (transition splits)",
    )

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


RESET = "\033[0m"
_BOLD = "\033[1m"


def _use_color(*, force: bool | None = None) -> bool:
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _bg_24bit(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


def _score_bg(value: float, *, vmin: float = 0.0, vmax: float = 1.0) -> str:
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        r = 170
        g = int(45 + t * 2 * 150)
        b = 45
    else:
        r = int(170 - (t - 0.5) * 2 * 120)
        g = int(195 + (t - 0.5) * 2 * 45)
        b = 45
    return _bg_24bit(r, g, b)


def _cm_bg(count: int, intensity: float, *, diagonal: bool) -> str:
    if count <= 0:
        return ""
    t = max(0.0, min(1.0, intensity))
    if diagonal:
        r = int(35 + (1.0 - t) * 35)
        g = int(45 + t * 170)
        b = int(35 + (1.0 - t) * 25)
    else:
        r = int(55 + t * 175)
        g = int(40 + (1.0 - t) * 25)
        b = int(40 + (1.0 - t) * 15)
    return _bg_24bit(r, g, b)


def _format_colored_value(
    text: str,
    *,
    bg: str,
    use_color: bool,
    width: int | None = None,
    bold: bool = False,
) -> str:
    if width is not None:
        text = text.rjust(width)
    if not use_color:
        return text
    prefix = f"{_BOLD}{bg}" if bold else bg
    if not bg and not bold:
        return text
    return f"{prefix}{text}{RESET}"


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _TeeStdout:
    def __init__(self, original: TextIO, buffer: list[str]) -> None:
        self._original = original
        self._buffer = buffer

    def write(self, text: str) -> int:
        self._original.write(text)
        self._buffer.append(text)
        return len(text)

    def flush(self) -> None:
        self._original.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    def fileno(self) -> int:
        return self._original.fileno()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


@contextmanager
def capture_report_output():
    buffer: list[str] = []
    original = sys.stdout
    sys.stdout = _TeeStdout(original, buffer)
    try:
        yield buffer
    finally:
        sys.stdout = original


def save_validation_report(
    run_dir: Path,
    *,
    buffer: list[str],
    checkpoint_hint: Path | None = None,
) -> Path:
    plain = _strip_ansi("".join(buffer)).rstrip()
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    meta_lines = [
        "# Validation Report",
        "",
        f"- **Generated:** {generated_at}",
        f"- **Run directory:** `{run_dir.resolve()}`",
    ]
    if checkpoint_hint is not None:
        meta_lines.append(f"- **Checkpoint hint:** `{checkpoint_hint.resolve()}`")
    meta_lines.extend(["", "---", "", "```text", plain, "```", ""])
    output_path = run_dir / VAL_REPORT_NAME
    output_path.write_text("\n".join(meta_lines), encoding="utf-8")
    return output_path


def options_cache_payload(options: ValidationOptions) -> dict[str, Any]:
    """Options that affect numeric validation results (not report formatting)."""
    return {
        "checkpoints": list(options.resolved_checkpoints()),
        "splits": list(options.resolved_splits()),
        "compare_best_vs_last": options.compare_best_vs_last,
        "session_min_samples": options.session_min_samples,
    }


def options_cache_compatible(cached: dict[str, Any], options: ValidationOptions) -> bool:
    current = options_cache_payload(options)
    for key in ("checkpoints", "splits", "compare_best_vs_last", "session_min_samples"):
        if cached.get(key) != current[key]:
            return False
    return True


def _metrics_to_jsonable(metrics: dict) -> dict:
    payload = dict(metrics)
    confusion = payload.get("confusion_matrix")
    if isinstance(confusion, np.ndarray):
        payload["confusion_matrix"] = confusion.tolist()
    return payload


def _metrics_from_jsonable(payload: dict) -> dict:
    metrics = dict(payload)
    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, list):
        metrics["confusion_matrix"] = np.asarray(confusion, dtype=np.int64)
    return metrics


def _serialize_evaluated(
    evaluated: dict[str, tuple[list[dict], dict[str, int], dict]],
) -> dict[str, dict]:
    serialized: dict[str, dict] = {}
    for kind, (metrics_list, label_to_idx, ckpt_meta) in evaluated.items():
        serialized[kind] = {
            "metrics": [_metrics_to_jsonable(metrics) for metrics in metrics_list],
            "label_to_idx": label_to_idx,
            "ckpt_meta": {
                key: ckpt_meta[key]
                for key in ("kind", "epoch", "epochs", "val_acc", "best_acc")
                if key in ckpt_meta
            },
        }
    return serialized


def _deserialize_evaluated(
    payload: dict[str, dict],
) -> dict[str, tuple[list[dict], dict[str, int], dict]]:
    evaluated: dict[str, tuple[list[dict], dict[str, int], dict]] = {}
    for kind, entry in payload.items():
        metrics_list = [_metrics_from_jsonable(metrics) for metrics in entry["metrics"]]
        evaluated[kind] = (metrics_list, entry["label_to_idx"], entry["ckpt_meta"])
    return evaluated


def validation_cache_fresh(
    run_dir: Path,
    *,
    checkpoint_kinds: tuple[str, ...],
    cache_path: Path,
) -> bool:
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for kind in checkpoint_kinds:
        checkpoint_path = run_dir / f"{kind}.pt"
        if checkpoint_path.exists() and checkpoint_path.stat().st_mtime > cache_mtime:
            return False
    return True


def save_validation_metrics(
    run_dir: Path,
    evaluated: dict[str, tuple[list[dict], dict[str, int], dict]],
    *,
    options: ValidationOptions,
) -> Path:
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "options": options_cache_payload(options),
        "checkpoints": _serialize_evaluated(evaluated),
    }
    output_path = run_dir / VAL_METRICS_NAME
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _split_name_from_display(display: str) -> str | None:
    normalized = display.strip()
    lowered = normalized.lower()
    if lowered in ("summary", "best vs last"):
        return None

    for split_name in ALL_SPLITS:
        if normalized == split_name or normalized == format_split_name(split_name):
            return split_name
        for kind in ("intra", "extra"):
            if normalized == f"{split_name} [{kind}]":
                return split_name

    token = normalized.split()[0]
    if token in ALL_SPLITS:
        return token
    if token in SPLIT_KIND.values():
        return {"intra": "val", "extra": "test"}[token]
    return None


_CHECKPOINT_SECTION_RE = re.compile(
    r"#{10,}\n# (?P<kind>BEST|LAST) CHECKPOINT\n#{10,}\n"
    r"checkpoint: .+\n"
    r"kind=(?P<kind_lower>\w+), epoch=(?P<epoch>\d+)/(?P<epochs>\d+), "
    r"val_acc=(?P<val_acc>[\d.]+), best_acc@train-time=(?P<best_acc>[\d.]+)\n",
    re.MULTILINE,
)
_SPLIT_SECTION_RE = re.compile(r"^=== (.+?) ===\n", re.MULTILINE)
_METRIC_LINE_RE = re.compile(
    r"^(samples|loss|accuracy|balanced_accuracy|macro_f1|weighted_f1):\s+([\d.]+)",
    re.MULTILINE,
)
_PER_CLASS_ROW_RE = re.compile(
    r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+",
    re.MULTILINE,
)
_SESSION_ROW_RE = re.compile(
    r"^(.+?)\s+([\d.]+)\s+[+-]?[\d.]+\s+\d+\s+(\d+)\s+",
)


def _load_label_to_idx_from_checkpoint(run_dir: Path, kind: str = "best") -> dict[str, int]:
    checkpoint_path = run_dir / f"{kind}.pt"
    if not checkpoint_path.exists():
        for fallback in ALL_CHECKPOINTS:
            checkpoint_path = run_dir / f"{fallback}.pt"
            if checkpoint_path.exists():
                break
        else:
            raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return dict(ckpt["label_to_idx"])


def _parse_split_sections(section_text: str) -> list[dict]:
    metrics_list: list[dict] = []
    matches = list(_SPLIT_SECTION_RE.finditer(section_text))
    for idx, match in enumerate(matches):
        split_display = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section_text)
        block = section_text[start:end]

        split_name = _split_name_from_display(split_display)
        if split_name is None:
            continue

        metric_values: dict[str, float | int] = {}
        for metric_match in _METRIC_LINE_RE.finditer(block):
            key, raw_value = metric_match.groups()
            if key == "samples":
                metric_values[key] = int(raw_value)
            else:
                metric_values[key] = float(raw_value)

        per_class: list[dict] = []
        per_class_start = block.find("per-class:")
        if per_class_start >= 0:
            per_class_block = block[per_class_start:]
            confusion_start = per_class_block.find("confusion matrix")
            if confusion_start >= 0:
                per_class_block = per_class_block[:confusion_start]
            header_end = per_class_block.find("\n", per_class_block.find("label"))
            if header_end >= 0:
                for row_match in _PER_CLASS_ROW_RE.finditer(per_class_block[header_end:]):
                    label, precision, recall, f1, support = row_match.groups()
                    per_class.append(
                        {
                            "class_idx": len(per_class),
                            "label": label,
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1),
                            "support": int(support),
                        }
                    )

        per_session: list[dict] = []
        session_start = block.find("per-session")
        if session_start >= 0:
            session_block = block[session_start:]
            header_end = session_block.find("\n", session_block.find("session"))
            if header_end >= 0:
                seen_sessions: set[str] = set()
                for line in session_block[header_end:].splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.endswith("sessions:"):
                        continue
                    row_match = _SESSION_ROW_RE.match(stripped)
                    if row_match is None:
                        continue
                    session, accuracy, n_samples = row_match.groups()
                    if session in seen_sessions:
                        continue
                    seen_sessions.add(session)
                    per_session.append(
                        {
                            "session": session,
                            "accuracy": float(accuracy),
                            "n_samples": int(n_samples),
                            "n_correct": int(round(float(accuracy) * int(n_samples))),
                        }
                    )
                per_session.sort(
                    key=lambda row: (row["accuracy"], row["n_samples"]),
                    reverse=True,
                )

        metrics_list.append(
            {
                "split": split_name,
                "n_samples": int(metric_values.get("samples", 0)),
                "loss": float(metric_values.get("loss", 0.0)),
                "accuracy": float(metric_values.get("accuracy", 0.0)),
                "balanced_accuracy": float(metric_values.get("balanced_accuracy", 0.0)),
                "macro_f1": float(metric_values.get("macro_f1", 0.0)),
                "weighted_f1": float(metric_values.get("weighted_f1", 0.0)),
                "per_class": per_class,
                "per_session": per_session,
            }
        )
    return metrics_list


def parse_validation_report(
    run_dir: Path,
    report_path: Path,
    *,
    options: ValidationOptions,
) -> dict[str, tuple[list[dict], dict[str, int], dict]] | None:
    """Rebuild evaluated metrics from a saved validation_report.md."""
    body = report_path.read_text(encoding="utf-8")
    text_match = re.search(r"```text\n(.*?)```", body, re.DOTALL)
    if text_match is None:
        return None
    report_text = text_match.group(1)

    evaluated: dict[str, tuple[list[dict], dict[str, int], dict]] = {}
    sections = list(_CHECKPOINT_SECTION_RE.finditer(report_text))
    if not sections:
        return None

    label_to_idx: dict[str, int] | None = None
    for idx, match in enumerate(sections):
        kind = match.group("kind_lower")
        start = match.end()
        end = sections[idx + 1].start() if idx + 1 < len(sections) else len(report_text)
        metrics_list = _parse_split_sections(report_text[start:end])
        if not metrics_list:
            continue

        if label_to_idx is None:
            label_to_idx = _load_label_to_idx_from_checkpoint(run_dir, kind)
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        for metrics in metrics_list:
            for class_idx, class_row in enumerate(metrics.get("per_class", [])):
                label = class_row.pop("label", None)
                if label is not None:
                    class_row["class_idx"] = label_to_idx.get(label, class_idx)
                else:
                    class_row["class_idx"] = class_idx
                class_row.setdefault("class_idx", class_idx)

        ckpt_meta = {
            "kind": kind,
            "epoch": int(match.group("epoch")),
            "epochs": int(match.group("epochs")),
            "val_acc": float(match.group("val_acc")),
            "best_acc": float(match.group("best_acc")),
        }
        evaluated[kind] = (metrics_list, label_to_idx, ckpt_meta)

    if not evaluated or label_to_idx is None:
        return None

    required_kinds = [kind for kind in options.resolved_checkpoints() if kind in evaluated]
    if "best" not in evaluated:
        return None
    if options.compare_best_vs_last and "last" in options.resolved_checkpoints() and "last" not in evaluated:
        return None
    if not required_kinds:
        return None

    required_splits = set(options.resolved_splits())
    for kind in required_kinds:
        split_names = {metrics["split"] for metrics in evaluated[kind][0]}
        if not required_splits.issubset(split_names):
            return None

    return evaluated


def load_cached_validation(
    run_dir: Path,
    *,
    options: ValidationOptions,
) -> dict[str, Any] | None:
    """Load validate_run_dir-compatible results from on-disk artifacts."""
    run_dir = Path(run_dir)
    report_path = run_dir / VAL_REPORT_NAME
    if not report_path.exists():
        return None

    metrics_path = run_dir / VAL_METRICS_NAME
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cached_options = payload.get("options", {})
        if options_cache_compatible(cached_options, options):
            checkpoint_kinds = tuple(payload.get("checkpoints", {}).keys())
            if validation_cache_fresh(
                run_dir,
                checkpoint_kinds=checkpoint_kinds,
                cache_path=metrics_path,
            ):
                evaluated = _deserialize_evaluated(payload["checkpoints"])
                if _evaluated_covers_options(evaluated, options):
                    return {
                        "run_dir": run_dir,
                        "evaluated": evaluated,
                        "report_path": report_path,
                        "options": options,
                        "from_cache": True,
                    }

    evaluated = parse_validation_report(run_dir, report_path, options=options)
    if evaluated is None or not _evaluated_covers_options(evaluated, options):
        return None

    save_validation_metrics(run_dir, evaluated, options=options)

    return {
        "run_dir": run_dir,
        "evaluated": evaluated,
        "report_path": report_path,
        "options": options,
        "from_cache": True,
    }


def _evaluated_covers_options(
    evaluated: dict[str, tuple[list[dict], dict[str, int], dict]],
    options: ValidationOptions,
) -> bool:
    if "best" not in evaluated:
        return False
    for kind in options.resolved_checkpoints():
        if kind not in evaluated:
            return False
    required_splits = set(options.resolved_splits())
    for kind in options.resolved_checkpoints():
        if kind not in evaluated:
            continue
        split_names = {metrics["split"] for metrics in evaluated[kind][0]}
        if not required_splits.issubset(split_names):
            return False
    return True


def session_name(session_dir: str | Path) -> str:
    return Path(session_dir).name


_SESSION_UID_MARKER = "_session_"


def format_session_display(name: str, *, width: int = 14) -> str:
    """Compact session label: ...{uid} for standard session folder names."""
    if _SESSION_UID_MARKER in name:
        uid = name.rsplit(_SESSION_UID_MARKER, 1)[-1]
        short = f"...{uid}"
    elif len(name) <= width:
        return name
    else:
        short = f"...{name[-(width - 3) :]}"
    return short if len(short) <= width else short[:width]


def format_split_display(split_name: str) -> str:
    return format_split_name(split_name)


def format_session_display_for_split(
    name: str,
    split_name: str,
    *,
    width: int = SESSION_DISPLAY_WIDTH,
) -> str:
    kind = SPLIT_KIND.get(split_name)
    suffix = f" [{kind}]" if kind else ""
    inner_width = max(1, width - len(suffix))
    return f"{format_session_display(name, width=inner_width)}{suffix}"


def compute_session_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    session_dirs: list[str],
    *,
    n_classes: int,
    min_samples: int = 1,
) -> list[dict]:
    by_session_cm: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((n_classes, n_classes), dtype=np.int64)
    )
    for t, p, session_dir in zip(y_true, y_pred, session_dirs, strict=True):
        by_session_cm[session_dir][t, p] += 1

    rows: list[dict] = []
    for session_dir, cm in by_session_cm.items():
        total = int(cm.sum())
        if total < min_samples:
            continue
        correct = int(np.trace(cm))

        per_class: list[dict] = []
        for class_idx in range(n_classes):
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            fn = cm[class_idx, :].sum() - tp
            support = cm[class_idx, :].sum()
            pred_support = cm[:, class_idx].sum()
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            per_class.append(
                {
                    "class_idx": class_idx,
                    "precision": float(precision),
                    "recall": float(recall),
                    "support": int(support),
                    "pred_support": int(pred_support),
                }
            )

        rows.append(
            {
                "session": session_name(session_dir),
                "session_dir": session_dir,
                "n_samples": total,
                "n_correct": correct,
                "accuracy": correct / total if total else 0.0,
                "per_class": per_class,
                "worst_recall": _worst_class_metric(per_class, metric="recall", min_key="support"),
                "worst_precision": _worst_class_metric(
                    per_class,
                    metric="precision",
                    min_key="pred_support",
                ),
            }
        )
    rows.sort(key=lambda row: (row["accuracy"], row["n_samples"]), reverse=True)
    return rows


def _worst_class_metric(
    per_class: list[dict],
    *,
    metric: str,
    min_key: str,
) -> dict | None:
    candidates = [row for row in per_class if row[min_key] > 0]
    if not candidates:
        return None
    worst = min(candidates, key=lambda row: (row[metric], -row[min_key]))
    return {
        "class_idx": worst["class_idx"],
        metric: worst[metric],
        "support": worst[min_key],
    }


def _format_worst_session_label(
    row: dict | None,
    *,
    metric: str,
    idx_to_label: dict[int, str],
    label_max: int = 12,
) -> str:
    if row is None:
        return "-"
    label = idx_to_label.get(row["class_idx"], str(row["class_idx"]))
    return f"{label[:label_max]} {row[metric]:.2f} (n={row['support']})"


def _most_confused_with(
    cm_row: np.ndarray,
    *,
    true_class_idx: int,
) -> tuple[int | None, int, float]:
    """Top off-diagonal prediction for a true class (pred_idx, count, fraction)."""
    support = int(cm_row.sum())
    if support <= 0:
        return None, 0, 0.0

    off_diag = cm_row.copy()
    off_diag[true_class_idx] = 0
    if off_diag.sum() == 0:
        return None, 0, 0.0

    pred_idx = int(off_diag.argmax())
    count = int(off_diag[pred_idx])
    return pred_idx, count, count / support


def _top_confusion_text(
    row: dict,
    *,
    idx_to_label: dict[int, str],
    label_max: int = 10,
) -> str:
    confused_idx = row.get("most_confused_with_idx")
    if confused_idx is None:
        return "-"
    confused_label = idx_to_label.get(confused_idx, str(confused_idx))
    count = int(row.get("most_confused_with_count", 0))
    frac = float(row.get("most_confused_with_fraction", 0.0))
    return f"{confused_label[:label_max]} {count} ({frac:.0%})"


def _format_top_confusion(
    row: dict,
    *,
    idx_to_label: dict[int, str],
    width: int = 22,
    label_max: int = 10,
) -> str:
    if row.get("support", 0) <= 0:
        return "n/a".rjust(width)
    return _top_confusion_text(
        row,
        idx_to_label=idx_to_label,
        label_max=label_max,
    ).rjust(width)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]
    label_to_idx: dict[str, int] = ckpt["label_to_idx"]
    state_dict = ckpt["model_state_dict"]

    model = build_model_from_config(model_config, state_dict=state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {
        "best_acc": float(ckpt.get("best_acc", float("nan"))),
        "epochs": int(ckpt.get("epochs", 0)),
        "epoch": int(ckpt.get("epoch", ckpt.get("epochs", 0))),
        "val_acc": float(ckpt.get("val_acc", float("nan"))),
        "kind": ckpt.get("kind", "unknown"),
        "checkpoint_device": ckpt.get("device", "unknown"),
        "model_config": model_config,
    }
    return model, label_to_idx, meta


_WORD_COVERAGE_EPS = 1e-9


def _word_coverage_bin(word_frac: float) -> str:
    if word_frac >= 1.0 - _WORD_COVERAGE_EPS:
        return "full"
    if word_frac >= TRANSITION_PURE_PHASE_FRAC:
        return "partial_word"
    if word_frac >= 1.0 - TRANSITION_PURE_PHASE_FRAC:
        return "transition"
    return "silence_side"


def _word_coverage_bin_label(bin_name: str, *, window_s: float) -> str:
    tau = TRANSITION_PURE_PHASE_FRAC
    ramp_s = window_s * (1.0 - tau)
    labels = {
        "full": f"full (phi_w=1.0, whole window in word)",
        "partial_word": (
            f"partial word ({tau:.0%}<=phi_w<1.0, ~{tau * window_s:.2f}-{window_s:.2f}s word in window)"
        ),
        "transition": (
            f"transition ({1.0 - tau:.0%}<=phi_w<{tau:.0%}, ~{ramp_s:.2f}-{tau * window_s:.2f}s word)"
        ),
        "silence_side": (
            f"silence side (phi_w<{1.0 - tau:.0%}, <{ramp_s:.2f}s word in window)"
        ),
    }
    return labels.get(bin_name, bin_name)


def _sample_target_word(
    event_type: str,
    event_id: str,
    hard_label: str,
) -> str | None:
    if event_type == SILENT_SPEECH_WORD_EVENT:
        return hard_label if hard_label in TARGET_WORDS else None
    if event_type in TRANSITION_EVENT_TYPES:
        parsed = _parse_scramble_breaks_transition_event_id(event_id)
        if parsed is None:
            return None
        word = parsed[1]
        return word if word in TARGET_WORDS else None
    return None


def _empty_word_coverage_bin() -> dict[str, float | int]:
    return {"n": 0, "correct": 0, "detection_rate": 0.0, "avg_word_frac": 0.0}


def compute_word_coverage_detection(
    dataset: FusionDataset,
    y_pred: np.ndarray,
    *,
    label_to_idx: dict[str, int],
) -> dict[str, Any]:
    """Lexical word hit rate stratified by how much of the window overlaps the word."""
    bins: dict[str, dict[str, float | int]] = {
        name: _empty_word_coverage_bin() for name in WORD_COVERAGE_BIN_NAMES
    }
    per_word: dict[str, dict[str, dict[str, float | int]]] = {
        word: {name: _empty_word_coverage_bin() for name in WORD_COVERAGE_BIN_NAMES}
        for word in TARGET_WORDS
    }
    word_frac_sums: dict[str, float] = defaultdict(float)
    n_transition_samples = 0
    n_word_event_samples = 0

    for dataset_idx, pred_idx in enumerate(y_pred.tolist()):
        base_idx = dataset.indices[dataset_idx]
        raw = dataset.base[base_idx]
        event_type = str(raw.get("event_type", ""))
        event_id = str(raw.get("event_id", ""))
        hard_label = str(raw.get("label", ""))

        target_word = _sample_target_word(event_type, event_id, hard_label)
        if target_word is None or target_word not in label_to_idx:
            continue

        word_frac = _sample_word_fraction(event_type, event_id)
        bin_name = _word_coverage_bin(word_frac)
        correct = int(pred_idx == label_to_idx[target_word])

        if event_type in TRANSITION_EVENT_TYPES:
            n_transition_samples += 1
        elif event_type == SILENT_SPEECH_WORD_EVENT:
            n_word_event_samples += 1

        for bucket in (bins, per_word[target_word]):
            row = bucket[bin_name]
            row["n"] = int(row["n"]) + 1
            row["correct"] = int(row["correct"]) + correct
            word_frac_sums[bin_name] += word_frac

    for bin_name, row in bins.items():
        n = int(row["n"])
        if n > 0:
            row["detection_rate"] = int(row["correct"]) / n
            row["avg_word_frac"] = word_frac_sums[bin_name] / n
        for word in TARGET_WORDS:
            word_row = per_word[word][bin_name]
            word_n = int(word_row["n"])
            if word_n > 0:
                word_row["detection_rate"] = int(word_row["correct"]) / word_n

    partial_n = sum(int(bins[name]["n"]) for name in ("partial_word", "transition", "silence_side"))
    partial_correct = sum(
        int(bins[name]["correct"]) for name in ("partial_word", "transition", "silence_side")
    )
    full_row = bins["full"]
    partial_word_row = bins["partial_word"]
    full_n = int(full_row["n"])
    partial_word_n = int(partial_word_row["n"])
    full_rate = float(full_row["detection_rate"]) if full_n else 0.0
    partial_word_rate = float(partial_word_row["detection_rate"]) if partial_word_n else 0.0
    partial_rate = partial_correct / partial_n if partial_n else 0.0

    return {
        "window_s": COLLECTION_SAY_S,
        "pure_phase_frac": TRANSITION_PURE_PHASE_FRAC,
        "bins": bins,
        "per_word": per_word,
        "n_transition_samples": n_transition_samples,
        "n_word_event_samples": n_word_event_samples,
        "comparisons": {
            "full_vs_partial_word": {
                "full_rate": full_rate,
                "partial_word_rate": partial_word_rate,
                "delta": full_rate - partial_word_rate,
                "full_n": full_n,
                "partial_word_n": partial_word_n,
            },
            "full_vs_all_partial": {
                "full_rate": full_rate,
                "partial_rate": partial_rate,
                "delta": full_rate - partial_rate,
                "full_n": full_n,
                "partial_n": partial_n,
            },
        },
    }


def print_word_coverage_detection(
    metrics: dict,
    *,
    use_color: bool = True,
) -> None:
    coverage = metrics.get("word_coverage_detection")
    if not coverage or coverage.get("n_transition_samples", 0) <= 0:
        return

    split_name = metrics["split"]
    window_s = float(coverage["window_s"])
    comparisons = coverage["comparisons"]
    full_vs_partial = comparisons["full_vs_partial_word"]
    full_vs_all = comparisons["full_vs_all_partial"]

    print(f"\nword coverage detection ({format_split_display(split_name)}):")
    print(
        "  lexical hit = model predicts the associated target word "
        "(not hard transition/silence label)"
    )
    print(
        f"  window={window_s:.2f}s, pure-transition band "
        f"phi_w in [{1.0 - coverage['pure_phase_frac']:.1f}, {coverage['pure_phase_frac']:.1f})"
    )
    print(
        f"{'bin':<52} {'detect':>10} {'n':>8} {'avg_phi_w':>10}"
    )
    for bin_name in WORD_COVERAGE_BIN_NAMES:
        row = coverage["bins"][bin_name]
        n = int(row["n"])
        if n <= 0:
            continue
        rate = float(row["detection_rate"])
        detect = _format_colored_value(
            f"{rate:.4f}",
            bg=_score_bg(rate),
            use_color=use_color,
            width=10,
        )
        label = _word_coverage_bin_label(bin_name, window_s=window_s)
        print(
            f"  {label:<50} {detect} {n:>8d} {float(row['avg_word_frac']):>10.3f}"
        )

    if int(full_vs_partial["full_n"]) > 0 and int(full_vs_partial["partial_word_n"]) > 0:
        delta = float(full_vs_partial["delta"])
        print(
            f"\n  full vs partial-word (phi_w>={coverage['pure_phase_frac']:.0%}): "
            f"{full_vs_partial['full_rate']:.4f} vs {full_vs_partial['partial_word_rate']:.4f} "
            f"(delta {delta:+.4f}, "
            f"n={full_vs_partial['full_n']}/{full_vs_partial['partial_word_n']})"
        )
    if int(full_vs_all["full_n"]) > 0 and int(full_vs_all["partial_n"]) > 0:
        delta = float(full_vs_all["delta"])
        print(
            f"  full vs all non-full: "
            f"{full_vs_all['full_rate']:.4f} vs {full_vs_all['partial_rate']:.4f} "
            f"(delta {delta:+.4f}, "
            f"n={full_vs_all['full_n']}/{full_vs_all['partial_n']})"
        )

    per_word_rows = []
    for word in TARGET_WORDS:
        word_bins = coverage["per_word"][word]
        full = word_bins["full"]
        partial = word_bins["partial_word"]
        full_n = int(full["n"])
        partial_n = int(partial["n"])
        if full_n == 0 and partial_n == 0:
            continue
        full_rate = float(full["detection_rate"]) if full_n else float("nan")
        partial_rate = float(partial["detection_rate"]) if partial_n else float("nan")
        delta = full_rate - partial_rate if full_n and partial_n else float("nan")
        per_word_rows.append((word, full_rate, full_n, partial_rate, partial_n, delta))

    if per_word_rows:
        print("\n  per-word full vs partial-word:")
        print(f"  {'word':<14} {'full':>10} {'n':>6} {'partial':>10} {'n':>6} {'delta':>10}")
        for word, full_rate, full_n, partial_rate, partial_n, delta in per_word_rows:
            full_text = f"{full_rate:.4f}" if full_n else "n/a"
            partial_text = f"{partial_rate:.4f}" if partial_n else "n/a"
            delta_text = f"{delta:+.4f}" if full_n and partial_n else "n/a"
            print(
                f"  {word:<14} {full_text:>10} {full_n:>6d} "
                f"{partial_text:>10} {partial_n:>6d} {delta_text:>10}"
            )


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataset: FusionDataset,
    *,
    device: torch.device,
    batch_size: int,
    split_name: str,
    session_min_samples: int = 3,
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_classes = len(dataset.label_to_idx)

    y_true: list[int] = []
    y_pred: list[int] = []
    session_dirs: list[str] = []
    running_loss = 0.0
    total = 0
    class_loss_sum = np.zeros(n_classes, dtype=np.float64)

    sample_offset = 0
    for eeg, emg, y_soft, y_hard in tqdm(loader, desc=split_name, leave=False):
        eeg, emg = eeg.to(device), emg.to(device)
        y_soft = y_soft.to(device)
        y_hard = y_hard.to(device)
        logits = model(eeg, emg)
        running_loss += soft_cross_entropy(logits, y_soft).item() * y_hard.size(0)
        loss_per_sample = F.cross_entropy(logits, y_hard, reduction="none")
        for class_idx, loss_val in zip(y_hard.tolist(), loss_per_sample.tolist(), strict=True):
            class_loss_sum[class_idx] += loss_val
        batch_preds = logits.argmax(1).cpu().tolist()
        batch_true = y_hard.cpu().tolist()
        y_true.extend(batch_true)
        y_pred.extend(batch_preds)
        for batch_idx in range(len(batch_true)):
            base_idx = dataset.indices[sample_offset + batch_idx]
            session_dirs.append(str(dataset.base[base_idx]["session_dir"]))
        sample_offset += len(batch_true)
        total += y_hard.size(0)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true_arr, y_pred_arr, strict=True):
        cm[t, p] += 1

    per_class = []
    for class_idx in range(n_classes):
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        support = cm[class_idx, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        confused_idx, confused_count, confused_frac = _most_confused_with(
            cm[class_idx],
            true_class_idx=class_idx,
        )
        class_loss = (
            float(class_loss_sum[class_idx] / support) if support > 0 else 0.0
        )
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
                "class_loss": class_loss,
                "most_confused_with_idx": confused_idx,
                "most_confused_with_count": confused_count,
                "most_confused_with_fraction": confused_frac,
            }
        )

    accuracy = float((y_true_arr == y_pred_arr).mean()) if total else 0.0
    recalls = [row["recall"] for row in per_class if row["support"] > 0]
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0

    weights = np.array([row["support"] for row in per_class], dtype=np.float64)
    weighted_f1 = (
        float(np.average([row["f1"] for row in per_class], weights=weights))
        if weights.sum() > 0
        else 0.0
    )

    per_session = compute_session_metrics(
        y_true_arr,
        y_pred_arr,
        session_dirs,
        n_classes=n_classes,
        min_samples=session_min_samples,
    )

    word_coverage_detection = compute_word_coverage_detection(
        dataset,
        y_pred_arr,
        label_to_idx=dataset.label_to_idx,
    )

    return {
        "split": split_name,
        "n_samples": total,
        "loss": running_loss / total if total else 0.0,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class": per_class,
        "per_session": per_session,
        "session_min_samples": session_min_samples,
        "word_coverage_detection": word_coverage_detection,
    }


def print_metrics(
    metrics: dict,
    *,
    idx_to_label: dict[int, str],
    use_color: bool = True,
    show_per_class: bool = True,
    show_confusion: bool = True,
) -> None:
    print(f"\n=== {format_split_display(metrics['split'])} ===")
    print(f"samples:           {metrics['n_samples']}")
    print(f"loss:              {metrics['loss']:.4f}")
    print(f"accuracy:          {metrics['accuracy']:.4f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"macro_f1:          {metrics['macro_f1']:.4f}")
    print(f"weighted_f1:       {metrics['weighted_f1']:.4f}")

    if show_per_class:
        print("\nper-class:")
        if use_color:
            print("  (cell color: metric value — red=low, green=high)")
        print(
            f"{'label':<20} {'precision':>10} {'recall':>10} {'f1':>10} "
            f"{'support':>10} {'top confusion':>28}"
        )
        for class_idx, row in enumerate(metrics["per_class"]):
            label = idx_to_label.get(class_idx, str(class_idx))
            precision = _format_colored_value(
                f"{row['precision']:.4f}",
                bg=_score_bg(row["precision"]),
                use_color=use_color,
                width=10,
            )
            recall = _format_colored_value(
                f"{row['recall']:.4f}",
                bg=_score_bg(row["recall"]),
                use_color=use_color,
                width=10,
            )
            f1 = _format_colored_value(
                f"{row['f1']:.4f}",
                bg=_score_bg(row["f1"]),
                use_color=use_color,
                width=10,
            )
            confused_idx = row.get("most_confused_with_idx")
            if confused_idx is None:
                top_confusion = "-"
            else:
                top_confusion = _top_confusion_text(
                    row,
                    idx_to_label=idx_to_label,
                    label_max=14,
                )
            print(
                f"{label:<20} "
                f"{precision} "
                f"{recall} "
                f"{f1} "
                f"{row['support']:>10d} "
                f"{top_confusion:>28}"
            )

    if show_confusion:
        print("\nconfusion matrix (rows=true, cols=pred):")
        if use_color:
            print("  (cell color: count intensity — green=correct, red=misclassified)")
        labels = [idx_to_label.get(i, str(i)) for i in range(len(metrics["per_class"]))]
        header = "true\\pred".ljust(20) + "".join(label[:12].rjust(12) for label in labels)
        print(header)

        cm = metrics["confusion_matrix"]
        cm_max = int(cm.max()) if cm.size else 0
        for class_idx, row in enumerate(cm):
            label = idx_to_label.get(class_idx, str(class_idx))
            row_max = int(row.max()) if row.size else 0
            cells: list[str] = []
            for pred_idx, value in enumerate(row):
                count = int(value)
                if cm_max > 0:
                    intensity = count / cm_max
                elif row_max > 0:
                    intensity = count / row_max
                else:
                    intensity = 0.0
                bg = _cm_bg(count, intensity, diagonal=class_idx == pred_idx)
                cells.append(
                    _format_colored_value(
                        str(count),
                        bg=bg,
                        use_color=use_color,
                        width=12,
                    )
                )
            print(f"{label[:20]:<20}{''.join(cells)}")


def print_session_rankings(
    metrics: dict,
    *,
    idx_to_label: dict[int, str],
    top_k: int = 5,
) -> None:
    per_session = metrics.get("per_session", [])
    if not per_session:
        min_samples = metrics.get("session_min_samples", 1)
        print(f"\nper-session: no sessions with >= {min_samples} samples")
        return

    overall_acc = metrics["accuracy"]
    split_name = metrics["split"]
    min_samples = metrics.get("session_min_samples", 1)
    print(
        f"\nper-session (>= {min_samples} samples, "
        f"{format_split_display(split_name)} accuracy={overall_acc:.4f}):"
    )

    def print_rows(title: str, rows: list[dict]) -> None:
        print(f"\n{title}:")
        print(
            f"{'session':<{SESSION_DISPLAY_WIDTH}} {'accuracy':>10} {'delta':>10} {'correct':>10} "
            f"{'samples':>10}  {'worst recall':<24} {'worst precision':<24}"
        )
        for row in rows:
            delta = row["accuracy"] - overall_acc
            worst_recall = _format_worst_session_label(
                row.get("worst_recall"),
                metric="recall",
                idx_to_label=idx_to_label,
            )
            worst_precision = _format_worst_session_label(
                row.get("worst_precision"),
                metric="precision",
                idx_to_label=idx_to_label,
            )
            session = format_session_display_for_split(row["session"], split_name)
            print(
                f"{session:<{SESSION_DISPLAY_WIDTH}} "
                f"{row['accuracy']:>10.4f} "
                f"{delta:>+10.4f} "
                f"{row['n_correct']:>10d} "
                f"{row['n_samples']:>10d}  "
                f"{worst_recall:<24} {worst_precision:<24}"
            )

    best = per_session[:top_k]
    if len(per_session) <= top_k:
        print_rows(f"all {len(per_session)} sessions", per_session)
        return

    worst = list(reversed(per_session[-top_k:]))
    print_rows(f"best {len(best)} sessions", best)
    print_rows(f"worst {len(worst)} sessions", worst)


def _format_recall(
    recall: float,
    support: int,
    *,
    use_color: bool = True,
    worst: bool = False,
) -> str:
    if support <= 0:
        return "     n/a"
    text = f"{recall:.4f}"
    return _format_colored_value(
        text,
        bg=_score_bg(recall),
        use_color=use_color,
        width=10,
        bold=worst,
    )


def _sessions_across_splits(metrics_list: list[dict]) -> list[dict]:
    combined: list[dict] = []
    for metrics in metrics_list:
        split_name = metrics["split"]
        split_acc = metrics["accuracy"]
        for row in metrics.get("per_session", []):
            combined.append(
                {
                    **row,
                    "split": split_name,
                    "delta": row["accuracy"] - split_acc,
                }
            )
    combined.sort(key=lambda row: (row["accuracy"], row["n_samples"]), reverse=True)
    return combined


def _print_cross_split_session_rankings(
    metrics_list: list[dict],
    *,
    idx_to_label: dict[int, str],
    top_k: int,
) -> None:
    combined = _sessions_across_splits(metrics_list)
    if not combined:
        min_samples = metrics_list[0].get("session_min_samples", 1) if metrics_list else 1
        print(f"\nper-session (all splits): no sessions with >= {min_samples} samples")
        return

    print(f"\nper-session across splits (top/bottom {top_k} by accuracy):")

    def print_rows(title: str, rows: list[dict]) -> None:
        print(f"\n{title}:")
        print(
            f"{'split':<{SPLIT_DISPLAY_WIDTH}} {'session':<{SESSION_DISPLAY_WIDTH}} {'accuracy':>10} "
            f"{'delta':>10} {'correct':>10} {'samples':>10}  "
            f"{'worst recall':<24} {'worst precision':<24}"
        )
        for row in rows:
            worst_recall = _format_worst_session_label(
                row.get("worst_recall"),
                metric="recall",
                idx_to_label=idx_to_label,
            )
            worst_precision = _format_worst_session_label(
                row.get("worst_precision"),
                metric="precision",
                idx_to_label=idx_to_label,
            )
            split_name = row["split"]
            session = format_session_display_for_split(row["session"], split_name)
            print(
                f"{format_split_display(split_name):<{SPLIT_DISPLAY_WIDTH}} "
                f"{session:<{SESSION_DISPLAY_WIDTH}} "
                f"{row['accuracy']:>10.4f} "
                f"{row['delta']:>+10.4f} "
                f"{row['n_correct']:>10d} "
                f"{row['n_samples']:>10d}  "
                f"{worst_recall:<24} {worst_precision:<24}"
            )

    if len(combined) <= top_k:
        print_rows(f"all {len(combined)} sessions", combined)
        return

    best = combined[:top_k]
    worst = list(reversed(combined[-top_k:]))
    print_rows(f"best {len(best)} sessions", best)
    print_rows(f"worst {len(worst)} sessions", worst)


def resolve_run_dir(
    checkpoint: Path | None,
    *,
    root: Path = CHECKPOINT_DIR,
    run_offset: int = 0,
) -> Path:
    if checkpoint is not None:
        if run_offset != 0:
            print("warning: --run-offset ignored when --checkpoint is set")
        return checkpoint.resolve().parent

    runs = list_run_dirs(root)
    if not runs:
        raise FileNotFoundError(f"No run directories found under {root}")

    run_dir = run_dir_by_offset(run_offset, root=root)
    if run_dir is None:
        raise FileNotFoundError(
            f"run offset {run_offset} out of range ({len(runs)} run(s) under {root}); "
            f"0=latest ({runs[0].name}), oldest={runs[-1].name}"
        )
    return run_dir


def checkpoint_paths_for_run(run_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for kind in ("best", "last"):
        path = run_dir / f"{kind}.pt"
        if path.exists():
            paths[kind] = path
    if not paths:
        raise FileNotFoundError(f"No best.pt or last.pt found in {run_dir}")
    return paths


SECTION_GAP_LINES = 4


def _print_section_gap() -> None:
    for _ in range(SECTION_GAP_LINES):
        print()


def print_checkpoint_banner(
    kind: str,
    checkpoint_path: Path,
    ckpt_meta: dict,
) -> None:
    bar = "#" * 72
    print(f"\n{bar}")
    print(f"# {kind.upper()} CHECKPOINT")
    print(f"{bar}")
    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(
        f"kind={ckpt_meta['kind']}, epoch={ckpt_meta['epoch']}/{ckpt_meta['epochs']}, "
        f"val_acc={ckpt_meta['val_acc']:.4f}, best_acc@train-time={ckpt_meta['best_acc']:.4f}"
    )


def _metrics_by_split(metrics_list: list[dict]) -> dict[str, dict]:
    return {metrics["split"]: metrics for metrics in metrics_list}


def _delta_text(delta: float, *, higher_is_better: bool, width: int = 10) -> str:
    improved = delta > 0 if higher_is_better else delta < 0
    worse = delta < 0 if higher_is_better else delta > 0
    text = f"{delta:+.4f}".rjust(width)
    if abs(delta) < 1e-9:
        return text
    if improved:
        return f"{text} ↑"
    if worse:
        return f"{text} ↓"
    return text


def _winner(
    best_value: float,
    last_value: float,
    *,
    higher_is_better: bool,
    eps: float = 1e-9,
) -> str:
    if abs(best_value - last_value) <= eps:
        return "tie"
    if higher_is_better:
        return "best" if best_value > last_value else "last"
    return "best" if best_value < last_value else "last"


COMPARE_SPLITS = ("train", "val", "test")
COMPARE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("loss", "loss", False),
    ("accuracy", "accuracy", True),
    ("bal_acc", "balanced_accuracy", True),
    ("macro_f1", "macro_f1", True),
    ("weighted_f1", "weighted_f1", True),
)


def print_model_comparison_meta(
    best_metrics_list: list[dict],
    last_metrics_list: list[dict],
    *,
    best_meta: dict,
    last_meta: dict,
    idx_to_label: dict[int, str],
    show_per_class: bool = True,
    show_sessions: bool = True,
) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print("=== best vs last ===")
    print(bar)

    print("\ncheckpoints:")
    print(f"{'model':<8} {'epoch':>12} {'val_acc@save':>14} {'best_acc@train':>16}")
    for kind, meta in (("best", best_meta), ("last", last_meta)):
        print(
            f"{kind:<8} "
            f"{meta['epoch']:>5}/{meta['epochs']:<5} "
            f"{meta['val_acc']:>14.4f} "
            f"{meta['best_acc']:>16.4f}"
        )

    best_by_split = _metrics_by_split(best_metrics_list)
    last_by_split = _metrics_by_split(last_metrics_list)
    split_names = [name for name in COMPARE_SPLITS if name in best_by_split and name in last_by_split]

    print("\nsplit metrics (delta = last - best; ↑ = last improved, ↓ = last worse):")
    header = (
        f"{'split':<{SPLIT_DISPLAY_WIDTH}}"
        + "".join(f"{label:>12}" for label, _, _ in COMPARE_METRICS)
        + f"{'winner':>10}"
    )
    print(header)
    for split_name in split_names:
        best_row = best_by_split[split_name]
        last_row = last_by_split[split_name]
        deltas = []
        winners: list[str] = []
        for _label, key, higher_is_better in COMPARE_METRICS:
            delta = last_row[key] - best_row[key]
            deltas.append(_delta_text(delta, higher_is_better=higher_is_better, width=12))
            winners.append(_winner(best_row[key], last_row[key], higher_is_better=higher_is_better))
        acc_winner = _winner(best_row["accuracy"], last_row["accuracy"], higher_is_better=True)
        winner_summary = acc_winner if acc_winner != "tie" else winners.count("last") - winners.count("best")
        if isinstance(winner_summary, int):
            if winner_summary > 0:
                winner_summary = "last"
            elif winner_summary < 0:
                winner_summary = "best"
            else:
                winner_summary = "tie"
        print(
            f"{format_split_display(split_name):<{SPLIT_DISPLAY_WIDTH}}"
            f"{''.join(deltas)}{str(winner_summary):>10}"
        )

    val_test_splits = [name for name in ("val", "test") if name in split_names]
    if show_per_class and val_test_splits:
        n_classes = len(next(iter(best_by_split.values()))["per_class"])
        print("\nper-class recall delta (last - best):")
        print(
            f"{'label':<20}"
            + "".join(
                f"{format_split_display(split) + ' Δ':>16}" for split in val_test_splits
            )
        )
        best_recall_wins = 0
        last_recall_wins = 0
        tie_recall_wins = 0
        for class_idx in range(n_classes):
            label = idx_to_label.get(class_idx, str(class_idx))
            cells: list[str] = []
            for split_name in val_test_splits:
                best_recall = best_by_split[split_name]["per_class"][class_idx]["recall"]
                last_recall = last_by_split[split_name]["per_class"][class_idx]["recall"]
                support = best_by_split[split_name]["per_class"][class_idx]["support"]
                if support <= 0:
                    cells.append("     n/a".rjust(16))
                    continue
                delta = last_recall - best_recall
                cells.append(_delta_text(delta, higher_is_better=True, width=16))
                winner = _winner(best_recall, last_recall, higher_is_better=True)
                if winner == "best":
                    best_recall_wins += 1
                elif winner == "last":
                    last_recall_wins += 1
                else:
                    tie_recall_wins += 1
            print(f"{label[:20]:<20}{''.join(cells)}")
        print(
            f"\nper-class recall head-to-head ({format_split_display('val')}+{format_split_display('test')}, support>0): "
            f"best={best_recall_wins}, last={last_recall_wins}, tie={tie_recall_wins}"
        )

    if show_sessions and val_test_splits:
        print("\nsession accuracy (last - best, shared sessions only):")
        for split_name in val_test_splits:
            best_sessions = {
                row["session_dir"]: row for row in best_by_split[split_name]["per_session"]
            }
            last_sessions = {
                row["session_dir"]: row for row in last_by_split[split_name]["per_session"]
            }
            common = sorted(set(best_sessions) & set(last_sessions))
            split_label = format_split_display(split_name)
            if not common:
                print(f"  {split_label}: no shared ranked sessions")
                continue
            deltas = [
                last_sessions[session_dir]["accuracy"] - best_sessions[session_dir]["accuracy"]
                for session_dir in common
            ]
            last_wins = sum(1 for delta in deltas if delta > 1e-9)
            best_wins = sum(1 for delta in deltas if delta < -1e-9)
            ties = len(deltas) - last_wins - best_wins
            mean_delta = float(np.mean(deltas))
            print(
                f"  {split_label}: n={len(common)}, mean Δacc={mean_delta:+.4f}, "
                f"last wins={last_wins}, best wins={best_wins}, tie={ties}"
            )


def _print_per_label_loss_acc_table(
    by_split: dict[str, dict],
    *,
    idx_to_label: dict[int, str],
    split_names: list[str],
) -> None:
    n_classes = len(next(iter(by_split.values()))["per_class"])
    split_width = max(
        len(format_split_display(name)) for name in split_names
    )

    def split_line(split_name: str, loss: float, acc: float) -> str:
        return (
            f"{format_split_display(split_name):<{split_width}}  "
            f"loss: {loss:.4f} --- acc: {acc:.4f}"
        )

    print("\nper-label loss and accuracy:")
    for class_idx in range(n_classes):
        label = idx_to_label.get(class_idx, str(class_idx))
        print(f"\n{label}")
        for split_name in split_names:
            row = by_split[split_name]["per_class"][class_idx]
            if row["support"] <= 0:
                print(f"  {format_split_display(split_name):<{split_width}}  n/a")
                continue
            print(f"  {split_line(split_name, row['class_loss'], row['recall'])}")


def print_summary_table(
    metrics_list: list[dict],
    *,
    idx_to_label: dict[int, str],
    use_color: bool = True,
    session_top_k: int = 7,
    show_per_class: bool = True,
    show_sessions: bool = True,
) -> None:
    print("\n=== summary ===")
    print(
        f"{'split':<{SPLIT_DISPLAY_WIDTH}} {'samples':>8} {'loss':>10} "
        f"{'accuracy':>10} {'bal_acc':>10} {'macro_f1':>10} {'weighted_f1':>12}"
    )
    for metrics in metrics_list:
        print(
            f"{format_split_display(metrics['split']):<{SPLIT_DISPLAY_WIDTH}} "
            f"{metrics['n_samples']:>8} "
            f"{metrics['loss']:>10.4f} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['balanced_accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['weighted_f1']:>12.4f}"
        )

    by_split = {metrics["split"]: metrics for metrics in metrics_list}
    split_names = [name for name in ALL_SPLITS if name in by_split]
    if not split_names:
        return

    if show_per_class:
        _print_per_label_loss_acc_table(
            by_split,
            idx_to_label=idx_to_label,
            split_names=split_names,
        )

        n_classes = len(next(iter(by_split.values()))["per_class"])
        confusion_width = 22
        print(
            "\nper-class recall and top confusion "
            f"(train vs {format_split_display('val')} vs {format_split_display('test')}):"
        )
        if use_color:
            print(
                "  (recall: red=low, green=high, bold=worst split; "
                "top confusion: pred label, count, fraction of class support)"
            )
        header = (
            f"{'label':<20}"
            + "".join(f"{format_split_display(name):>14}" for name in split_names)
            + "".join(
                f"{format_split_display(name) + ' confusion':>{confusion_width}}"
                for name in split_names
            )
        )
        print(header)
        for class_idx in range(n_classes):
            label = idx_to_label.get(class_idx, str(class_idx))
            split_rows: list[tuple[str, float, int]] = []
            for split_name in split_names:
                row = by_split[split_name]["per_class"][class_idx]
                split_rows.append((split_name, row["recall"], row["support"]))

            supported = [(name, recall) for name, recall, support in split_rows if support > 0]
            worst_split: str | None = None
            if len(supported) >= 2:
                worst_split = min(supported, key=lambda item: item[1])[0]

            recalls = []
            for split_name, recall, support in split_rows:
                recalls.append(
                    _format_recall(
                        recall,
                        support,
                        use_color=use_color,
                        worst=split_name == worst_split,
                    )
                )
            confusions = []
            for split_name in split_names:
                row = by_split[split_name]["per_class"][class_idx]
                confusions.append(
                    _format_top_confusion(
                        row,
                        idx_to_label=idx_to_label,
                        width=confusion_width,
                    )
                )
            print(f"{label[:20]:<20}{''.join(recalls)}{''.join(confusions)}")

    if show_sessions:
        _print_cross_split_session_rankings(
            metrics_list,
            idx_to_label=idx_to_label,
            top_k=session_top_k,
        )


def evaluate_checkpoint_report(
    kind: str,
    checkpoint_path: Path,
    *,
    splits,
    device: torch.device,
    options: ValidationOptions,
    use_color: bool,
) -> tuple[list[dict], dict[str, int], dict]:
    model, label_to_idx, ckpt_meta = load_checkpoint(checkpoint_path, device)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    print_checkpoint_banner(kind, checkpoint_path, ckpt_meta)

    split_indices = {
        "train": splits.train.indices,
        "val": splits.val.indices,
        "test": splits.test.indices,
    }
    review_splits = options.resolved_splits()

    all_metrics: list[dict] = []
    for split_name in review_splits:
        dataset = FusionDataset(
            splits.dataset,
            split_indices[split_name],
            label_to_idx,
            **fusion_dataset_kwargs(ckpt_meta["model_config"]),
        )
        metrics = evaluate_split(
            model,
            dataset,
            device=device,
            batch_size=options.batch_size,
            split_name=split_name,
            session_min_samples=options.session_min_samples,
        )
        all_metrics.append(metrics)
        print_metrics(
            metrics,
            idx_to_label=idx_to_label,
            use_color=use_color,
            show_per_class=options.show_split_per_class,
            show_confusion=options.show_split_confusion,
        )
        if options.show_split_sessions:
            print_session_rankings(
                metrics,
                idx_to_label=idx_to_label,
                top_k=options.session_top_k,
            )
        if options.show_word_coverage_detection:
            print_word_coverage_detection(
                metrics,
                use_color=use_color,
            )

    if options.show_summary:
        print_summary_table(
            all_metrics,
            idx_to_label=idx_to_label,
            use_color=use_color,
            session_top_k=options.session_top_k,
            show_per_class=options.show_summary_per_class,
            show_sessions=options.show_summary_sessions,
        )

    return all_metrics, label_to_idx, ckpt_meta


SPLIT_MARKERS = {
    "val": "o",
    "test": "s",
}


@dataclass(frozen=True, slots=True)
class EmbeddingSample:
    vectors: dict[str, np.ndarray]
    label: str
    split: str


def _sample_label(dataset: FusionDataset, dataset_index: int) -> str:
    base_idx = dataset.indices[dataset_index]
    return str(dataset.base[base_idx]["label"])


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    dataset: FusionDataset,
    *,
    device: torch.device,
    batch_size: int,
    split_name: str,
    max_per_label: int | None,
    rng: np.random.Generator,
) -> list[EmbeddingSample]:
    model.eval()
    tap_keys = list(get_embedding_taps(model))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    per_label: dict[str, list[EmbeddingSample]] = defaultdict(list)
    sample_offset = 0

    for eeg, emg, _y_soft, _y_hard in tqdm(loader, desc=f"embeddings:{split_name}", leave=False):
        eeg, emg = eeg.to(device), emg.to(device)
        taps = model.forward_embeddings(eeg, emg)
        batch_arrays = {key: taps[key].cpu().numpy() for key in tap_keys}
        batch_size_actual = next(iter(batch_arrays.values())).shape[0]

        for batch_idx in range(batch_size_actual):
            label = _sample_label(dataset, sample_offset + batch_idx)
            per_label[label].append(
                EmbeddingSample(
                    vectors={key: batch_arrays[key][batch_idx] for key in tap_keys},
                    label=label,
                    split=split_name,
                )
            )
        sample_offset += batch_size_actual

    selected: list[EmbeddingSample] = []
    for label, samples in sorted(per_label.items()):
        if max_per_label is not None and len(samples) > max_per_label:
            pick = rng.choice(len(samples), size=max_per_label, replace=False)
            samples = [samples[int(i)] for i in pick]
        selected.extend(samples)
    return selected


def _stack_embeddings(
    samples: list[EmbeddingSample],
    tap: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    matrix = np.stack([sample.vectors[tap] for sample in samples], axis=0)
    labels = [sample.label for sample in samples]
    splits = [sample.split for sample in samples]
    return matrix, labels, splits


def collect_embedding_samples(
    model: nn.Module,
    splits,
    label_to_idx: dict[str, int],
    *,
    model_config: dict,
    device: torch.device,
    batch_size: int,
    embedding_splits: tuple[str, ...],
    max_per_label: int,
    seed: int,
) -> list[EmbeddingSample]:
    split_indices = {
        "val": splits.val.indices,
        "test": splits.test.indices,
    }
    rng = np.random.default_rng(seed)
    samples: list[EmbeddingSample] = []

    for split_name in embedding_splits:
        if split_name not in split_indices:
            raise ValueError(f"unknown embedding split '{split_name}' (choose val, test)")
        dataset = FusionDataset(
            splits.dataset,
            split_indices[split_name],
            label_to_idx,
            **fusion_dataset_kwargs(model_config),
        )
        samples.extend(
            collect_embeddings(
                model,
                dataset,
                device=device,
                batch_size=batch_size,
                split_name=split_name,
                max_per_label=max_per_label,
                rng=rng,
            )
        )
    return samples


def _draw_umap_panel(
    ax: plt.Axes,
    samples: list[EmbeddingSample],
    *,
    tap_key: str,
    tap_title: str,
    label_colors: dict[str, str],
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    import umap

    labels_present = sorted({sample.label for sample in samples})
    matrix, labels, splits = _stack_embeddings(samples, tap_key)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, len(samples) - 1)),
        min_dist=min_dist,
        random_state=seed,
    )
    coords = reducer.fit_transform(matrix)

    for split_name, marker in SPLIT_MARKERS.items():
        split_mask = np.array([split == split_name for split in splits])
        if not split_mask.any():
            continue
        for label in labels_present:
            mask = split_mask & np.array([label_name == label for label_name in labels])
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[label_colors[label]],
                marker=marker,
                s=28,
                alpha=0.75,
                linewidths=0.4,
                edgecolors="white",
            )

    ax.set_title(tap_title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.25)


def plot_embedding_umap(
    checkpoint_samples: list[tuple[str, list[EmbeddingSample]]],
    *,
    embedding_taps: dict[str, str],
    output_path: Path,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    checkpoint_samples = [(kind, samples) for kind, samples in checkpoint_samples if samples]
    if not checkpoint_samples:
        print("embeddings: no samples collected; skipping UMAP plot")
        return

    all_samples = [sample for _, samples in checkpoint_samples for sample in samples]
    labels_present = sorted({sample.label for sample in all_samples})
    label_colors = dict(zip(labels_present, _per_label_colors(len(labels_present)), strict=True))

    n_rows = len(checkpoint_samples) * len(embedding_taps)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    row_idx = 0
    for kind, samples in checkpoint_samples:
        for tap_key, tap_title in embedding_taps.items():
            title = f"{kind.upper()} — {tap_title}"
            _draw_umap_panel(
                axes[row_idx],
                samples,
                tap_key=tap_key,
                tap_title=title,
                label_colors=label_colors,
                seed=seed,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            )
            row_idx += 1

    split_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="gray",
            linestyle="None",
            markersize=7,
            label=f"{format_split_display(split_name)} ({'dot' if marker == 'o' else 'square'})",
        )
        for split_name, marker in SPLIT_MARKERS.items()
        if any(sample.split == split_name for sample in all_samples)
    ]
    label_handles = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="None", markersize=7, label=label)
        for label, color in label_colors.items()
    ]

    fig.legend(
        handles=split_handles + label_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Fusion embeddings (dropout off)", y=1.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"embeddings: saved UMAP plot to {output_path.resolve()}")


def run_embedding_umap(
    ckpt_paths: dict[str, Path],
    splits,
    *,
    device: torch.device,
    batch_size: int,
    embedding_splits: tuple[str, ...],
    max_per_label: int,
    output_path: Path,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    kinds = [kind for kind in ("best", "last") if kind in ckpt_paths]
    if not kinds:
        return

    checkpoint_samples: list[tuple[str, list[EmbeddingSample]]] = []
    seen_paths: set[Path] = set()
    embedding_taps: dict[str, str] | None = None
    for kind in kinds:
        checkpoint_path = ckpt_paths[kind]
        resolved = checkpoint_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)

        model, label_to_idx, ckpt_meta = load_checkpoint(checkpoint_path, device)
        if embedding_taps is None:
            embedding_taps = get_embedding_taps(model)
        samples = collect_embedding_samples(
            model,
            splits,
            label_to_idx,
            model_config=ckpt_meta["model_config"],
            device=device,
            batch_size=batch_size,
            embedding_splits=embedding_splits,
            max_per_label=max_per_label,
            seed=seed,
        )
        print(
            f"embeddings ({kind}): collected {len(samples)} samples "
            f"from {', '.join(format_split_display(split) for split in embedding_splits)} "
            f"(max {max_per_label} per label per split)"
        )
        checkpoint_samples.append((kind, samples))

    if embedding_taps is None:
        return

    plot_embedding_umap(
        checkpoint_samples,
        embedding_taps=embedding_taps,
        output_path=output_path,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )


def validate_run_dir(
    run_dir: Path,
    *,
    splits,
    options: ValidationOptions | None = None,
    device: torch.device | None = None,
    use_color: bool = True,
    seed: int = 0,
    checkpoint_hint: Path | None = None,
) -> dict[str, Any]:
    """Evaluate selected checkpoints in a run directory; optional report and UMAP."""
    options = options or ValidationOptions()
    run_dir = Path(run_dir)
    if device is None:
        device = get_device()

    available = checkpoint_paths_for_run(run_dir)
    eval_kinds = options.resolved_checkpoints()
    ckpt_paths = {kind: available[kind] for kind in eval_kinds if kind in available}
    if not ckpt_paths:
        raise FileNotFoundError(
            f"None of requested checkpoints {eval_kinds} found in {run_dir} "
            f"(available: {sorted(available)})"
        )

    evaluated: dict[str, tuple[list[dict], dict[str, int], dict]] = {}
    report_path: Path | None = None

    with capture_report_output() as report_buffer:
        print(f"device: {device}")
        print(f"run_dir: {run_dir.resolve()}")
        print(
            f"options: checkpoints={list(ckpt_paths)} splits={list(options.resolved_splits())} "
            f"compare_best_vs_last={options.compare_best_vs_last} "
            f"embeddings={options.run_embeddings}"
        )

        for kind in eval_kinds:
            if kind not in available:
                print(f"\nwarning: {kind}.pt not found in {run_dir}, skipping")
                continue

            if evaluated:
                _print_section_gap()

            checkpoint_path = available[kind]
            if (
                kind == "last"
                and "best" in evaluated
                and "best" in available
                and checkpoint_path.resolve() == available["best"].resolve()
            ):
                _, label_to_idx, ckpt_meta = load_checkpoint(checkpoint_path, device)
                print_checkpoint_banner(kind, checkpoint_path, ckpt_meta)
                print("(identical to best.pt — reusing evaluation results)\n")
                evaluated[kind] = (evaluated["best"][0], label_to_idx, ckpt_meta)
                continue

            evaluated[kind] = evaluate_checkpoint_report(
                kind,
                checkpoint_path,
                splits=splits,
                device=device,
                options=options,
                use_color=use_color,
            )

        if options.compare_best_vs_last:
            if "best" in evaluated and "last" in evaluated:
                _print_section_gap()
                best_metrics, _best_label_to_idx, best_meta = evaluated["best"]
                last_metrics, _last_label_to_idx, last_meta = evaluated["last"]
                idx_to_label = {idx: label for label, idx in _best_label_to_idx.items()}
                print_model_comparison_meta(
                    best_metrics,
                    last_metrics,
                    best_meta=best_meta,
                    last_meta=last_meta,
                    idx_to_label=idx_to_label,
                    show_per_class=options.show_compare_per_class,
                    show_sessions=options.show_compare_sessions,
                )
            elif len(evaluated) == 1:
                only_kind = next(iter(evaluated))
                print(
                    f"\n(note: only {only_kind}.pt was evaluated; "
                    "need both best and last for comparison)"
                )

        if options.run_embeddings:
            run_embedding_umap(
                ckpt_paths,
                splits,
                device=device,
                batch_size=options.batch_size,
                embedding_splits=options.embedding_splits,
                max_per_label=options.embedding_max_per_label,
                output_path=run_dir / EMBEDDINGS_OUTPUT_NAME,
                seed=seed,
                n_neighbors=options.embeddings_n_neighbors,
                min_dist=options.embeddings_min_dist,
            )

    if options.save_report:
        report_path = save_validation_report(
            run_dir,
            buffer=report_buffer,
            checkpoint_hint=checkpoint_hint,
        )
        save_validation_metrics(run_dir, evaluated, options=options)

    return {
        "run_dir": run_dir,
        "evaluated": evaluated,
        "report_path": report_path,
        "options": options,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints from a training run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to any checkpoint in a run dir (default: latest run)",
    )
    parser.add_argument(
        "--run-offset",
        type=int,
        default=0,
        metavar="N",
        help="select run by age when --checkpoint is omitted: 0=latest, -1=previous, -2=...",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "splits",
        help="Directory with splits_manifest.json and splits_windows.npz",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in printed tables",
    )
    add_validation_option_args(parser)
    args = parser.parse_args()

    use_color = _use_color(force=not args.no_color)
    options = validation_options_from_args(args)

    seed_everything(args.seed)

    run_dir = resolve_run_dir(args.checkpoint, run_offset=args.run_offset)
    splits = load_dataset_splits(args.splits_dir)

    result = validate_run_dir(
        run_dir,
        splits=splits,
        options=options,
        use_color=use_color,
        seed=args.seed,
        checkpoint_hint=args.checkpoint,
    )
    if result["report_path"] is not None:
        print(f"report: saved validation report to {result['report_path'].resolve()}")


if __name__ == "__main__":
    main()
