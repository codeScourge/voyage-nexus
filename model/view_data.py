"""Inspect cached dataset splits — per-split labels and transition geometry."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from data import (
    COLLECTION_SAY_S,
    MERGE_TRANSITIONS_INTO_SILENCE,
    SILENT_SPEECH_WORD_EVENT,
    TRANSITION_EVENT_TYPES,
    DatasetSplits,
    _parse_scramble_breaks_transition_event_id,
    _sample_silence_fraction,
    _sample_word_fraction,
    _transition_phase_fractions,
    format_split_name,
    load_dataset_splits,
    transition_label_probs_from_event_id,
)

DEFAULT_SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits"
SPLIT_CHOICES = ("train", "val", "test", "all")
_PURE_FRAC_EPS = 1e-9


def _format_probs(probs: dict[str, float]) -> str:
    ordered = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{label}: {prob:.0%}" for label, prob in ordered)


def _boundary_label(kind: str) -> str:
    if kind == "silence_to_word":
        return "word start"
    if kind == "word_to_silence":
        return "word end"
    return kind


def _is_full_silence_window(event_type: str, event_id: str) -> bool:
    return _sample_silence_fraction(event_type, event_id) >= 1.0 - _PURE_FRAC_EPS


def _transition_geometry_tag(event_type: str, event_id: str) -> Optional[str]:
    if event_type not in TRANSITION_EVENT_TYPES:
        return None
    if _is_full_silence_window(event_type, event_id):
        return "full silence"
    if _sample_word_fraction(event_type, event_id) >= 1.0 - _PURE_FRAC_EPS:
        return "full word"
    return "partial"


def _sample_kind(event_type: str, event_id: str) -> str:
    if event_type == SILENT_SPEECH_WORD_EVENT:
        return "word event"
    if event_type in TRANSITION_EVENT_TYPES:
        return "transition"
    if _is_full_silence_window(event_type, event_id):
        return "silence"
    return "other"


def _label_probs_for_index(dataset, index: int) -> Optional[dict[str, float]]:
    batch = dataset.batch
    if batch.label_probs:
        stored = batch.label_probs[index]
        if stored is not None:
            return stored
    event_type = batch.event_types[index]
    if event_type in TRANSITION_EVENT_TYPES:
        return transition_label_probs_from_event_id(
            batch.event_ids[index],
            event_type=event_type,
        )
    return None


def _iter_split_samples(dataset, indices: Sequence[int]) -> list[dict[str, Any]]:
    batch = dataset.batch
    samples: list[dict[str, Any]] = []

    for index in indices:
        event_type = batch.event_types[index]
        event_id = batch.event_ids[index]
        hard_label = batch.labels[index]
        label_probs = _label_probs_for_index(dataset, index)
        word_frac = _sample_word_fraction(event_type, event_id)
        silence_frac = _sample_silence_fraction(event_type, event_id)
        kind = _sample_kind(event_type, event_id)
        geometry = _transition_geometry_tag(event_type, event_id)

        word = hard_label
        boundary = ""
        shift_s: Optional[float] = None
        if event_type in TRANSITION_EVENT_TYPES:
            parsed = _parse_scramble_breaks_transition_event_id(event_id)
            if parsed is not None:
                transition_kind, word, shift_s = parsed
                boundary = _boundary_label(transition_kind)
                silence_frac, word_frac = _transition_phase_fractions(
                    shift_s=shift_s,
                    window_s=COLLECTION_SAY_S,
                    kind=transition_kind,
                )

        is_soft = label_probs is not None and any(
            prob > 0.0 and prob < 1.0 - _PURE_FRAC_EPS for prob in label_probs.values()
        )

        samples.append(
            {
                "index": int(index),
                "kind": kind,
                "geometry": geometry,
                "boundary": boundary,
                "word": word,
                "shift_s": shift_s,
                "silence_frac": silence_frac,
                "word_frac": word_frac,
                "label_probs": label_probs,
                "hard_label": hard_label,
                "is_soft": is_soft,
                "event_type": event_type,
                "event_id": event_id,
            }
        )
    return samples


def _summarize_samples(samples: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(samples),
        "word_events": 0,
        "transitions": 0,
        "transitions_full_word": 0,
        "transitions_full_silence": 0,
        "transitions_partial": 0,
        "silence_100pct": 0,
        "soft_label": 0,
    }
    for sample in samples:
        if sample["kind"] == "word event":
            counts["word_events"] += 1
        elif sample["kind"] == "transition":
            counts["transitions"] += 1
            geometry = sample["geometry"]
            if geometry == "full word":
                counts["transitions_full_word"] += 1
            elif geometry == "full silence":
                counts["transitions_full_silence"] += 1
            elif geometry == "partial":
                counts["transitions_partial"] += 1
        if _is_full_silence_window(sample["event_type"], sample["event_id"]):
            counts["silence_100pct"] += 1
        if sample["is_soft"]:
            counts["soft_label"] += 1
    return counts


def _break_transition_flags(manifest_path: Path) -> tuple[Optional[bool], Optional[bool], Optional[bool]]:
    if not manifest_path.exists():
        return None, None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("include_transitions_from_breaks_train"),
        manifest.get("include_transitions_from_breaks_val"),
        manifest.get("include_transitions_from_breaks_test"),
    )


def _evenly_pick(group: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not group:
        return []
    if len(group) <= n:
        return list(group)
    if n == 1:
        return [group[len(group) // 2]]
    step = (len(group) - 1) / (n - 1)
    indices = {round(i * step) for i in range(n)}
    return [group[i] for i in sorted(indices)]


def _pick_samples(samples: Sequence[dict[str, Any]], *, n: int) -> list[dict[str, Any]]:
    if n <= 0 or not samples:
        return []

    buckets: dict[str, list[dict[str, Any]]] = {
        "word event": [],
        "partial": [],
        "full silence": [],
        "full word": [],
    }
    for sample in samples:
        if sample["kind"] == "word event":
            buckets["word event"].append(sample)
        elif sample["geometry"] == "partial":
            buckets["partial"].append(sample)
        elif sample["geometry"] == "full silence":
            buckets["full silence"].append(sample)
        elif sample["geometry"] == "full word":
            buckets["full word"].append(sample)

    bucket_order = ("partial", "full silence", "full word", "word event")
    active = [
        (
            name,
            sorted(
                buckets[name],
                key=lambda s: (s.get("word_frac", 0.0), s["index"]),
            ),
        )
        for name in bucket_order
        if buckets[name]
    ]
    if not active:
        return []

    picked: list[dict[str, Any]] = []
    remaining = n
    for i, (_name, group) in enumerate(active):
        quota = remaining // (len(active) - i)
        remaining -= quota
        picked.extend(_evenly_pick(group, quota))

    picked.sort(key=lambda s: (s["kind"], s.get("word_frac", 0.0), s["index"]))
    return picked[:n]


def _format_labels_line(sample: dict[str, Any]) -> str:
    if sample["kind"] != "transition":
        return f"labels: hard only — {sample['hard_label']!r} (word event, no soft target)"
    probs = sample["label_probs"]
    if probs is None:
        return "labels: (missing)"
    if sample["is_soft"]:
        return f"labels: {_format_probs(probs)}"
    return f"labels: {_format_probs(probs)} (pure window — 100% one class)"


def _print_sample(sample: dict[str, Any]) -> None:
    tag = sample["geometry"] or sample["kind"]
    if sample["kind"] == "transition":
        print(
            f"[{sample['index']:>5}] {tag:>13} | {sample['boundary']:>10} | "
            f"word={sample['word']!r} | shift={sample['shift_s']:+.3f}s"
        )
    elif sample["kind"] == "silence":
        print(f"[{sample['index']:>5}] {tag:>13} | silence gap")
    else:
        print(f"[{sample['index']:>5}] {tag:>13} | word={sample['word']!r}")

    print(
        f"        word in window: {sample['word_frac']:.0%}  "
        f"(silence: {sample['silence_frac']:.0%})"
    )
    print(f"        {_format_labels_line(sample)}")
    print()


def _split_subset(splits: DatasetSplits, split_name: str):
    if split_name == "train":
        return splits.train
    if split_name == "val":
        return splits.val
    if split_name == "test":
        return splits.test
    raise ValueError(f"Unknown split: {split_name!r}")


def print_split_inspection(
    splits: DatasetSplits,
    *,
    split_name: str,
    n: int,
    manifest_path: Path,
) -> None:
    subset = _split_subset(splits, split_name)
    display_name = format_split_name(split_name)
    samples = _iter_split_samples(splits.dataset, subset.indices)
    summary = _summarize_samples(samples)
    picked = _pick_samples(samples, n=n)

    train_flag, val_flag, test_flag = _break_transition_flags(manifest_path)
    split_flag = {
        "train": train_flag,
        "val": val_flag,
        "test": test_flag,
    }.get(split_name)

    print(f"=== {display_name} ({split_name}) — {summary['total']} samples ===")
    if split_flag is not None:
        print(f"include_transitions_from_breaks_{split_name}={split_flag}")
    print(
        f"MERGE_TRANSITIONS_INTO_SILENCE={MERGE_TRANSITIONS_INTO_SILENCE}  "
        f"window={COLLECTION_SAY_S:.1f}s"
    )
    print(
        f"word_events={summary['word_events']}  "
        f"transitions={summary['transitions']} "
        f"(full_word={summary['transitions_full_word']}, "
        f"full_silence={summary['transitions_full_silence']}, "
        f"partial={summary['transitions_partial']})  "
        f"silence_100pct={summary['silence_100pct']}  "
        f"soft_label={summary['soft_label']}"
    )
    if summary["transitions"] == 0 and split_flag is False:
        print(
            "no break transitions (expected — flag is False); "
            "only word events here, so no soft labels"
        )
    elif summary["transitions"] > 0:
        print(
            "soft labels apply to transition windows only "
            f"({summary['soft_label']} mixed-label, "
            f"{summary['transitions'] - summary['soft_label']} pure-geometry)"
        )
    if summary["transitions_partial"] == 0 and summary["transitions"] > 0:
        print("no partial transitions in this split")
    print()

    if not picked:
        print("(no samples to show)\n")
        return

    print(f"Showing {len(picked)} of {n} requested samples:\n")
    for sample in picked:
        _print_sample(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect per-split samples: word events, transitions, and 100% silence.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help="Directory with splits_manifest.json and splits_windows.npz",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="all",
        help="Which split to inspect (val=intra, test=extra)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of samples to print per split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_dataset_splits(args.splits_dir)
    manifest_path = args.splits_dir / "splits_manifest.json"

    split_names = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split_name in split_names:
        print_split_inspection(
            splits,
            split_name=split_name,
            n=args.n,
            manifest_path=manifest_path,
        )


if __name__ == "__main__":
    main()
