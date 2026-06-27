"""Inspect cached dataset splits — transition windows and soft labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import (
    COLLECTION_SAY_S,
    TRANSITION_EVENT_TYPES,
    _parse_scramble_breaks_transition_event_id,
    _transition_phase_fractions,
    load_dataset_splits,
    transition_label_probs_from_event_id,
)

DEFAULT_SPLITS_DIR = Path(__file__).resolve().parent / "splits"


def _format_probs(probs: dict[str, float]) -> str:
    ordered = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{label}: {prob:.0%}" for label, prob in ordered)


def _boundary_label(kind: str) -> str:
    if kind == "silence_to_word":
        return "word start"
    if kind == "word_to_silence":
        return "word end"
    return kind


def _iter_transition_samples(dataset) -> list[dict]:
    batch = dataset.batch
    samples: list[dict] = []
    label_probs_rows = batch.label_probs or tuple(None for _ in batch.labels)

    for index, (event_type, event_id, hard_label, label_probs) in enumerate(
        zip(
            batch.event_types,
            batch.event_ids,
            batch.labels,
            label_probs_rows,
            strict=True,
        )
    ):
        if event_type not in TRANSITION_EVENT_TYPES:
            continue

        parsed = _parse_scramble_breaks_transition_event_id(event_id)
        if parsed is None:
            continue
        kind, word, shift_s = parsed
        silence_frac, word_frac = _transition_phase_fractions(
            shift_s=shift_s,
            window_s=COLLECTION_SAY_S,
            kind=kind,
        )
        probs = label_probs or transition_label_probs_from_event_id(
            event_id,
            event_type=event_type,
        )
        if probs is None:
            continue

        samples.append(
            {
                "index": index,
                "kind": kind,
                "boundary": _boundary_label(kind),
                "word": word,
                "shift_s": shift_s,
                "silence_frac": silence_frac,
                "word_frac": word_frac,
                "label_probs": probs,
                "hard_label": hard_label,
                "event_id": event_id,
            }
        )
    return samples


def _pick_samples(samples: list[dict], *, n: int) -> list[dict]:
    if n <= 0 or not samples:
        return []

    by_boundary: dict[str, list[dict]] = {"word start": [], "word end": []}
    for sample in samples:
        by_boundary.setdefault(sample["boundary"], []).append(sample)

    picked: list[dict] = []
    per_boundary = max(1, n // 2)
    for boundary in ("word start", "word end"):
        group = sorted(by_boundary.get(boundary, []), key=lambda s: s["word_frac"])
        if not group:
            continue
        if len(group) <= per_boundary:
            picked.extend(group)
            continue
        # Evenly spaced across the word-fraction range for this boundary.
        step = (len(group) - 1) / (per_boundary - 1)
        indices = {round(i * step) for i in range(per_boundary)}
        picked.extend(group[i] for i in sorted(indices))

    picked.sort(key=lambda s: (s["boundary"], s["word_frac"], s["index"]))
    if len(picked) > n:
        stride = len(picked) / n
        picked = [picked[round(i * stride)] for i in range(n)]
    return picked


def print_transition_samples(
    dataset,
    *,
    n: int = 10,
) -> None:
    samples = _iter_transition_samples(dataset)
    picked = _pick_samples(samples, n=n)

    print(f"Loaded {len(dataset)} windows ({len(samples)} transition samples)")
    print(f"Showing {len(picked)} samples (window = {COLLECTION_SAY_S:.1f}s)\n")

    for sample in picked:
        print(
            f"[{sample['index']:>5}] {sample['boundary']:>10} | "
            f"word={sample['word']!r} | shift={sample['shift_s']:+.3f}s"
        )
        print(
            f"        word in window: {sample['word_frac']:.0%}  "
            f"(silence: {sample['silence_frac']:.0%})"
        )
        print(f"        smoothed labels: {_format_probs(sample['label_probs'])}")
        print(f"        hard label: {sample['hard_label']}")
        print(f"        event_id: {sample['event_id']}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print transition-window samples with word fraction and soft labels.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help="Directory with splits_manifest.json and splits_windows.npz",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of transition samples to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_dataset_splits(args.splits_dir)
    print_transition_samples(splits.dataset, n=args.n)


if __name__ == "__main__":
    main()
