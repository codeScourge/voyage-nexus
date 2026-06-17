"""Load labelled session windows and build train / val / test splits.

Val uses only sessions that never appear in train. Test mixes held-out events
from train sessions with samples from new sessions that are not used for val.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats
from torch.utils.data import Subset

from _viewer_core import discover_sessions
from data import SessionEventDataset, load_session_channels


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    dataset: SessionEventDataset
    train: Subset
    val: Subset
    test: Subset
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_sessions: tuple[Path, ...]
    val_sessions: tuple[Path, ...]
    test_new_sessions: tuple[Path, ...]


def split_sample_indices(
    sessions: list[Path],
    per_sample_sessions: Sequence[Path],
    *,
    seed: int = 42,
    train_session_fraction: float = 0.6,
    val_session_fraction_of_new: float = 0.5,
    test_event_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Return train/val/test sample indices and the session groups used for each."""
    if not sessions:
        raise ValueError("At least one session is required")
    if not (0.0 < train_session_fraction < 1.0):
        raise ValueError("train_session_fraction must be between 0 and 1")
    if not (0.0 < val_session_fraction_of_new <= 1.0):
        raise ValueError("val_session_fraction_of_new must be between 0 and 1")
    if not (0.0 < test_event_fraction < 1.0):
        raise ValueError("test_event_fraction must be between 0 and 1")

    session_to_indices: dict[Path, list[int]] = defaultdict(list)
    for index, session_dir in enumerate(per_sample_sessions):
        session_to_indices[Path(session_dir)].append(index)

    rng = np.random.default_rng(seed)
    session_order = rng.permutation(len(sessions))

    n_train_sessions = int(round(len(sessions) * train_session_fraction))
    if len(sessions) > 1:
        n_train_sessions = max(1, min(n_train_sessions, len(sessions) - 1))
    else:
        n_train_sessions = 1

    train_sessions = {sessions[int(i)] for i in session_order[:n_train_sessions]}
    new_sessions = [sessions[int(i)] for i in session_order[n_train_sessions:]]

    val_sessions: set[Path] = set()
    test_new_sessions: set[Path] = set()
    if new_sessions:
        n_val_sessions = max(1, int(round(len(new_sessions) * val_session_fraction_of_new)))
        n_val_sessions = min(n_val_sessions, len(new_sessions))
        val_sessions = set(new_sessions[:n_val_sessions])
        test_new_sessions = set(new_sessions[n_val_sessions:])

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for session_dir in sorted(val_sessions):
        val_indices.extend(session_to_indices[session_dir])

    for session_dir in sorted(test_new_sessions):
        test_indices.extend(session_to_indices[session_dir])

    for session_dir in sorted(train_sessions):
        indices = session_to_indices.get(session_dir, [])
        if not indices:
            continue
        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        event_order = rng.permutation(len(indices))
        n_test = max(1, int(round(len(indices) * test_event_fraction)))
        n_test = min(n_test, len(indices) - 1)
        test_indices.extend(indices[int(i)] for i in event_order[:n_test])
        train_indices.extend(indices[int(i)] for i in event_order[n_test:])

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(val_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
        tuple(sorted(train_sessions)),
        tuple(sorted(val_sessions)),
        tuple(sorted(test_new_sessions)),
    )


def load_dataset_splits(
    recordings_path: Path,
    *,
    pre_ms: float = 300.0,
    post_ms: float = 700.0,
    seed: int = 42,
    train_session_fraction: float = 0.6,
    val_session_fraction_of_new: float = 0.5,
    test_event_fraction: float = 0.15,
) -> DatasetSplits:
    sessions = discover_sessions(recordings_path)
    dataset = SessionEventDataset(
        sessions,
        pre_ms=pre_ms,
        post_ms=post_ms,
    )
    (
        train_indices,
        val_indices,
        test_indices,
        train_sessions,
        val_sessions,
        test_new_sessions,
    ) = split_sample_indices(
        sessions,
        dataset._session_dirs,
        seed=seed,
        train_session_fraction=train_session_fraction,
        val_session_fraction_of_new=val_session_fraction_of_new,
        test_event_fraction=test_event_fraction,
    )
    return DatasetSplits(
        dataset=dataset,
        train=Subset(dataset, train_indices.tolist()),
        val=Subset(dataset, val_indices.tolist()),
        test=Subset(dataset, test_indices.tolist()),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        test_new_sessions=test_new_sessions,
    )


def print_split_summary(splits: DatasetSplits) -> None:
    print(f"train: {len(splits.train)} samples from {len(splits.train_sessions)} sessions")
    print(f"val:   {len(splits.val)} samples from {len(splits.val_sessions)} new sessions")
    print(
        f"test:  {len(splits.test)} samples "
        f"({len(splits.test_new_sessions)} new sessions + held-out events from train sessions)"
    )


def window_amplitude_features(window: np.ndarray) -> dict[str, float]:
    """Simple per-window amplitude stats averaged across channels."""
    return {
        "rms": float(np.sqrt(np.mean(window**2))),
        "mean_abs": float(np.mean(np.abs(window))),
        "peak_to_peak": float(np.mean(np.ptp(window, axis=0))),
        "std": float(np.mean(np.std(window, axis=0))),
    }


def _outside_window_starts(
    n_rows: int,
    window_len: int,
    inside_mask: np.ndarray,
) -> np.ndarray:
    """Row indices where a window of length window_len lies entirely outside events."""
    if window_len > n_rows:
        return np.array([], dtype=np.int64)
    starts: list[int] = []
    for start in range(n_rows - window_len + 1):
        if not inside_mask[start : start + window_len].any():
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def collect_inside_outside_features(
    dataset: SessionEventDataset,
    *,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Collect per-window amplitude features inside labelled windows vs outside them."""
    batch = dataset.batch
    window_len = batch.window_len
    pre_samples = dataset.pre_samples
    post_samples = dataset.post_samples

    inside_by_feature: dict[str, list[float]] = defaultdict(list)
    outside_by_feature: dict[str, list[float]] = defaultdict(list)

    for window in batch.x:
        for name, value in window_amplitude_features(window).items():
            inside_by_feature[name].append(value)

    indices_by_session: dict[Path, list[int]] = defaultdict(list)
    for index, session_dir in enumerate(dataset._session_dirs):
        indices_by_session[Path(session_dir)].append(index)

    rng = np.random.default_rng(seed)
    for session_dir, sample_indices in indices_by_session.items():
        session = load_session_channels(session_dir)
        index_to_row = {
            int(sample_idx): row for row, sample_idx in enumerate(session.sample_indices)
        }

        inside_mask = np.zeros(session.frame_count, dtype=bool)
        for sample_index in batch.center_sample_index[sample_indices]:
            row_idx = index_to_row.get(int(sample_index))
            if row_idx is None:
                continue
            start = row_idx - pre_samples
            end = row_idx + post_samples + 1
            if start < 0 or end > session.frame_count:
                continue
            inside_mask[start:end] = True

        starts = _outside_window_starts(session.frame_count, window_len, inside_mask)
        if starts.size == 0:
            continue

        n_outside = len(sample_indices)
        chosen = rng.choice(starts, size=n_outside, replace=starts.size < n_outside)
        for start in chosen:
            window = session.channels[int(start) : int(start) + window_len, :]
            for name, value in window_amplitude_features(window).items():
                outside_by_feature[name].append(value)

    return {
        name: (
            np.asarray(inside_by_feature[name], dtype=np.float64),
            np.asarray(outside_by_feature.get(name, []), dtype=np.float64),
        )
        for name in inside_by_feature
    }


@dataclass(frozen=True, slots=True)
class AmplitudeStatResult:
    feature: str
    inside_mean: float
    outside_mean: float
    cohens_d: float
    welch_t_p: float
    mannwhitney_p: float


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def run_amplitude_stat_tests(
    inside_outside: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[AmplitudeStatResult]:
    results: list[AmplitudeStatResult] = []
    for feature, (inside, outside) in inside_outside.items():
        if inside.size == 0 or outside.size == 0:
            continue
        _, welch_p = stats.ttest_ind(inside, outside, equal_var=False)
        _, mw_p = stats.mannwhitneyu(inside, outside, alternative="two-sided")
        results.append(
            AmplitudeStatResult(
                feature=feature,
                inside_mean=float(inside.mean()),
                outside_mean=float(outside.mean()),
                cohens_d=_cohens_d(inside, outside),
                welch_t_p=float(welch_p),
                mannwhitney_p=float(mw_p),
            )
        )
    return results


def print_amplitude_stat_tests(results: Sequence[AmplitudeStatResult]) -> None:
    if not results:
        print("No amplitude stat tests (missing inside or outside windows).")
        return

    print("\nInside labelled windows vs outside (per-window amplitude features):")
    print(
        f"{'feature':<14} {'inside':>10} {'outside':>10} {'d':>8} "
        f"{'welch_p':>10} {'mw_p':>10}"
    )
    for row in results:
        print(
            f"{row.feature:<14} "
            f"{row.inside_mean:10.3f} {row.outside_mean:10.3f} {row.cohens_d:8.3f} "
            f"{row.welch_t_p:10.4g} {row.mannwhitney_p:10.4g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load EEG event windows and build train/val/test splits.")
    parser.add_argument(
        "recordings",
        nargs="?",
        type=Path,
        default=Path("../client/recordings"),
        help="Session directory or parent folder containing session_* dirs",
    )
    parser.add_argument("--pre-ms", type=float, default=300.0)
    parser.add_argument("--post-ms", type=float, default=700.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-session-fraction", type=float, default=0.6)
    parser.add_argument("--val-session-fraction-of-new", type=float, default=0.5)
    parser.add_argument("--test-event-fraction", type=float, default=0.15)
    parser.add_argument(
        "--stat-tests",
        action="store_true",
        help="Compare amplitude features inside labelled windows vs background EEG",
    )
    args = parser.parse_args()

    splits = load_dataset_splits(
        args.recordings,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        seed=args.seed,
        train_session_fraction=args.train_session_fraction,
        val_session_fraction_of_new=args.val_session_fraction_of_new,
        test_event_fraction=args.test_event_fraction,
    )
    print_split_summary(splits)

    if args.stat_tests:
        features = collect_inside_outside_features(splits.dataset, seed=args.seed)
        print_amplitude_stat_tests(run_amplitude_stat_tests(features))


if __name__ == "__main__":
    main()
