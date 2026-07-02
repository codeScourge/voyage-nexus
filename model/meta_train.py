"""Run a grid of training experiments and compare results.

Output layout (all under repo ``checkpoints/``)::

    checkpoints/meta_<timestamp>_<id>/
      intermediate_fusion_eegnet_p033/   # one training run per experiment
      intermediate_fusion_eegnet_p066/
      cat_net_p100/
      manifest.json
      comparison.md
      comparison.png

Examples::

    uv run meta_train.py
    uv run meta_train.py --models intermediate_fusion_eegnet cat_net --fractions 0.33 0.66 1.0
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
import zlib
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset

from data import SPLITS_OUTPUT_DIR, DatasetSplits, load_dataset_splits
from models import ARCHITECTURES
from train import CHECKPOINT_DIR, SEED, format_duration, run_training

META_DIR_NAME_RE = re.compile(r"^meta_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
DEFAULT_DATA_FRACTIONS = (0.33, 0.66, 1.0)


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    architecture: str
    train_fraction: float

    @property
    def slug(self) -> str:
        pct = int(round(self.train_fraction * 100))
        return f"{self.architecture}_p{pct:03d}"


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    run_dir: Path
    train_samples: int
    best_val_acc: float
    best_epoch: int
    elapsed_s: float | None = None


def new_meta_dir_name(now: datetime | None = None) -> str:
    when = now or datetime.now()
    uid = uuid.uuid4().hex[:8]
    stamp = when.strftime("%Y-%m-%d_%H-%M-%S")
    return f"meta_{stamp}_{uid}"


def create_meta_dir(root: Path = CHECKPOINT_DIR) -> Path:
    base_name = new_meta_dir_name()
    meta_dir = root / base_name
    suffix = 2
    while meta_dir.exists():
        meta_dir = root / f"{base_name}_{suffix:02d}"
        suffix += 1
    meta_dir.mkdir(parents=True, exist_ok=False)
    return meta_dir


def subsample_seed(base_seed: int, architecture: str, fraction: float) -> int:
    tag = f"{architecture}:{fraction:.6f}".encode()
    return (base_seed + zlib.crc32(tag)) % (2**31)


def stratified_train_subset(
    splits: DatasetSplits,
    fraction: float,
    *,
    seed: int,
) -> DatasetSplits:
    """Keep val/test fixed; subsample train indices stratified by label."""
    if fraction >= 1.0 - 1e-9:
        return splits

    train_indices = list(splits.train.indices)
    if not train_indices:
        raise ValueError("train split is empty")

    by_label: dict[str, list[int]] = defaultdict(list)
    for index in train_indices:
        label = splits.dataset[index]["label"]
        by_label[label].append(index)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label_indices in by_label.values():
        n_keep = max(1, int(round(len(label_indices) * fraction)))
        n_keep = min(n_keep, len(label_indices))
        picked = rng.choice(label_indices, size=n_keep, replace=False)
        selected.extend(int(i) for i in picked)

    selected.sort()
    train_indices_arr = np.asarray(selected, dtype=np.int64)
    train_subset = Subset(splits.dataset, selected)
    return replace(
        splits,
        train=train_subset,
        train_indices=train_indices_arr,
    )


def default_experiment_grid(
    architectures: list[str] | None = None,
    fractions: list[float] | None = None,
) -> list[ExperimentSpec]:
    archs = architectures or sorted(ARCHITECTURES)
    fracs = fractions or list(DEFAULT_DATA_FRACTIONS)
    return [
        ExperimentSpec(architecture=arch, train_fraction=frac)
        for arch in archs
        for frac in fracs
    ]


def write_comparison_report(meta_dir: Path, results: list[ExperimentResult]) -> Path:
    path = meta_dir / "comparison.md"
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    architectures = sorted({r.spec.architecture for r in results})
    fractions = sorted({r.spec.train_fraction for r in results})
    lookup = {(r.spec.architecture, r.spec.train_fraction): r for r in results}

    lines = [
        "# Meta-training comparison",
        "",
        f"- **Generated:** {generated}",
        f"- **Meta directory:** `{meta_dir.resolve()}`",
        f"- **Experiments:** {len(results)}",
        "",
        "## Summary",
        "",
        "| architecture | train % | train n | best val acc | best epoch | run dir |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arch in architectures:
        for frac in fractions:
            result = lookup.get((arch, frac))
            if result is None:
                continue
            pct = int(round(frac * 100))
            lines.append(
                f"| {arch} | {pct}% | {result.train_samples} | "
                f"{result.best_val_acc:.4f} | {result.best_epoch} | `{result.run_dir.name}` |"
            )

    if results:
        best = max(results, key=lambda r: r.best_val_acc)
        lines.extend([
            "",
            "## Best run",
            "",
            f"- **{best.spec.architecture}** at **{int(round(best.spec.train_fraction * 100))}%** "
            f"train data — val acc **{best.best_val_acc:.4f}** (epoch {best.best_epoch})",
            f"- `{best.run_dir.resolve()}`",
        ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved comparison report -> {path}")
    return path


def plot_comparison(meta_dir: Path, results: list[ExperimentResult]) -> Path | None:
    if not results:
        return None

    architectures = sorted({r.spec.architecture for r in results})
    fractions = sorted({r.spec.train_fraction for r in results})
    lookup = {(r.spec.architecture, r.spec.train_fraction): r.best_val_acc for r in results}

    matrix = np.full((len(architectures), len(fractions)), np.nan)
    for i, arch in enumerate(architectures):
        for j, frac in enumerate(fractions):
            matrix[i, j] = lookup.get((arch, frac), np.nan)

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(fractions)), max(3, 0.8 * len(architectures))))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{int(round(f * 100))}%" for f in fractions])
    ax.set_yticks(range(len(architectures)))
    ax.set_yticklabels(architectures)
    ax.set_xlabel("training data fraction")
    ax.set_ylabel("architecture")
    ax.set_title("Best validation accuracy by model and data fraction")

    for i, arch in enumerate(architectures):
        for j, frac in enumerate(fractions):
            value = lookup.get((arch, frac))
            if value is None:
                continue
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="val acc")
    fig.tight_layout()
    path = meta_dir / "comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved comparison plot -> {path}")
    return path


def save_manifest(meta_dir: Path, results: list[ExperimentResult], *, seed: int) -> Path:
    payload = {
        "seed": seed,
        "generated_utc": datetime.now(UTC).isoformat(),
        "experiments": [
            {
                "architecture": r.spec.architecture,
                "train_fraction": r.spec.train_fraction,
                "slug": r.spec.slug,
                "train_samples": r.train_samples,
                "best_val_acc": r.best_val_acc,
                "best_epoch": r.best_epoch,
                "elapsed_s": r.elapsed_s,
                "run_dir": str(r.run_dir.resolve()),
            }
            for r in results
        ],
    }
    path = meta_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_meta_experiments(
    *,
    splits: DatasetSplits,
    experiments: list[ExperimentSpec],
    meta_dir: Path,
    seed: int = SEED,
    train_kwargs: dict | None = None,
) -> list[ExperimentResult]:
    import time

    results: list[ExperimentResult] = []
    total = len(experiments)

    for idx, spec in enumerate(experiments, start=1):
        print("")
        print("=" * 72)
        print(
            f"experiment {idx}/{total}: {spec.architecture} "
            f"@ {spec.train_fraction:.0%} train data ({spec.slug})"
        )
        print("=" * 72)

        subset_seed = subsample_seed(seed, spec.architecture, spec.train_fraction)
        experiment_splits = stratified_train_subset(
            splits,
            spec.train_fraction,
            seed=subset_seed,
        )
        train_n = len(experiment_splits.train.indices)
        print(f"train samples: {train_n} / {len(splits.train.indices)} full")

        run_dir = meta_dir / spec.slug
        t0 = time.perf_counter()
        outcome = run_training(
            architecture=spec.architecture,
            splits=experiment_splits,
            run_dir=run_dir,
            seed=seed,
            train_kwargs=train_kwargs,
        )
        elapsed = time.perf_counter() - t0

        result = ExperimentResult(
            spec=spec,
            run_dir=Path(outcome["run_dir"]),
            train_samples=train_n,
            best_val_acc=float(outcome["best_val_acc"]),
            best_epoch=int(outcome["best_epoch"]),
            elapsed_s=elapsed,
        )
        results.append(result)
        print(
            f"finished {spec.slug}: best val acc={result.best_val_acc:.4f} "
            f"@ epoch {result.best_epoch} in {format_duration(elapsed)}"
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple training experiments (model x data fraction) and compare.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(ARCHITECTURES),
        default=sorted(ARCHITECTURES),
        metavar="MODEL",
        help=f"architectures to train (default: all — {', '.join(sorted(ARCHITECTURES))})",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_DATA_FRACTIONS),
        help=f"training data fractions (default: {' '.join(str(f) for f in DEFAULT_DATA_FRACTIONS)})",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for fraction in args.fractions:
        if not (0.0 < fraction <= 1.0):
            raise SystemExit(f"invalid fraction {fraction!r}; expected 0 < f <= 1")

    splits = load_dataset_splits(SPLITS_OUTPUT_DIR)
    experiments = default_experiment_grid(args.models, args.fractions)
    meta_dir = create_meta_dir()

    grid_summary = ", ".join(f"{s.architecture}@{s.train_fraction:.0%}" for s in experiments)
    print(f"checkpoints: {CHECKPOINT_DIR.resolve()}")
    print(f"splits: {SPLITS_OUTPUT_DIR.resolve()}")
    print(f"meta run: {meta_dir.resolve()}")
    print(f"grid ({len(experiments)}): {grid_summary}")

    results = run_meta_experiments(
        splits=splits,
        experiments=experiments,
        meta_dir=meta_dir,
        seed=args.seed,
    )

    save_manifest(meta_dir, results, seed=args.seed)
    write_comparison_report(meta_dir, results)
    plot_comparison(meta_dir, results)

    print("")
    print(f"meta training complete — artifacts under {meta_dir}")


if __name__ == "__main__":
    main()
