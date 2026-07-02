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
    uv run meta_train.py --continue
    uv run meta_train.py --continue --meta-dir checkpoints/meta_2026-07-02_12-00-00_abc12345
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
import zlib
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

from data import SPLITS_OUTPUT_DIR, DatasetSplits, load_dataset_splits
from models import ARCHITECTURES
from train import CHECKPOINT_DIR, SEED, format_duration, run_training

META_DIR_NAME_RE = re.compile(r"^meta_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
MANIFEST_NAME = "manifest.json"
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


def latest_meta_dir(root: Path = CHECKPOINT_DIR) -> Path | None:
    metas = [path for path in root.iterdir() if path.is_dir() and META_DIR_NAME_RE.match(path.name)]
    if not metas:
        return None
    return max(metas, key=lambda path: path.name)


def resolve_continue_meta_dir(meta_dir: Path | None, *, root: Path = CHECKPOINT_DIR) -> Path:
    if meta_dir is not None:
        path = Path(meta_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Meta directory not found: {path}")
        return path
    latest = latest_meta_dir(root)
    if latest is None:
        raise FileNotFoundError(f"No meta_* directories found under {root}")
    return latest


def experiment_complete(run_dir: Path) -> bool:
    return (run_dir / "best.pt").is_file()


def find_resume_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    for name in ("last.pt", "best.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    epoch_ckpts = sorted(run_dir.glob("epoch_*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    return None


def load_meta_manifest(meta_dir: Path) -> dict | None:
    path = meta_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_grid_from_manifest(payload: dict) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            architecture=entry["architecture"],
            train_fraction=float(entry["train_fraction"]),
        )
        for entry in payload.get("experiments", [])
    ]


def experiment_result_from_manifest(entry: dict, spec: ExperimentSpec) -> ExperimentResult:
    return ExperimentResult(
        spec=spec,
        run_dir=Path(entry["run_dir"]),
        train_samples=int(entry["train_samples"]),
        best_val_acc=float(entry["best_val_acc"]),
        best_epoch=int(entry["best_epoch"]),
        elapsed_s=entry.get("elapsed_s"),
    )


def load_experiment_result_from_checkpoint(
    spec: ExperimentSpec,
    run_dir: Path,
    *,
    splits: DatasetSplits,
    seed: int,
    elapsed_s: float | None = None,
) -> ExperimentResult:
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    subset_seed = subsample_seed(seed, spec.architecture, spec.train_fraction)
    experiment_splits = stratified_train_subset(
        splits,
        spec.train_fraction,
        seed=subset_seed,
    )
    return ExperimentResult(
        spec=spec,
        run_dir=run_dir,
        train_samples=len(experiment_splits.train.indices),
        best_val_acc=float(ckpt.get("val_acc", float("nan"))),
        best_epoch=int(ckpt.get("epoch", 0)),
        elapsed_s=elapsed_s,
    )


def partition_experiments(
    experiments: list[ExperimentSpec],
    meta_dir: Path,
    *,
    splits: DatasetSplits,
    seed: int,
    manifest_payload: dict | None,
) -> tuple[list[ExperimentResult], list[ExperimentSpec]]:
    manifest_by_slug: dict[str, dict] = {}
    if manifest_payload is not None:
        manifest_by_slug = {
            entry["slug"]: entry
            for entry in manifest_payload.get("experiments", [])
            if entry.get("slug")
        }

    completed: list[ExperimentResult] = []
    pending: list[ExperimentSpec] = []

    for spec in experiments:
        run_dir = meta_dir / spec.slug
        if not experiment_complete(run_dir):
            pending.append(spec)
            continue

        entry = manifest_by_slug.get(spec.slug)
        if entry is not None:
            completed.append(experiment_result_from_manifest(entry, spec))
        else:
            completed.append(
                load_experiment_result_from_checkpoint(
                    spec,
                    run_dir,
                    splits=splits,
                    seed=seed,
                )
            )
        print(f"skip {spec.slug}: already complete (best.pt)")

    return completed, pending


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


def run_single_experiment(
    spec: ExperimentSpec,
    *,
    meta_dir: Path,
    splits: DatasetSplits,
    seed: int,
    train_kwargs: dict | None = None,
) -> ExperimentResult:
    import time

    run_dir = meta_dir / spec.slug
    subset_seed = subsample_seed(seed, spec.architecture, spec.train_fraction)
    experiment_splits = stratified_train_subset(
        splits,
        spec.train_fraction,
        seed=subset_seed,
    )
    train_n = len(experiment_splits.train.indices)
    print(f"train samples: {train_n} / {len(splits.train.indices)} full")

    resume_from = find_resume_checkpoint(run_dir)
    if resume_from is not None:
        print(f"resuming {spec.slug} from {resume_from.name}")
    elif run_dir.exists():
        shutil.rmtree(run_dir)

    t0 = time.perf_counter()
    outcome = run_training(
        architecture=spec.architecture,
        splits=experiment_splits,
        run_dir=run_dir,
        continue_from=resume_from,
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
    print(
        f"finished {spec.slug}: best val acc={result.best_val_acc:.4f} "
        f"@ epoch {result.best_epoch} in {format_duration(elapsed)}"
    )
    return result


def run_meta_experiments(
    *,
    splits: DatasetSplits,
    experiments: list[ExperimentSpec],
    meta_dir: Path,
    seed: int = SEED,
    train_kwargs: dict | None = None,
    completed_count: int = 0,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    total = completed_count + len(experiments)

    for idx, spec in enumerate(experiments, start=1):
        print("")
        print("=" * 72)
        print(
            f"experiment {completed_count + idx}/{total}: {spec.architecture} "
            f"@ {spec.train_fraction:.0%} train data ({spec.slug})"
        )
        print("=" * 72)

        results.append(
            run_single_experiment(
                spec,
                meta_dir=meta_dir,
                splits=splits,
                seed=seed,
                train_kwargs=train_kwargs,
            )
        )

    return results


def merge_experiment_results(
    experiments: list[ExperimentSpec],
    completed: list[ExperimentResult],
    fresh: list[ExperimentResult],
) -> list[ExperimentResult]:
    by_slug = {result.spec.slug: result for result in (*completed, *fresh)}
    return [by_slug[spec.slug] for spec in experiments]


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
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_meta",
        help="resume the latest (or --meta-dir) meta run; skip completed experiments",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=None,
        help="meta run directory to continue (default: latest meta_* under checkpoints/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for fraction in args.fractions:
        if not (0.0 < fraction <= 1.0):
            raise SystemExit(f"invalid fraction {fraction!r}; expected 0 < f <= 1")

    splits = load_dataset_splits(SPLITS_OUTPUT_DIR)

    if args.continue_meta:
        meta_dir = resolve_continue_meta_dir(args.meta_dir)
        manifest = load_meta_manifest(meta_dir)
        if manifest is not None:
            seed = int(manifest.get("seed", args.seed))
            experiments = experiment_grid_from_manifest(manifest)
            if not experiments:
                raise SystemExit(f"manifest in {meta_dir} has no experiments")
            print(f"loaded grid from {meta_dir / MANIFEST_NAME}")
        else:
            seed = args.seed
            experiments = default_experiment_grid(args.models, args.fractions)
            print(f"no manifest found — using CLI grid ({len(experiments)} experiments)")

        completed, pending = partition_experiments(
            experiments,
            meta_dir,
            splits=splits,
            seed=seed,
            manifest_payload=manifest,
        )
        print(f"checkpoints: {CHECKPOINT_DIR.resolve()}")
        print(f"splits: {SPLITS_OUTPUT_DIR.resolve()}")
        print(f"meta run: {meta_dir.resolve()}")
        print(f"status: {len(completed)} complete, {len(pending)} remaining")

        fresh = run_meta_experiments(
            splits=splits,
            experiments=pending,
            meta_dir=meta_dir,
            seed=seed,
            completed_count=len(completed),
        ) if pending else []

        results = merge_experiment_results(experiments, completed, fresh)
    else:
        if args.meta_dir is not None:
            raise SystemExit("--meta-dir is only valid with --continue")

        experiments = default_experiment_grid(args.models, args.fractions)
        meta_dir = create_meta_dir()
        seed = args.seed

        grid_summary = ", ".join(f"{s.architecture}@{s.train_fraction:.0%}" for s in experiments)
        print(f"checkpoints: {CHECKPOINT_DIR.resolve()}")
        print(f"splits: {SPLITS_OUTPUT_DIR.resolve()}")
        print(f"meta run: {meta_dir.resolve()}")
        print(f"grid ({len(experiments)}): {grid_summary}")

        results = run_meta_experiments(
            splits=splits,
            experiments=experiments,
            meta_dir=meta_dir,
            seed=seed,
        )

    save_manifest(meta_dir, results, seed=seed)
    write_comparison_report(meta_dir, results)
    plot_comparison(meta_dir, results)

    print("")
    print(f"meta training complete — artifacts under {meta_dir}")


if __name__ == "__main__":
    main()
