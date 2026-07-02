"""Run full validation across a meta_train grid and aggregate results.

For each experiment subfolder under a meta run directory, calls the same
logic as ``val.py`` (per-run ``validation_report.md`` + ``embeddings_umap.png``),
then writes meta-level comparison artifacts:

    checkpoints/meta_<timestamp>_<id>/
      intermediate_fusion_eegnet_p033/
        validation_report.md      # per-experiment (from val.py)
        embeddings_umap.png
      ...
      validation_comparison.md    # cross-experiment summary
      validation_comparison.png   # val/test accuracy heatmaps
      validation_manifest.json

Examples::

    uv run meta_val.py
    uv run meta_val.py --meta-dir checkpoints/meta_2026-07-02_12-00-00_abc12345
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import SPLITS_OUTPUT_DIR, load_dataset_splits
from meta_train import META_DIR_NAME_RE
from train import CHECKPOINT_DIR, SEED, seed_everything
from val import (
    _metrics_by_split,
    checkpoint_paths_for_run,
    get_device,
    validate_run_dir,
)

SLUG_RE = re.compile(r"^(.+)_p(\d{3})$")
VALIDATION_COMPARISON_NAME = "validation_comparison.md"
VALIDATION_COMPARISON_PLOT_NAME = "validation_comparison.png"
VALIDATION_MANIFEST_NAME = "validation_manifest.json"


@dataclass
class ExperimentValidationResult:
    run_dir: Path
    slug: str
    architecture: str | None
    train_fraction: float | None
    train_samples: int | None
    metrics_by_split: dict[str, dict]
    ckpt_meta: dict


def latest_meta_dir(root: Path = CHECKPOINT_DIR) -> Path | None:
    metas = [path for path in root.iterdir() if path.is_dir() and META_DIR_NAME_RE.match(path.name)]
    if not metas:
        return None
    return max(metas, key=lambda path: path.name)


def resolve_meta_dir(meta_dir: Path | None, *, root: Path = CHECKPOINT_DIR) -> Path:
    if meta_dir is not None:
        path = Path(meta_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Meta directory not found: {path}")
        return path
    latest = latest_meta_dir(root)
    if latest is None:
        raise FileNotFoundError(f"No meta_* directories found under {root}")
    return latest


def discover_experiment_dirs(meta_dir: Path) -> list[Path]:
    runs: list[Path] = []
    for child in sorted(meta_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            checkpoint_paths_for_run(child)
        except FileNotFoundError:
            continue
        runs.append(child)
    return runs


def parse_experiment_slug(slug: str) -> tuple[str | None, float | None]:
    match = SLUG_RE.match(slug)
    if not match:
        return None, None
    architecture = match.group(1)
    fraction = int(match.group(2)) / 100.0
    return architecture, fraction


def load_training_manifest(meta_dir: Path) -> dict[str, dict]:
    path = meta_dir / "manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, dict] = {}
    for experiment in payload.get("experiments", []):
        slug = experiment.get("slug")
        run_dir = experiment.get("run_dir")
        if slug:
            lookup[slug] = experiment
        if run_dir:
            lookup[Path(run_dir).name] = experiment
    return lookup


def experiment_metadata(
    run_dir: Path,
    *,
    training_lookup: dict[str, dict],
) -> tuple[str, str | None, float | None, int | None]:
    slug = run_dir.name
    training = training_lookup.get(slug)
    if training is not None:
        return (
            slug,
            training.get("architecture"),
            training.get("train_fraction"),
            training.get("train_samples"),
        )
    architecture, fraction = parse_experiment_slug(slug)
    return slug, architecture, fraction, None


def split_metric(metrics_by_split: dict[str, dict], split_name: str, key: str) -> float | None:
    row = metrics_by_split.get(split_name)
    if row is None:
        return None
    value = row.get(key)
    return float(value) if value is not None else None


def run_meta_validation(
    *,
    meta_dir: Path,
    splits,
    device,
    batch_size: int,
    session_min_samples: int,
    session_top_k: int,
    use_color: bool,
    seed: int,
    run_embeddings: bool,
) -> list[ExperimentValidationResult]:
    experiment_dirs = discover_experiment_dirs(meta_dir)
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment run directories found under {meta_dir}")

    training_lookup = load_training_manifest(meta_dir)
    results: list[ExperimentValidationResult] = []
    total = len(experiment_dirs)

    for idx, run_dir in enumerate(experiment_dirs, start=1):
        slug, architecture, train_fraction, train_samples = experiment_metadata(
            run_dir,
            training_lookup=training_lookup,
        )
        print("")
        print("=" * 72)
        print(f"meta validation {idx}/{total}: {slug}")
        print("=" * 72)

        outcome = validate_run_dir(
            run_dir,
            splits=splits,
            device=device,
            batch_size=batch_size,
            session_min_samples=session_min_samples,
            session_top_k=session_top_k,
            use_color=use_color,
            seed=seed,
            run_embeddings=run_embeddings,
            save_report=True,
        )
        if outcome["report_path"] is not None:
            print(f"report: saved validation report to {outcome['report_path'].resolve()}")

        evaluated = outcome["evaluated"]
        if "best" not in evaluated:
            print(f"warning: no best.pt evaluation for {run_dir.name}, skipping aggregation")
            continue

        best_metrics, _label_to_idx, ckpt_meta = evaluated["best"]
        results.append(
            ExperimentValidationResult(
                run_dir=run_dir,
                slug=slug,
                architecture=architecture,
                train_fraction=train_fraction,
                train_samples=train_samples,
                metrics_by_split=_metrics_by_split(best_metrics),
                ckpt_meta=ckpt_meta,
            )
        )

    return results


def write_validation_comparison_report(
    meta_dir: Path,
    results: list[ExperimentValidationResult],
) -> Path:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Meta validation comparison",
        "",
        f"- **Generated:** {generated}",
        f"- **Meta directory:** `{meta_dir.resolve()}`",
        f"- **Experiments validated:** {len(results)}",
        "",
        "Per-experiment metrics use each run's `best.pt`.",
        "",
        "## Summary (best checkpoint)",
        "",
        "| architecture | train % | train n | val acc | test acc | val bal_acc | test bal_acc | val macro_f1 | test macro_f1 | run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for result in sorted(
        results,
        key=lambda row: (
            row.architecture or row.slug,
            row.train_fraction if row.train_fraction is not None else -1.0,
        ),
    ):
        metrics = result.metrics_by_split
        pct = "-" if result.train_fraction is None else f"{int(round(result.train_fraction * 100))}%"
        train_n = "-" if result.train_samples is None else str(result.train_samples)
        arch = result.architecture or result.slug

        def fmt(split: str, key: str) -> str:
            value = split_metric(metrics, split, key)
            return "-" if value is None else f"{value:.4f}"

        lines.append(
            f"| {arch} | {pct} | {train_n} | "
            f"{fmt('val', 'accuracy')} | {fmt('test', 'accuracy')} | "
            f"{fmt('val', 'balanced_accuracy')} | {fmt('test', 'balanced_accuracy')} | "
            f"{fmt('val', 'macro_f1')} | {fmt('test', 'macro_f1')} | `{result.slug}` |"
        )

    test_accuracies = [
        (result, split_metric(result.metrics_by_split, "test", "accuracy"))
        for result in results
    ]
    test_accuracies = [(result, value) for result, value in test_accuracies if value is not None]
    if test_accuracies:
        best_result, best_test_acc = max(test_accuracies, key=lambda item: item[1])
        pct_text = (
            f"**{int(round(best_result.train_fraction * 100))}%**"
            if best_result.train_fraction is not None
            else "unknown train %"
        )
        lines.extend([
            "",
            "## Best on held-out sessions (test)",
            "",
            f"- **{best_result.architecture or best_result.slug}** "
            f"at {pct_text} train data "
            f"— test acc **{best_test_acc:.4f}**",
            f"- `{best_result.run_dir.resolve()}`",
        ])

    path = meta_dir / VALIDATION_COMPARISON_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved validation comparison report -> {path}")
    return path


def _metric_matrix(
    results: list[ExperimentValidationResult],
    *,
    split_name: str,
    metric_key: str,
) -> tuple[list[str], list[float], np.ndarray]:
    architectures = sorted({result.architecture or result.slug for result in results})
    fractions = sorted(
        {
            result.train_fraction
            for result in results
            if result.train_fraction is not None
        }
    )
    lookup = {
        (result.architecture or result.slug, result.train_fraction): split_metric(
            result.metrics_by_split,
            split_name,
            metric_key,
        )
        for result in results
    }

    matrix = np.full((len(architectures), len(fractions)), np.nan)
    for i, architecture in enumerate(architectures):
        for j, fraction in enumerate(fractions):
            matrix[i, j] = lookup.get((architecture, fraction), np.nan)
    return architectures, fractions, matrix


def plot_validation_comparison(
    meta_dir: Path,
    results: list[ExperimentValidationResult],
) -> Path | None:
    if not results:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 3.6 * 2), max(3.5, 0.8 * len({r.architecture or r.slug for r in results}))))
    panels = (
        ("val", "accuracy", "Validation accuracy (best.pt)"),
        ("test", "accuracy", "Test accuracy (best.pt)"),
    )

    for ax, (split_name, metric_key, title) in zip(axes, panels, strict=True):
        architectures, fractions, matrix = _metric_matrix(
            results,
            split_name=split_name,
            metric_key=metric_key,
        )
        im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(len(fractions)))
        ax.set_xticklabels([f"{int(round(f * 100))}%" for f in fractions])
        ax.set_yticks(range(len(architectures)))
        ax.set_yticklabels(architectures)
        ax.set_xlabel("training data fraction")
        ax.set_ylabel("architecture")
        ax.set_title(title)
        for i, architecture in enumerate(architectures):
            for j, fraction in enumerate(fractions):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = meta_dir / VALIDATION_COMPARISON_PLOT_NAME
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved validation comparison plot -> {path}")
    return path


def save_validation_manifest(
    meta_dir: Path,
    results: list[ExperimentValidationResult],
    *,
    seed: int,
) -> Path:
    payload = {
        "seed": seed,
        "generated_utc": datetime.now(UTC).isoformat(),
        "experiments": [
            {
                "slug": result.slug,
                "architecture": result.architecture,
                "train_fraction": result.train_fraction,
                "train_samples": result.train_samples,
                "run_dir": str(result.run_dir.resolve()),
                "checkpoint_kind": "best",
                "train_val_acc_at_save": result.ckpt_meta.get("val_acc"),
                "metrics": {
                    split_name: {
                        key: row.get(key)
                        for key in (
                            "n_samples",
                            "loss",
                            "accuracy",
                            "balanced_accuracy",
                            "macro_f1",
                            "weighted_f1",
                        )
                    }
                    for split_name, row in result.metrics_by_split.items()
                },
            }
            for result in results
        ],
    }
    path = meta_dir / VALIDATION_MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all experiment runs under a meta_train directory.",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=None,
        help="meta run directory (default: latest meta_* under checkpoints/)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--session-top-k", type=int, default=7)
    parser.add_argument("--session-min-samples", type=int, default=3)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="skip per-experiment UMAP plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    meta_dir = resolve_meta_dir(args.meta_dir)
    splits = load_dataset_splits(SPLITS_OUTPUT_DIR)
    experiment_dirs = discover_experiment_dirs(meta_dir)

    print(f"checkpoints: {CHECKPOINT_DIR.resolve()}")
    print(f"splits: {SPLITS_OUTPUT_DIR.resolve()}")
    print(f"meta dir: {meta_dir.resolve()}")
    print(f"experiments: {len(experiment_dirs)}")

    results = run_meta_validation(
        meta_dir=meta_dir,
        splits=splits,
        device=device,
        batch_size=args.batch_size,
        session_min_samples=args.session_min_samples,
        session_top_k=args.session_top_k,
        use_color=not args.no_color,
        seed=args.seed,
        run_embeddings=not args.no_embeddings,
    )

    save_validation_manifest(meta_dir, results, seed=args.seed)
    write_validation_comparison_report(meta_dir, results)
    plot_validation_comparison(meta_dir, results)

    print("")
    print(f"meta validation complete — artifacts under {meta_dir.resolve()}")


if __name__ == "__main__":
    main()
