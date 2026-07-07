"""Run full validation across a meta_train grid and aggregate results.

For each experiment subfolder under a meta run directory, reuses existing
per-run ``validation_report.md`` / ``validation_metrics.json`` when present
(otherwise calls ``val.py``), then writes meta-level comparison artifacts:

    checkpoints/meta_<timestamp>_<id>/
      intermediate_fusion_eegnet_p033/
        validation_report.md      # per-experiment (from val.py)
        validation_metrics.json   # structured cache (from val.py)
        embeddings_umap.png       # if embeddings enabled
      ...
      validation_comparison.md    # summary tables + concatenated reports
      validation_comparison.png   # val/test heatmaps (acc + macro_f1)
      validation_manifest.json

Examples::

    uv run meta_val.py
    uv run meta_val.py --meta-dir checkpoints/meta_2026-07-02_12-00-00_abc12345
    uv run meta_val.py --no-compare-best-vs-last --splits val test --no-embeddings
    uv run meta_val.py --force-revalidate
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import SPLITS_OUTPUT_DIR, load_dataset_splits
from meta_train import META_DIR_NAME_RE
from train import CHECKPOINT_DIR, SEED, seed_everything
from eval import (
    _metrics_by_split,
    add_validation_option_args,
    checkpoint_paths_for_run,
    format_session_display,
    format_split_display,
    get_device,
    load_cached_validation,
    validate_run_dir,
    validation_options_from_args,
    ValidationOptions,
)

SLUG_RE = re.compile(r"^(.+)_p(\d{3})$")
VALIDATION_COMPARISON_NAME = "validation_comparison.md"
VALIDATION_COMPARISON_PLOT_NAME = "validation_comparison.png"
VALIDATION_MANIFEST_NAME = "validation_manifest.json"

PRIMARY_SPLITS = ("test", "val")
PRIMARY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


@dataclass
class ExperimentValidationResult:
    run_dir: Path
    slug: str
    architecture: str | None
    train_fraction: float | None
    train_samples: int | None
    metrics_by_kind: dict[str, dict[str, dict]]
    ckpt_meta_by_kind: dict[str, dict]
    label_to_idx: dict[str, int]
    report_path: Path | None

    @property
    def metrics_by_split(self) -> dict[str, dict]:
        """Best-checkpoint metrics by split (meta aggregation default)."""
        return self.metrics_by_kind.get("best", {})

    @property
    def ckpt_meta(self) -> dict:
        return self.ckpt_meta_by_kind.get("best", {})

    @property
    def idx_to_label(self) -> dict[int, str]:
        return {idx: label for label, idx in self.label_to_idx.items()}


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


def _fmt(value: float | None, *, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_delta(value: float | None, *, digits: int = 4) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def _pct(result: ExperimentValidationResult) -> str:
    if result.train_fraction is None:
        return "-"
    return f"{int(round(result.train_fraction * 100))}%"


def _arch(result: ExperimentValidationResult) -> str:
    return result.architecture or result.slug


def _sorted_results(results: list[ExperimentValidationResult]) -> list[ExperimentValidationResult]:
    return sorted(
        results,
        key=lambda row: (
            row.architecture or row.slug,
            row.train_fraction if row.train_fraction is not None else -1.0,
        ),
    )


def _gap(
    metrics_by_split: dict[str, dict],
    from_split: str,
    to_split: str,
    key: str = "accuracy",
) -> float | None:
    left = split_metric(metrics_by_split, from_split, key)
    right = split_metric(metrics_by_split, to_split, key)
    if left is None or right is None:
        return None
    return left - right


def _worst_class(
    metrics_by_split: dict[str, dict],
    split_name: str,
    idx_to_label: dict[int, str],
) -> dict | None:
    row = metrics_by_split.get(split_name)
    if row is None:
        return None
    per_class = row.get("per_class", [])
    candidates: list[tuple[int, dict]] = []
    for class_idx, class_row in enumerate(per_class):
        if class_row.get("support", 0) <= 0:
            continue
        idx = int(class_row.get("class_idx", class_idx))
        candidates.append((idx, class_row))
    if not candidates:
        return None
    class_idx, class_row = min(
        candidates,
        key=lambda item: (item[1]["recall"], -item[1]["support"]),
    )
    return {
        "label": idx_to_label.get(class_idx, str(class_idx)),
        "recall": float(class_row["recall"]),
        "support": int(class_row["support"]),
    }


def _session_stats(metrics_by_split: dict[str, dict], split_name: str) -> dict | None:
    row = metrics_by_split.get(split_name)
    if row is None:
        return None
    sessions = row.get("per_session", [])
    if not sessions:
        return None
    best = sessions[0]
    worst = sessions[-1]
    return {
        "best_acc": float(best["accuracy"]),
        "best_session": best["session"],
        "worst_acc": float(worst["accuracy"]),
        "worst_session": worst["session"],
        "spread": float(best["accuracy"] - worst["accuracy"]),
        "n_sessions": len(sessions),
    }


def run_meta_validation(
    *,
    meta_dir: Path,
    splits,
    device,
    options: ValidationOptions,
    use_color: bool,
    seed: int,
    force_revalidate: bool = False,
) -> list[ExperimentValidationResult]:
    experiment_dirs = discover_experiment_dirs(meta_dir)
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment run directories found under {meta_dir}")

    training_lookup = load_training_manifest(meta_dir)
    results: list[ExperimentValidationResult] = []
    total = len(experiment_dirs)

    # Meta aggregation always needs best.pt metrics.
    options = replace(
        options,
        checkpoints=tuple(dict.fromkeys(("best",) + tuple(options.resolved_checkpoints()))),
        save_report=True,
    )

    for idx, run_dir in enumerate(experiment_dirs, start=1):
        slug, architecture, train_fraction, train_samples = experiment_metadata(
            run_dir,
            training_lookup=training_lookup,
        )
        print("")
        print("=" * 72)
        print(f"meta validation {idx}/{total}: {slug}")
        print("=" * 72)

        outcome = None
        if not force_revalidate:
            outcome = load_cached_validation(run_dir, options=options)
            if outcome is not None:
                print(
                    f"using cached validation artifacts from {run_dir.resolve()} "
                    f"(report + metrics)"
                )

        if outcome is None:
            outcome = validate_run_dir(
                run_dir,
                splits=splits,
                options=options,
                device=device,
                use_color=use_color,
                seed=seed,
            )
            if outcome["report_path"] is not None:
                print(f"report: saved validation report to {outcome['report_path'].resolve()}")

        evaluated = outcome["evaluated"]
        if "best" not in evaluated:
            print(f"warning: no best.pt evaluation for {run_dir.name}, skipping aggregation")
            continue

        metrics_by_kind: dict[str, dict[str, dict]] = {}
        ckpt_meta_by_kind: dict[str, dict] = {}
        label_to_idx: dict[str, int] | None = None
        for kind, (metrics_list, kind_label_to_idx, ckpt_meta) in evaluated.items():
            metrics_by_kind[kind] = _metrics_by_split(metrics_list)
            ckpt_meta_by_kind[kind] = ckpt_meta
            if label_to_idx is None:
                label_to_idx = kind_label_to_idx

        results.append(
            ExperimentValidationResult(
                run_dir=run_dir,
                slug=slug,
                architecture=architecture,
                train_fraction=train_fraction,
                train_samples=train_samples,
                metrics_by_kind=metrics_by_kind,
                ckpt_meta_by_kind=ckpt_meta_by_kind,
                label_to_idx=label_to_idx or {},
                report_path=outcome["report_path"],
            )
        )

    return results


def _options_summary_lines(options: ValidationOptions) -> list[str]:
    return [
        f"- **Checkpoints:** `{', '.join(options.resolved_checkpoints())}`",
        f"- **Compare best vs last:** `{options.compare_best_vs_last}`",
        f"- **Splits:** `{', '.join(options.resolved_splits())}`",
        f"- **Split detail:** per-class=`{options.show_split_per_class}`, "
        f"confusion=`{options.show_split_confusion}`, sessions=`{options.show_split_sessions}`",
        f"- **Summary detail:** enabled=`{options.show_summary}`, "
        f"per-class=`{options.show_summary_per_class}`, sessions=`{options.show_summary_sessions}`",
        f"- **Compare detail:** per-class=`{options.show_compare_per_class}`, "
        f"sessions=`{options.show_compare_sessions}`",
        f"- **UMAP embeddings:** `{options.run_embeddings}` "
        f"(splits=`{', '.join(options.embedding_splits)}`)",
    ]


def _leader_line(
    results: list[ExperimentValidationResult],
    *,
    split_name: str,
    key: str,
    higher_is_better: bool = True,
) -> str | None:
    scored = [
        (result, split_metric(result.metrics_by_split, split_name, key))
        for result in results
    ]
    scored = [(result, value) for result, value in scored if value is not None]
    if not scored:
        return None
    best_result, best_value = (max if higher_is_better else min)(scored, key=lambda item: item[1])
    return (
        f"- **{format_split_display(split_name)} {key}:** "
        f"`{_arch(best_result)}` @ `{_pct(best_result)}` → **{_fmt(best_value)}** "
        f"(`{best_result.slug}`)"
    )


def _write_primary_metrics_section(results: list[ExperimentValidationResult]) -> list[str]:
    lines = [
        "## Primary metrics (best.pt)",
        "",
        "Headline number is **macro_f1** (class imbalance: silence dominates support). "
        "Gaps are `train_acc − split_acc` (positive = overfit).",
        "",
        "| architecture | train % | train n "
        f"| {format_split_display('test')} acc | bal_acc | **macro_f1** "
        f"| {format_split_display('val')} acc | bal_acc | **macro_f1** "
        "| train→test gap | train→val gap | run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    ranked = sorted(
        results,
        key=lambda result: (
            split_metric(result.metrics_by_split, "test", "macro_f1") is None,
            -(split_metric(result.metrics_by_split, "test", "macro_f1") or -1.0),
            -(split_metric(result.metrics_by_split, "val", "macro_f1") or -1.0),
        ),
    )
    for result in ranked:
        metrics = result.metrics_by_split
        train_n = "-" if result.train_samples is None else str(result.train_samples)
        lines.append(
            f"| {_arch(result)} | {_pct(result)} | {train_n} | "
            f"{_fmt(split_metric(metrics, 'test', 'accuracy'))} | "
            f"{_fmt(split_metric(metrics, 'test', 'balanced_accuracy'))} | "
            f"**{_fmt(split_metric(metrics, 'test', 'macro_f1'))}** | "
            f"{_fmt(split_metric(metrics, 'val', 'accuracy'))} | "
            f"{_fmt(split_metric(metrics, 'val', 'balanced_accuracy'))} | "
            f"**{_fmt(split_metric(metrics, 'val', 'macro_f1'))}** | "
            f"{_fmt_delta(_gap(metrics, 'train', 'test'))} | "
            f"{_fmt_delta(_gap(metrics, 'train', 'val'))} | "
            f"`{result.slug}` |"
        )

    lines.extend(["", "### Leaders", ""])
    for split_name in PRIMARY_SPLITS:
        for key in ("macro_f1", "balanced_accuracy", "accuracy"):
            line = _leader_line(results, split_name=split_name, key=key)
            if line is not None:
                lines.append(line)
    gap_scored = [
        (result, _gap(result.metrics_by_split, "train", "test"))
        for result in results
    ]
    gap_scored = [(result, value) for result, value in gap_scored if value is not None]
    if gap_scored:
        best_result, best_gap = min(gap_scored, key=lambda item: item[1])
        lines.append(
            f"- **Lowest train→test acc gap:** `{_arch(best_result)}` @ `{_pct(best_result)}` "
            f"→ **{_fmt_delta(best_gap)}** (`{best_result.slug}`)"
        )
    lines.append("")
    return lines


def _write_per_class_section(results: list[ExperimentValidationResult]) -> list[str]:
    lines = [
        "## Per-class robustness (best.pt)",
        "",
        "Worst-class recall (support > 0). Architectures can hide minority-class failure "
        "behind high overall accuracy.",
        "",
        f"| architecture | train % "
        f"| {format_split_display('test')} worst class | recall | support "
        f"| {format_split_display('val')} worst class | recall | support | run |",
        "|---|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for result in _sorted_results(results):
        metrics = result.metrics_by_split
        idx_to_label = result.idx_to_label
        test_worst = _worst_class(metrics, "test", idx_to_label)
        val_worst = _worst_class(metrics, "val", idx_to_label)
        lines.append(
            f"| {_arch(result)} | {_pct(result)} | "
            f"{'-' if test_worst is None else test_worst['label']} | "
            f"{_fmt(None if test_worst is None else test_worst['recall'])} | "
            f"{'-' if test_worst is None else test_worst['support']} | "
            f"{'-' if val_worst is None else val_worst['label']} | "
            f"{_fmt(None if val_worst is None else val_worst['recall'])} | "
            f"{'-' if val_worst is None else val_worst['support']} | "
            f"`{result.slug}` |"
        )
    lines.append("")
    return lines


def _write_session_section(results: list[ExperimentValidationResult]) -> list[str]:
    lines = [
        "## Session stability (best.pt)",
        "",
        "Worst-session accuracy and best−worst spread. High spread ⇒ brittle to "
        "session-specific noise (placement, impedance, etc.).",
        "",
        f"| architecture | train % "
        f"| {format_split_display('test')} worst session | worst acc | best−worst "
        f"| {format_split_display('val')} worst session | worst acc | best−worst | run |",
        "|---|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for result in _sorted_results(results):
        metrics = result.metrics_by_split
        test_stats = _session_stats(metrics, "test")
        val_stats = _session_stats(metrics, "val")
        lines.append(
            f"| {_arch(result)} | {_pct(result)} | "
            f"{'-' if test_stats is None else format_session_display(test_stats['worst_session'])} | "
            f"{_fmt(None if test_stats is None else test_stats['worst_acc'])} | "
            f"{_fmt(None if test_stats is None else test_stats['spread'])} | "
            f"{'-' if val_stats is None else format_session_display(val_stats['worst_session'])} | "
            f"{_fmt(None if val_stats is None else val_stats['worst_acc'])} | "
            f"{_fmt(None if val_stats is None else val_stats['spread'])} | "
            f"`{result.slug}` |"
        )
    lines.append("")
    return lines


def _write_best_vs_last_section(results: list[ExperimentValidationResult]) -> list[str]:
    comparable = [
        result for result in results
        if "best" in result.metrics_by_kind and "last" in result.metrics_by_kind
    ]
    lines = [
        "## Best vs last checkpoint",
        "",
        "Delta = `last − best`. Positive accuracy/bal_acc/macro_f1 means last improved. "
        "Divergence by split is the overfit signal (e.g. last wins train/val but loses test).",
        "",
    ]
    if not comparable:
        lines.extend([
            "_No experiment had both `best.pt` and `last.pt` evaluated._",
            "",
        ])
        return lines

    lines.extend([
        "| architecture | train % | split | best acc | last acc | Δacc "
        "| best bal_acc | last bal_acc | Δbal_acc "
        "| best macro_f1 | last macro_f1 | Δmacro_f1 | run |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for result in _sorted_results(comparable):
        best = result.metrics_by_kind["best"]
        last = result.metrics_by_kind["last"]
        split_names = [
            name for name in ("train", "val", "test")
            if name in best and name in last
        ]
        for split_name in split_names:
            cells: list[str] = []
            for key in PRIMARY_METRICS:
                best_value = split_metric(best, split_name, key)
                last_value = split_metric(last, split_name, key)
                delta = (
                    None if best_value is None or last_value is None
                    else last_value - best_value
                )
                cells.extend([_fmt(best_value), _fmt(last_value), _fmt_delta(delta)])
            lines.append(
                f"| {_arch(result)} | {_pct(result)} | {format_split_display(split_name)} | "
                f"{' | '.join(cells)} | `{result.slug}` |"
            )
    lines.append("")
    return lines


def _write_concatenated_reports(results: list[ExperimentValidationResult]) -> list[str]:
    lines = [
        "## Per-experiment reports",
        "",
        "Full `validation_report.md` for each experiment (same options as this meta run).",
        "",
    ]
    for result in _sorted_results(results):
        lines.extend([
            f"### `{result.slug}`",
            "",
            f"- **Run directory:** `{result.run_dir.resolve()}`",
            f"- **Architecture:** `{_arch(result)}`",
            f"- **Train fraction:** `{_pct(result)}`",
        ])
        if result.report_path is not None and result.report_path.exists():
            lines.append(f"- **Report:** `{result.report_path.resolve()}`")
            lines.append("")
            report_text = result.report_path.read_text(encoding="utf-8").strip()
            # Avoid competing top-level H1 inside the concatenated doc.
            if report_text.startswith("# "):
                report_text = "#" + report_text
            lines.append(report_text)
        else:
            lines.extend(["", "_No validation_report.md was written for this experiment._"])
        lines.extend(["", "---", ""])
    return lines


def write_validation_comparison_report(
    meta_dir: Path,
    results: list[ExperimentValidationResult],
    *,
    options: ValidationOptions,
) -> Path:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Meta validation comparison",
        "",
        f"- **Generated:** {generated}",
        f"- **Meta directory:** `{meta_dir.resolve()}`",
        f"- **Experiments validated:** {len(results)}",
        *_options_summary_lines(options),
        "",
    ]

    if not results:
        lines.extend([
            "_No experiments produced aggregatable metrics._",
            "",
        ])
    else:
        lines.extend(_write_primary_metrics_section(results))
        lines.extend(_write_per_class_section(results))
        lines.extend(_write_session_section(results))
        if options.compare_best_vs_last:
            lines.extend(_write_best_vs_last_section(results))
        lines.extend(_write_concatenated_reports(results))

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

    panels = (
        ("test", "macro_f1", f"{format_split_display('test')} macro_f1 (best.pt)"),
        ("val", "macro_f1", f"{format_split_display('val')} macro_f1 (best.pt)"),
        ("test", "accuracy", f"{format_split_display('test')} accuracy (best.pt)"),
        ("val", "accuracy", f"{format_split_display('val')} accuracy (best.pt)"),
    )
    n_arch = len({result.architecture or result.slug for result in results})
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(max(10, 3.6 * 2), max(6.0, 0.9 * n_arch * 2)),
    )
    axes_flat = axes.ravel()

    for ax, (split_name, metric_key, title) in zip(axes_flat, panels, strict=True):
        architectures, fractions, matrix = _metric_matrix(
            results,
            split_name=split_name,
            metric_key=metric_key,
        )
        if not architectures or not fractions:
            ax.set_axis_off()
            ax.set_title(f"{title} (no grid)")
            continue
        im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(len(fractions)))
        ax.set_xticklabels([f"{int(round(f * 100))}%" for f in fractions])
        ax.set_yticks(range(len(architectures)))
        ax.set_yticklabels(architectures)
        ax.set_xlabel("training data fraction")
        ax.set_ylabel("architecture")
        ax.set_title(title)
        for i in range(len(architectures)):
            for j in range(len(fractions)):
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
    options: ValidationOptions,
) -> Path:
    payload = {
        "seed": seed,
        "generated_utc": datetime.now(UTC).isoformat(),
        "options": {
            "compare_best_vs_last": options.compare_best_vs_last,
            "checkpoints": list(options.resolved_checkpoints()),
            "splits": list(options.resolved_splits()),
            "run_embeddings": options.run_embeddings,
            "embedding_splits": list(options.embedding_splits),
            "show_split_per_class": options.show_split_per_class,
            "show_split_sessions": options.show_split_sessions,
            "show_summary_per_class": options.show_summary_per_class,
            "show_summary_sessions": options.show_summary_sessions,
        },
        "experiments": [],
    }

    for result in results:
        experiment = {
            "slug": result.slug,
            "architecture": result.architecture,
            "train_fraction": result.train_fraction,
            "train_samples": result.train_samples,
            "run_dir": str(result.run_dir.resolve()),
            "report_path": (
                None if result.report_path is None
                else str(result.report_path.resolve())
            ),
            "checkpoints": {},
        }
        for kind, metrics_by_split in result.metrics_by_kind.items():
            meta = result.ckpt_meta_by_kind.get(kind, {})
            experiment["checkpoints"][kind] = {
                "epoch": meta.get("epoch"),
                "val_acc_at_save": meta.get("val_acc"),
                "best_acc_at_train": meta.get("best_acc"),
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
                    for split_name, row in metrics_by_split.items()
                },
                "worst_class": {
                    split_name: _worst_class(metrics_by_split, split_name, result.idx_to_label)
                    for split_name in PRIMARY_SPLITS
                    if split_name in metrics_by_split
                },
                "session_stats": {
                    split_name: _session_stats(metrics_by_split, split_name)
                    for split_name in PRIMARY_SPLITS
                    if split_name in metrics_by_split
                },
                "gaps": {
                    "train_to_test_acc": _gap(metrics_by_split, "train", "test"),
                    "train_to_val_acc": _gap(metrics_by_split, "train", "val"),
                },
            }
        payload["experiments"].append(experiment)

    path = meta_dir / VALIDATION_MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved validation manifest -> {path}")
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--force-revalidate",
        action="store_true",
        help="re-run val.py even when validation_report.md / validation_metrics.json exist",
    )
    add_validation_option_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    options = validation_options_from_args(args)

    meta_dir = resolve_meta_dir(args.meta_dir)
    splits = load_dataset_splits(SPLITS_OUTPUT_DIR)
    experiment_dirs = discover_experiment_dirs(meta_dir)

    print(f"checkpoints: {CHECKPOINT_DIR.resolve()}")
    print(f"splits: {SPLITS_OUTPUT_DIR.resolve()}")
    print(f"meta dir: {meta_dir.resolve()}")
    print(f"experiments: {len(experiment_dirs)}")
    print(
        f"options: compare_best_vs_last={options.compare_best_vs_last} "
        f"splits={list(options.resolved_splits())} embeddings={options.run_embeddings}"
    )

    results = run_meta_validation(
        meta_dir=meta_dir,
        splits=splits,
        device=device,
        options=options,
        use_color=not args.no_color,
        seed=args.seed,
        force_revalidate=args.force_revalidate,
    )

    save_validation_manifest(meta_dir, results, seed=args.seed, options=options)
    write_validation_comparison_report(meta_dir, results, options=options)
    plot_validation_comparison(meta_dir, results)

    print("")
    print(f"meta validation complete — artifacts under {meta_dir.resolve()}")
    print(f"  {VALIDATION_COMPARISON_NAME}")
    print(f"  {VALIDATION_COMPARISON_PLOT_NAME}")
    print(f"  {VALIDATION_MANIFEST_NAME}")


if __name__ == "__main__":
    main()
