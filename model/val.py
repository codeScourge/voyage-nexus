from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from collections import defaultdict

from data import load_dataset_splits
from train import (
    CHECKPOINT_DIR,
    FusionDataset,
    IntermediateFusionEEGNet,
    _per_label_colors,
    get_device,
    latest_run_dir,
    seed_everything,
    soft_cross_entropy,
)

# --- embeddings / UMAP (always runs after metrics)
EMBEDDING_SPLITS = ("val", "test")
EMBEDDING_MAX_PER_LABEL = 200
EMBEDDINGS_N_NEIGHBORS = 15
EMBEDDINGS_MIN_DIST = 0.1
EMBEDDINGS_OUTPUT_NAME = "embeddings_umap.png"


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
    f2 = int(state_dict["bn3.weight"].shape[0])

    model = IntermediateFusionEEGNet(
        n_eeg=model_config["n_eeg"],
        n_emg=model_config["n_emg"],
        n_classes=model_config["n_classes"],
        T=model_config["T"],
        F2=f2,
    )
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
    }
    return model, label_to_idx, meta


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

    sample_offset = 0
    for eeg, emg, y_soft, y_hard in tqdm(loader, desc=split_name, leave=False):
        eeg, emg = eeg.to(device), emg.to(device)
        y_soft = y_soft.to(device)
        y_hard = y_hard.to(device)
        logits = model(eeg, emg)
        running_loss += soft_cross_entropy(logits, y_soft).item() * y_hard.size(0)
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
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
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
    }


def print_metrics(
    metrics: dict,
    *,
    idx_to_label: dict[int, str],
    use_color: bool = True,
) -> None:
    print(f"\n=== {metrics['split']} ===")
    print(f"samples:           {metrics['n_samples']}")
    print(f"loss:              {metrics['loss']:.4f}")
    print(f"accuracy:          {metrics['accuracy']:.4f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"macro_f1:          {metrics['macro_f1']:.4f}")
    print(f"weighted_f1:       {metrics['weighted_f1']:.4f}")

    print("\nper-class:")
    if use_color:
        print("  (cell color: metric value — red=low, green=high)")
    print(f"{'label':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10} {'top confusion':>28}")
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
    min_samples = metrics.get("session_min_samples", 1)
    print(f"\nper-session (>= {min_samples} samples, split accuracy={overall_acc:.4f}):")

    def print_rows(title: str, rows: list[dict]) -> None:
        print(f"\n{title}:")
        print(
            f"{'session':<14} {'accuracy':>10} {'delta':>10} {'correct':>10} "
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
            session = format_session_display(row["session"])
            print(
                f"{session:<14} "
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
            f"{'split':<8} {'session':<14} {'accuracy':>10} "
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
            session = format_session_display(row["session"])
            print(
                f"{row['split']:<8} "
                f"{session:<14} "
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


def resolve_run_dir(checkpoint: Path | None, *, root: Path = CHECKPOINT_DIR) -> Path:
    if checkpoint is not None:
        return checkpoint.resolve().parent
    run_dir = latest_run_dir(root)
    if run_dir is None:
        raise FileNotFoundError(f"No run directories found under {root}")
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
    header = f"{'split':<8}" + "".join(f"{label:>12}" for label, _, _ in COMPARE_METRICS) + f"{'winner':>10}"
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
        print(f"{split_name:<8}{''.join(deltas)}{str(winner_summary):>10}")

    val_test_splits = [name for name in ("val", "test") if name in split_names]
    if val_test_splits:
        n_classes = len(next(iter(best_by_split.values()))["per_class"])
        print("\nper-class recall delta (last - best):")
        print(f"{'label':<20}" + "".join(f"{split + ' Δ':>12}" for split in val_test_splits))
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
                    cells.append("     n/a".rjust(12))
                    continue
                delta = last_recall - best_recall
                cells.append(_delta_text(delta, higher_is_better=True, width=12))
                winner = _winner(best_recall, last_recall, higher_is_better=True)
                if winner == "best":
                    best_recall_wins += 1
                elif winner == "last":
                    last_recall_wins += 1
                else:
                    tie_recall_wins += 1
            print(f"{label[:20]:<20}{''.join(cells)}")
        print(
            f"\nper-class recall head-to-head (val+test, support>0): "
            f"best={best_recall_wins}, last={last_recall_wins}, tie={tie_recall_wins}"
        )

    print("\nsession accuracy (last - best, shared sessions only):")
    for split_name in val_test_splits:
        best_sessions = {
            row["session_dir"]: row for row in best_by_split[split_name]["per_session"]
        }
        last_sessions = {
            row["session_dir"]: row for row in last_by_split[split_name]["per_session"]
        }
        common = sorted(set(best_sessions) & set(last_sessions))
        if not common:
            print(f"  {split_name}: no shared ranked sessions")
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
            f"  {split_name}: n={len(common)}, mean Δacc={mean_delta:+.4f}, "
            f"last wins={last_wins}, best wins={best_wins}, tie={ties}"
        )


def print_summary_table(
    metrics_list: list[dict],
    *,
    idx_to_label: dict[int, str],
    use_color: bool = True,
    session_top_k: int = 7,
) -> None:
    print("\n=== summary ===")
    print(
        f"{'split':<8} {'samples':>8} {'loss':>10} "
        f"{'accuracy':>10} {'bal_acc':>10} {'macro_f1':>10} {'weighted_f1':>12}"
    )
    for metrics in metrics_list:
        print(
            f"{metrics['split']:<8} "
            f"{metrics['n_samples']:>8} "
            f"{metrics['loss']:>10.4f} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['balanced_accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['weighted_f1']:>12.4f}"
        )

    by_split = {metrics["split"]: metrics for metrics in metrics_list}
    split_names = [name for name in ("train", "val", "test") if name in by_split]
    if not split_names:
        return

    n_classes = len(next(iter(by_split.values()))["per_class"])
    confusion_width = 22
    print("\nper-class recall and top confusion (train vs val vs test):")
    if use_color:
        print(
            "  (recall: red=low, green=high, bold=worst split; "
            "top confusion: pred label, count, fraction of class support)"
        )
    header = (
        f"{'label':<20}"
        + "".join(f"{name:>10}" for name in split_names)
        + "".join(f"{name + ' confusion':>{confusion_width}}" for name in split_names)
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
    batch_size: int,
    session_min_samples: int,
    session_top_k: int,
    use_color: bool,
    seed: int,
) -> tuple[list[dict], dict[str, int], dict]:
    model, label_to_idx, ckpt_meta = load_checkpoint(checkpoint_path, device)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    print_checkpoint_banner(kind, checkpoint_path, ckpt_meta)

    split_specs = (
        ("train", splits.train.indices),
        ("val", splits.val.indices),
        ("test", splits.test.indices),
    )

    all_metrics: list[dict] = []
    for split_name, indices in split_specs:
        dataset = FusionDataset(splits.dataset, indices, label_to_idx)
        metrics = evaluate_split(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
            split_name=split_name,
            session_min_samples=session_min_samples,
        )
        all_metrics.append(metrics)
        print_metrics(metrics, idx_to_label=idx_to_label, use_color=use_color)
        print_session_rankings(
            metrics,
            idx_to_label=idx_to_label,
            top_k=session_top_k,
        )

    print_summary_table(
        all_metrics,
        idx_to_label=idx_to_label,
        use_color=use_color,
        session_top_k=session_top_k,
    )

    embedding_stem = EMBEDDINGS_OUTPUT_NAME.rsplit(".", 1)[0]
    embedding_suffix = EMBEDDINGS_OUTPUT_NAME.rsplit(".", 1)[-1]
    embedding_output = checkpoint_path.parent / f"{embedding_stem}_{kind}.{embedding_suffix}"
    run_embedding_umap(
        model,
        splits,
        label_to_idx,
        device=device,
        batch_size=batch_size,
        embedding_splits=EMBEDDING_SPLITS,
        max_per_label=EMBEDDING_MAX_PER_LABEL,
        output_path=embedding_output,
        seed=seed,
        n_neighbors=EMBEDDINGS_N_NEIGHBORS,
        min_dist=EMBEDDINGS_MIN_DIST,
    )

    return all_metrics, label_to_idx, ckpt_meta


EMBEDDING_TAPS: dict[str, str] = {
    "classifier_input": "post-ELU / post-pool2 (classifier input)",
    "pre_pool2": "bn3 ELU (pre-pool2)",
}

SPLIT_MARKERS = {
    "val": "o",
    "test": "s",
}


@dataclass(frozen=True, slots=True)
class EmbeddingSample:
    pre_pool2: np.ndarray
    classifier_input: np.ndarray
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    per_label: dict[str, list[EmbeddingSample]] = defaultdict(list)
    sample_offset = 0

    for eeg, emg, _y_soft, _y_hard in tqdm(loader, desc=f"embeddings:{split_name}", leave=False):
        eeg, emg = eeg.to(device), emg.to(device)
        taps = model.forward_embeddings(eeg, emg)

        pre_pool2 = taps["pre_pool2"].cpu().numpy()
        classifier_input = taps["classifier_input"].cpu().numpy()
        batch_size_actual = pre_pool2.shape[0]

        for batch_idx in range(batch_size_actual):
            label = _sample_label(dataset, sample_offset + batch_idx)
            per_label[label].append(
                EmbeddingSample(
                    pre_pool2=pre_pool2[batch_idx],
                    classifier_input=classifier_input[batch_idx],
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
    matrix = np.stack([getattr(sample, tap) for sample in samples], axis=0)
    labels = [sample.label for sample in samples]
    splits = [sample.split for sample in samples]
    return matrix, labels, splits


def plot_embedding_umap(
    samples: list[EmbeddingSample],
    *,
    output_path: Path,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    import umap

    if not samples:
        print("embeddings: no samples collected; skipping UMAP plot")
        return

    labels_present = sorted({sample.label for sample in samples})
    label_colors = dict(zip(labels_present, _per_label_colors(len(labels_present)), strict=True))

    fig, axes = plt.subplots(1, len(EMBEDDING_TAPS), figsize=(7 * len(EMBEDDING_TAPS), 6))
    if len(EMBEDDING_TAPS) == 1:
        axes = [axes]

    for ax, (tap_key, tap_title) in zip(axes, EMBEDDING_TAPS.items(), strict=True):
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
                    label=f"{label} ({split_name})",
                )

        ax.set_title(tap_title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.25)

    split_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="gray",
            linestyle="None",
            markersize=7,
            label=f"{split_name} ({'dot' if marker == 'o' else 'square'})",
        )
        for split_name, marker in SPLIT_MARKERS.items()
        if any(sample.split == split_name for sample in samples)
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
    fig.suptitle("Fusion embeddings (dropout off)", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"embeddings: saved UMAP plot to {output_path.resolve()}")


def run_embedding_umap(
    model: nn.Module,
    splits,
    label_to_idx: dict[str, int],
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
    split_indices = {
        "val": splits.val.indices,
        "test": splits.test.indices,
    }
    rng = np.random.default_rng(seed)
    samples: list[EmbeddingSample] = []

    for split_name in embedding_splits:
        if split_name not in split_indices:
            raise ValueError(f"unknown embedding split '{split_name}' (choose val, test)")
        dataset = FusionDataset(splits.dataset, split_indices[split_name], label_to_idx)
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

    print(
        f"embeddings: collected {len(samples)} samples "
        f"from {', '.join(embedding_splits)} "
        f"(max {max_per_label} per label per split)"
    )
    plot_embedding_umap(
        samples,
        output_path=output_path,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate best.pt and last.pt from a training run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to any checkpoint in a run dir (default: latest run, evaluates best.pt and last.pt)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "splits",
        help="Directory with splits_manifest.json and splits_windows.npz",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--session-top-k",
        type=int,
        default=7,
        help="Number of best/worst sessions to print per split and in summary",
    )
    parser.add_argument(
        "--session-min-samples",
        type=int,
        default=3,
        help="Minimum samples required to include a session in rankings",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in printed tables",
    )
    args = parser.parse_args()

    use_color = _use_color(force=not args.no_color)

    seed_everything(args.seed)
    device = get_device()
    print(f"device: {device}")

    run_dir = resolve_run_dir(args.checkpoint)
    ckpt_paths = checkpoint_paths_for_run(run_dir)
    print(f"run_dir: {run_dir.resolve()}")

    splits = load_dataset_splits(args.splits_dir)

    evaluated: dict[str, tuple[list[dict], dict[str, int], dict]] = {}
    for kind in ("best", "last"):
        if kind not in ckpt_paths:
            print(f"\nwarning: {kind}.pt not found in {run_dir}, skipping")
            continue

        checkpoint_path = ckpt_paths[kind]
        if (
            kind == "last"
            and "best" in evaluated
            and checkpoint_path.resolve() == ckpt_paths["best"].resolve()
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
            batch_size=args.batch_size,
            session_min_samples=args.session_min_samples,
            session_top_k=args.session_top_k,
            use_color=use_color,
            seed=args.seed,
        )

    if "best" in evaluated and "last" in evaluated:
        best_metrics, _best_label_to_idx, best_meta = evaluated["best"]
        last_metrics, _last_label_to_idx, last_meta = evaluated["last"]
        idx_to_label = {idx: label for label, idx in _best_label_to_idx.items()}
        print_model_comparison_meta(
            best_metrics,
            last_metrics,
            best_meta=best_meta,
            last_meta=last_meta,
            idx_to_label=idx_to_label,
        )
    elif len(evaluated) == 1:
        only_kind = next(iter(evaluated))
        print(f"\n(note: only {only_kind}.pt was evaluated; need both best and last for comparison)")


if __name__ == "__main__":
    main()
