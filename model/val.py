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
    FusionDataset,
    IntermediateFusionEEGNet,
    _per_label_colors,
    default_checkpoint_path,
    get_device,
    seed_everything,
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


def compute_session_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    session_dirs: list[str],
    *,
    min_samples: int = 1,
) -> list[dict]:
    by_session: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for t, p, session_dir in zip(y_true, y_pred, session_dirs, strict=True):
        row = by_session[session_dir]
        row["total"] += 1
        if t == p:
            row["correct"] += 1

    rows: list[dict] = []
    for session_dir, counts in by_session.items():
        total = counts["total"]
        if total < min_samples:
            continue
        correct = counts["correct"]
        rows.append(
            {
                "session": session_name(session_dir),
                "session_dir": session_dir,
                "n_samples": total,
                "n_correct": correct,
                "accuracy": correct / total if total else 0.0,
            }
        )
    rows.sort(key=lambda row: (row["accuracy"], row["n_samples"]), reverse=True)
    return rows


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

    crit = nn.CrossEntropyLoss()

    sample_offset = 0
    for eeg, emg, y in tqdm(loader, desc=split_name, leave=False):
        eeg, emg, y = eeg.to(device), emg.to(device), y.to(device)
        logits = model(eeg, emg)
        running_loss += crit(logits, y).item() * y.size(0)
        batch_preds = logits.argmax(1).cpu().tolist()
        batch_true = y.cpu().tolist()
        y_true.extend(batch_true)
        y_pred.extend(batch_preds)
        for batch_idx in range(len(batch_true)):
            base_idx = dataset.indices[sample_offset + batch_idx]
            session_dirs.append(str(dataset.base[base_idx]["session_dir"]))
        sample_offset += len(batch_true)
        total += y.size(0)

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
        print(f"{'session':<40} {'accuracy':>10} {'delta':>10} {'correct':>10} {'samples':>10}")
        for row in rows:
            delta = row["accuracy"] - overall_acc
            print(
                f"{row['session']:<40} "
                f"{row['accuracy']:>10.4f} "
                f"{delta:>+10.4f} "
                f"{row['n_correct']:>10d} "
                f"{row['n_samples']:>10d}"
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


def print_summary_table(
    metrics_list: list[dict],
    *,
    idx_to_label: dict[int, str],
    use_color: bool = True,
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
        + "".join(f"{name + ' conf':>{confusion_width}}" for name in split_names)
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

    for eeg, emg, _y in tqdm(loader, desc=f"embeddings:{split_name}", leave=False):
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
    parser = argparse.ArgumentParser(description="Evaluate a saved fusion EEGNet checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint .pt file (default: best.pt from latest run)",
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
        default=5,
        help="Number of best/worst sessions to print per split",
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
    checkpoint_path = args.checkpoint or default_checkpoint_path("best")
    print(f"device: {device}")

    model, label_to_idx, ckpt_meta = load_checkpoint(checkpoint_path, device)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(
        f"kind={ckpt_meta['kind']}, epoch={ckpt_meta['epoch']}/{ckpt_meta['epochs']}, "
        f"val_acc={ckpt_meta['val_acc']:.4f}, best_acc@train-time={ckpt_meta['best_acc']:.4f}"
    )

    splits = load_dataset_splits(args.splits_dir)
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
            batch_size=args.batch_size,
            split_name=split_name,
            session_min_samples=args.session_min_samples,
        )
        all_metrics.append(metrics)
        print_metrics(metrics, idx_to_label=idx_to_label, use_color=use_color)
        print_session_rankings(metrics, top_k=args.session_top_k)

    print_summary_table(all_metrics, idx_to_label=idx_to_label, use_color=use_color)

    run_embedding_umap(
        model,
        splits,
        label_to_idx,
        device=device,
        batch_size=args.batch_size,
        embedding_splits=EMBEDDING_SPLITS,
        max_per_label=EMBEDDING_MAX_PER_LABEL,
        output_path=checkpoint_path.parent / EMBEDDINGS_OUTPUT_NAME,
        seed=args.seed,
        n_neighbors=EMBEDDINGS_N_NEIGHBORS,
        min_dist=EMBEDDINGS_MIN_DIST,
    )


if __name__ == "__main__":
    main()
