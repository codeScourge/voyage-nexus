from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_dataset_splits
from train import (
    FusionDataset,
    IntermediateFusionEEGNet,
    default_checkpoint_path,
    get_device,
    seed_everything,
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]
    label_to_idx: dict[str, int] = ckpt["label_to_idx"]

    model = IntermediateFusionEEGNet(
        n_eeg=model_config["n_eeg"],
        n_emg=model_config["n_emg"],
        n_classes=model_config["n_classes"],
        T=model_config["T"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
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
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_classes = len(dataset.label_to_idx)

    y_true: list[int] = []
    y_pred: list[int] = []
    running_loss = 0.0
    total = 0

    crit = nn.CrossEntropyLoss()

    for eeg, emg, y in tqdm(loader, desc=split_name, leave=False):
        eeg, emg, y = eeg.to(device), emg.to(device), y.to(device)
        logits = model(eeg, emg)
        running_loss += crit(logits, y).item() * y.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
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
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
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
    }


def print_metrics(
    metrics: dict,
    *,
    idx_to_label: dict[int, str],
) -> None:
    print(f"\n=== {metrics['split']} ===")
    print(f"samples:           {metrics['n_samples']}")
    print(f"loss:              {metrics['loss']:.4f}")
    print(f"accuracy:          {metrics['accuracy']:.4f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"macro_f1:          {metrics['macro_f1']:.4f}")
    print(f"weighted_f1:       {metrics['weighted_f1']:.4f}")

    print("\nper-class:")
    print(f"{'label':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for class_idx, row in enumerate(metrics["per_class"]):
        label = idx_to_label.get(class_idx, str(class_idx))
        print(
            f"{label:<20} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} "
            f"{row['support']:>10d}"
        )

    print("\nconfusion matrix (rows=true, cols=pred):")
    labels = [idx_to_label.get(i, str(i)) for i in range(len(metrics["per_class"]))]
    header = "true\\pred".ljust(20) + "".join(label[:12].rjust(12) for label in labels)
    print(header)
    for class_idx, row in enumerate(metrics["confusion_matrix"]):
        label = idx_to_label.get(class_idx, str(class_idx))
        counts = "".join(str(int(v)).rjust(12) for v in row)
        print(f"{label[:20]:<20}{counts}")


def print_summary_table(metrics_list: list[dict]) -> None:
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
    args = parser.parse_args()

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
        )
        all_metrics.append(metrics)
        print_metrics(metrics, idx_to_label=idx_to_label)

    print_summary_table(all_metrics)


if __name__ == "__main__":
    main()
