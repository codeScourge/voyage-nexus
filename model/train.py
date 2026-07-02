from __future__ import annotations

import argparse
import os
import random
import re
import time
import uuid
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ARCHITECTURES, build_fusion_model
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import default_label_to_idx, label_probs_to_vector, load_dataset_splits

# ---
SEED = 56 # 42 always
TORCH_DETERMINISTIC = False
RUN_DIR_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

# --- modality selection (set True to drop an entire modality from training)
NOT_USE_EEG = True
NOT_USE_EMG = False

if NOT_USE_EEG and NOT_USE_EMG:
    raise ValueError("At least one of EEG or EMG must be enabled")

# --- channel selection (set False to exclude from training)
# EEG1 = 0
# EEG2 = 1
# EEG3 = 0
# EEG4 = 1
# EEG5 = 1
# EEG6 = 0
# EEG7 = 1
# EEG8 = 1
# EEG9 = 1
# EEG10 = 1
# EEG11 = 0
# EEG12 = 1
# EEG13 = 0
# EEG14 = 0
# EEG15 = 0
# EEG16 = 0

# EMG1 = 1
# EMG2 = 0
# EMG3 = 1
# EMG4 = 1
# EMG5 = 0
# EMG6 = 1
# EMG7 = 0
# EMG8 = 1
# EMG9 = 0
# EMG10 = 1
# EMG11 = 0
# EMG12 = 0
# EMG13 = 0
# EMG14 = 1
# EMG15 = 0
# EMG16 = 1

EEG1 = 1
EEG2 = 1
EEG3 = 1
EEG4 = 1
EEG5 = 1
EEG6 = 1
EEG7 = 1
EEG8 = 1
EEG9 = 1
EEG10 = 1
EEG11 = 1
EEG12 = 1
EEG13 = 1
EEG14 = 1
EEG15 = 1
EEG16 = 1

EMG1 = 1
EMG2 = 1
EMG3 = 1
EMG4 = 1
EMG5 = 1
EMG6 = 1
EMG7 = 1
EMG8 = 1
EMG9 = 1
EMG10 = 1
EMG11 = 1
EMG12 = 1
EMG13 = 1
EMG14 = 1
EMG15 = 1
EMG16 = 1

_EEG_CHANNEL_USE = (
    EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8,
    EEG9, EEG10, EEG11, EEG12, EEG13, EEG14, EEG15, EEG16,
)
_EMG_CHANNEL_USE = (
    EMG1, EMG2, EMG3, EMG4, EMG5, EMG6, EMG7, EMG8,
    EMG9, EMG10, EMG11, EMG12, EMG13, EMG14, EMG15, EMG16,
)
ACTIVE_EEG_INDICES = [] if NOT_USE_EEG else [i for i, use in enumerate(_EEG_CHANNEL_USE) if use]
ACTIVE_EMG_INDICES = [] if NOT_USE_EMG else [16 + i for i, use in enumerate(_EMG_CHANNEL_USE) if use]

if not NOT_USE_EEG and not ACTIVE_EEG_INDICES:
    raise ValueError("At least one EEG channel must be enabled when EEG is used")
if not NOT_USE_EMG and not ACTIVE_EMG_INDICES:
    raise ValueError("At least one EMG channel must be enabled when EMG is used")


def seed_everything(seed: int = 0, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- setup
device = get_device()
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_SAVE_INTERVAL = 5
EARLY_STOPPING_METRIC = "loss"  # "loss" or "acc" — stop signal only, not best.pt selection
EARLY_STOPPING_SMOOTH_WINDOW = 4
MODEL_ARCHITECTURE = "intermediate_fusion_eegnet"  # or "cat_net"


def use_eeg_from_config(model_config: dict) -> bool:
    return not model_config.get("not_use_eeg", False)


def use_emg_from_config(model_config: dict) -> bool:
    return not model_config.get("not_use_emg", False)


def fusion_dataset_kwargs(model_config: dict | None = None) -> dict:
    if model_config is None:
        return {}
    return {
        "not_use_eeg": model_config.get("not_use_eeg", False),
        "not_use_emg": model_config.get("not_use_emg", False),
    }


def build_model_from_config(
    model_config: dict,
    *,
    state_dict: dict | None = None,
) -> nn.Module:
    use_eeg = use_eeg_from_config(model_config)
    use_emg = use_emg_from_config(model_config)
    active_eeg = model_config.get("active_eeg_indices", [])
    active_emg = model_config.get("active_emg_indices", [])
    n_eeg = len(active_eeg) if use_eeg else 0
    n_emg = len(active_emg) if use_emg else 0
    if use_eeg and n_eeg == 0:
        n_eeg = int(model_config["n_eeg"])
    if use_emg and n_emg == 0:
        n_emg = int(model_config["n_emg"])
    return build_fusion_model(
        model_config.get("architecture", MODEL_ARCHITECTURE),
        n_eeg=n_eeg,
        n_emg=n_emg,
        n_classes=model_config["n_classes"],
        T=model_config["T"],
        use_eeg=use_eeg,
        use_emg=use_emg,
        state_dict=state_dict,
    )


# --- dataset adapter
class FusionDataset(torch.utils.data.Dataset):
    """Wraps the split so __getitem__ returns (eeg, emg, label) batched-ready."""

    def __init__(
        self,
        base_dataset,
        indices,
        label_to_idx,
        *,
        not_use_eeg: bool | None = None,
        not_use_emg: bool | None = None,
    ):
        self.base = base_dataset
        self.indices = list(indices)
        self.label_to_idx = label_to_idx
        self.not_use_eeg = NOT_USE_EEG if not_use_eeg is None else not_use_eeg
        self.not_use_emg = NOT_USE_EMG if not_use_emg is None else not_use_emg

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        sample = self.base[self.indices[i]]
        x = sample["x"]
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

        if self.not_use_eeg:
            eeg = torch.empty(1, 0, x.shape[0], dtype=x.dtype)
        else:
            eeg = x[:, ACTIVE_EEG_INDICES].T.unsqueeze(0)
        if self.not_use_emg:
            emg = torch.empty(1, 0, x.shape[0], dtype=x.dtype)
        else:
            emg = x[:, ACTIVE_EMG_INDICES].T.unsqueeze(0)
        y_soft = torch.from_numpy(
            label_probs_to_vector(
                sample.get("label_probs"),
                sample["label"],
                self.label_to_idx,
            )
        )
        y_hard = torch.tensor(self.label_to_idx[sample["label"]], dtype=torch.long)
        return eeg, emg, y_soft, y_hard


def build_label_map(base_dataset, indices) -> dict:
    label_to_idx = default_label_to_idx()
    seen = {base_dataset[i]["label"] for i in indices}
    unknown = seen - set(label_to_idx)
    if unknown:
        raise ValueError(f"Unknown labels in training data: {sorted(unknown)}")
    return label_to_idx


# --- train
def construct_model(splits, architecture: str = MODEL_ARCHITECTURE):
    label_to_idx = build_label_map(splits.dataset, splits.train.indices)
    n_classes = len(label_to_idx)
    print(f"classes ({n_classes}): {', '.join(label_to_idx)}")

    # infer shapes from one sample
    eeg0, emg0, _, _ = FusionDataset(splits.dataset, splits.train.indices, label_to_idx)[0]
    use_eeg = not NOT_USE_EEG
    use_emg = not NOT_USE_EMG
    n_eeg = len(ACTIVE_EEG_INDICES) if use_eeg else 0
    n_emg = len(ACTIVE_EMG_INDICES) if use_emg else 0
    T = eeg0.shape[2] if use_eeg else emg0.shape[2]

    print("\n\n")
    print("--- example sample ---")
    if use_eeg:
        print("eeg: ", eeg0)
    else:
        print("eeg: disabled")
    print("\n")
    if use_emg:
        print("emg: ", emg0)
    else:
        print("emg: disabled")
    print("------")
    print("\n\n")

    model = build_fusion_model(
        architecture,
        n_eeg=n_eeg,
        n_emg=n_emg,
        n_classes=n_classes,
        T=T,
        use_eeg=use_eeg,
        use_emg=use_emg,
    ).to(device)

    return model, label_to_idx, {
        "architecture": architecture,
        "n_eeg": n_eeg,
        "n_emg": n_emg,
        "n_classes": n_classes,
        "T": T,
        "active_eeg_indices": ACTIVE_EEG_INDICES,
        "active_emg_indices": ACTIVE_EMG_INDICES,
        "not_use_eeg": NOT_USE_EEG,
        "not_use_emg": NOT_USE_EMG,
    }

def new_run_dir_name(now: datetime | None = None) -> str:
    """e.g. 2026-06-20_01-44-45_run_a3f8b2c1 (mirrors client recording folders)."""
    when = now or datetime.now()
    uid = uuid.uuid4().hex[:8]
    stamp = when.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_run_{uid}"


def create_run_dir(root: Path = CHECKPOINT_DIR, *, continued: bool = False) -> Path:
    base_name = new_run_dir_name()
    if continued:
        base_name = f"{base_name}-continued"
    run_dir = root / base_name
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def latest_checkpoint_in_run(run_dir: Path) -> Path:
    for name in ("last.pt", "best.pt"):
        path = run_dir / name
        if path.exists():
            return path
    epoch_ckpts = sorted(run_dir.glob("epoch_*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def load_training_checkpoint(path: Path) -> tuple[nn.Module, dict, dict, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]
    label_to_idx: dict[str, int] = ckpt["label_to_idx"]

    architecture = model_config.get("architecture", "intermediate_fusion_eegnet")
    model = build_model_from_config(model_config, state_dict=ckpt["model_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])

    resume_meta = {
        "source_checkpoint": str(path),
        "source_run_dir": str(path.parent),
        "source_epoch": int(ckpt.get("epoch", 0)),
        "source_val_acc": float(ckpt.get("val_acc", float("nan"))),
        "source_kind": ckpt.get("kind", "unknown"),
    }
    return model, label_to_idx, model_config, resume_meta


def latest_run_dir(root: Path = CHECKPOINT_DIR) -> Path | None:
    runs = [p for p in root.iterdir() if p.is_dir() and RUN_DIR_NAME_RE.match(p.name)]
    if not runs:
        return None
    return max(runs, key=lambda p: p.name)


def default_checkpoint_path(kind: str = "best", root: Path = CHECKPOINT_DIR) -> Path:
    run_dir = latest_run_dir(root)
    if run_dir is not None:
        path = run_dir / f"{kind}.pt"
        if path.exists():
            return path
    legacy = root / "fusion_eegnet.pt"
    return legacy if legacy.exists() else (run_dir / f"{kind}.pt" if run_dir else legacy)


# Legacy alias for scripts that import CHECKPOINT_PATH.
CHECKPOINT_PATH = default_checkpoint_path("best")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    label_to_idx: dict,
    model_config: dict,
    *,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_acc: float,
    total_epochs: int,
    kind: str,
    state_dict: dict | None = None,
    continued_from: dict | None = None,
    log: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": state_dict if state_dict is not None else model.state_dict(),
        "label_to_idx": label_to_idx,
        "model_config": model_config,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "best_acc": best_acc,
        "epochs": total_epochs,
        "kind": kind,
        "device": str(device),
    }
    if continued_from is not None:
        payload["continued_from"] = continued_from
    torch.save(payload, path)

    if log:
        tqdm.write(f"saved checkpoint ({kind}, epoch {epoch}) -> {path}")


def format_epoch_summary(
    epoch_num: int,
    total_epochs: int,
    *,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    test_loss: float,
    test_acc: float,
    best_acc: float,
    best_epoch: int,
    epoch_time: float,
    epochs_without_improve: int | None = None,
    smoothed_stop: float | None = None,
    notes: list[str] | None = None,
) -> str:
    width = len(str(total_epochs))
    parts = [
        f"epoch {epoch_num:>{width}}/{total_epochs}",
        f"train_loss={train_loss:.4f}",
        f"train_acc={train_acc:.4f}",
        f"val_loss={val_loss:.4f}",
        f"val_acc={val_acc:.4f}",
        f"test_loss={test_loss:.4f}",
        f"test_acc={test_acc:.4f}",
        f"best={best_acc:.4f} (epoch={best_epoch})",
        f"{epoch_time:.1f}s",
    ]
    if epochs_without_improve is not None:
        parts.append(f"no_improve={epochs_without_improve}")
    if smoothed_stop is not None:
        parts.append(f"stop_smooth={smoothed_stop:.4f}")
    if notes:
        parts.extend(notes)
    return " | ".join(parts)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def format_flops(flops: int) -> str:
    for scale, unit in (
        (1e18, "EFLOPs"),
        (1e15, "PFLOPs"),
        (1e12, "TFLOPs"),
        (1e9, "GFLOPs"),
        (1e6, "MFLOPs"),
        (1e3, "kFLOPs"),
    ):
        if flops >= scale:
            return f"{flops / scale:.3f} {unit} ({flops:,} FLOPs)"
    return f"{flops:,} FLOPs"


def smoothed_tail(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def init_per_class_counters(n_classes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(n_classes, device=device),
        torch.zeros(n_classes, device=device),
        torch.zeros(n_classes, device=device),
    )


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unsupported reduction: {reduction!r}")


def update_per_class_counters(
    logits: torch.Tensor,
    y_hard: torch.Tensor,
    pred: torch.Tensor,
    loss_sum: torch.Tensor,
    correct: torch.Tensor,
    count: torch.Tensor,
) -> None:
    loss_per_sample = F.cross_entropy(logits, y_hard, reduction="none")
    ones = torch.ones_like(y_hard, dtype=loss_sum.dtype)
    count.scatter_add_(0, y_hard, ones)
    loss_sum.scatter_add_(0, y_hard, loss_per_sample)
    correct.scatter_add_(0, y_hard, (pred == y_hard).to(loss_sum.dtype))


def per_class_metrics_from_counters(
    loss_sum: torch.Tensor,
    correct: torch.Tensor,
    count: torch.Tensor,
    idx_to_label: dict[int, str],
) -> tuple[dict[str, float], dict[str, float]]:
    losses: dict[str, float] = {}
    accs: dict[str, float] = {}
    for idx, label in idx_to_label.items():
        n = count[idx].item()
        if n > 0:
            losses[label] = (loss_sum[idx] / n).item()
            accs[label] = (correct[idx] / n).item()
        else:
            losses[label] = float("nan")
            accs[label] = float("nan")
    return losses, accs


@torch.no_grad()
def evaluate_fusion_split(
    model,
    dataloader: DataLoader,
    dataset_size: int,
    *,
    n_classes: int,
    idx_to_label: dict[int, str],
    desc: str,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    model.eval()
    running = 0.0
    correct = total = 0
    loss_sum, correct_per_class, count_per_class = init_per_class_counters(n_classes, device)
    for eeg, emg, y_soft, y_hard in tqdm(dataloader, desc=desc, leave=False):
        eeg, emg = eeg.to(device), emg.to(device)
        y_soft = y_soft.to(device)
        y_hard = y_hard.to(device)
        logits = model(eeg, emg)
        running += soft_cross_entropy(logits, y_soft).item() * y_hard.size(0)
        pred = logits.argmax(1)
        correct += (pred == y_hard).sum().item()
        total += y_hard.size(0)
        update_per_class_counters(
            logits, y_hard, pred,
            loss_sum, correct_per_class, count_per_class,
        )
    split_loss = running / dataset_size if dataset_size else 0.0
    split_acc = correct / total if total else 0.0
    loss_per_label, acc_per_label = per_class_metrics_from_counters(
        loss_sum, correct_per_class, count_per_class, idx_to_label,
    )
    return split_loss, split_acc, loss_per_label, acc_per_label


def _plot_per_label_split(
    ax,
    epochs: range,
    loss_per_label: dict[str, list[float]],
    *,
    title: str,
) -> None:
    labels = list(loss_per_label)
    label_colors = _per_label_colors(len(labels))
    for color, label in zip(label_colors, labels):
        ax.plot(epochs, loss_per_label[label], color=color, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def _plot_per_label_acc_split(
    ax,
    epochs: range,
    acc_per_label: dict[str, list[float]],
    *,
    title: str,
) -> None:
    labels = list(acc_per_label)
    label_colors = _per_label_colors(len(labels))
    for color, label in zip(label_colors, labels):
        ax.plot(epochs, acc_per_label[label], color=color, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def _per_label_colors(n: int) -> list:
    """Distinct colors for per-label plots; scales beyond the default 10-color cycle."""
    if n <= 0:
        return []
    if n <= 20:
        return list(plt.colormaps["tab20"].colors[:n])
    if n <= 40:
        tab20 = plt.colormaps["tab20"].colors
        set3 = plt.colormaps["Set3"].colors
        return list(tab20) + list(set3)[: n - len(tab20)]
    cmap = plt.colormaps["hsv"]
    return [cmap(i / n) for i in range(n)]


def plot_training_history(
    history: dict[str, list[float] | dict[str, list[float]]],
    path: Path,
    *,
    idx_to_label: dict[int, str],
    best_epoch: int | None = None,
) -> None:
    n_epochs = len(history["train_loss"])
    epochs = range(1, n_epochs + 1)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    ax_loss, ax_acc = axes[0]
    ax_train_loss_label, ax_train_acc_label = axes[1]
    ax_val_loss_label, ax_val_acc_label = axes[2]
    ax_test_loss_label, ax_test_acc_label = axes[3]

    def mark_best_epoch(ax, *, with_label: bool = False) -> None:
        if best_epoch is None or not (1 <= best_epoch <= n_epochs):
            return
        ax.axvline(
            best_epoch,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=f"best val acc (epoch {best_epoch})" if with_label else None,
        )

    ax_loss.plot(epochs, history["train_loss"], label="train loss")
    ax_loss.plot(epochs, history["val_loss"], label="val loss")
    ax_loss.plot(epochs, history["test_loss"], label="test loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training, validation, and test loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train acc")
    ax_acc.plot(epochs, history["val_acc"], label="val acc")
    ax_acc.plot(epochs, history["test_acc"], label="test acc")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Training, validation, and test accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)

    _plot_per_label_split(
        ax_train_loss_label,
        epochs,
        history["train_loss_per_label"],
        title="Per-label train loss",
    )
    _plot_per_label_acc_split(
        ax_train_acc_label,
        epochs,
        history["train_acc_per_label"],
        title="Per-label train accuracy",
    )
    _plot_per_label_split(
        ax_val_loss_label,
        epochs,
        history["val_loss_per_label"],
        title="Per-label val loss",
    )
    _plot_per_label_acc_split(
        ax_val_acc_label,
        epochs,
        history["val_acc_per_label"],
        title="Per-label val accuracy",
    )
    _plot_per_label_split(
        ax_test_loss_label,
        epochs,
        history["test_loss_per_label"],
        title="Per-label test loss",
    )
    _plot_per_label_acc_split(
        ax_test_acc_label,
        epochs,
        history["test_acc_per_label"],
        title="Per-label test accuracy",
    )

    for row in axes:
        for ax in row:
            mark_best_epoch(ax)
    mark_best_epoch(ax_acc, with_label=True)
    for row in axes:
        for ax in row:
            ax.set_xlim(1, n_epochs)
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved training plot -> {path}")



def train(
    model,
    splits,
    label_to_idx,
    model_config,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    *,
    run_dir: Path,
    save_interval: int = CHECKPOINT_SAVE_INTERVAL,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = EARLY_STOPPING_METRIC,
    early_stopping_smooth_window: int = EARLY_STOPPING_SMOOTH_WINDOW,
    num_workers=0,
    pin_memory=False,
    continued_from: dict | None = None,
):
    n_classes = len(label_to_idx)
    model = model.to(device)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_ds = FusionDataset(splits.dataset, splits.train.indices, label_to_idx)
    val_ds = FusionDataset(splits.dataset, splits.val.indices, label_to_idx)
    test_ds = FusionDataset(splits.dataset, splits.test.indices, label_to_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_state: dict | None = None
    best_metrics = {
        "epoch": 0,
        "train_loss": 0.0,
        "train_acc": 0.0,
        "val_loss": 0.0,
        "val_acc": 0.0,
    }
    history: dict[str, list[float] | dict[str, list[float]]] = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "train_loss_per_label": {label: [] for label in label_to_idx},
        "val_loss_per_label": {label: [] for label in label_to_idx},
        "test_loss_per_label": {label: [] for label in label_to_idx},
        "train_acc_per_label": {label: [] for label in label_to_idx},
        "val_acc_per_label": {label: [] for label in label_to_idx},
        "test_acc_per_label": {label: [] for label in label_to_idx},
    }
    epochs_without_improve = 0
    stop_metric_key = "val_loss" if early_stopping_metric == "loss" else "val_acc"
    best_smoothed_stop = float("inf") if early_stopping_metric == "loss" else float("-inf")
    best_smoothed_stop_epoch = 0

    def write_checkpoint(
        path: Path,
        *,
        kind: str,
        epoch_num: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        state_dict: dict | None = None,
        log: bool = True,
    ) -> None:
        save_checkpoint(
            path,
            model,
            label_to_idx,
            model_config,
            epoch=epoch_num,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            best_acc=best_acc,
            total_epochs=epochs,
            kind=kind,
            state_dict=state_dict,
            continued_from=continued_from,
            log=log,
        )

    epoch_bar = tqdm(
        range(epochs),
        desc="epochs",
        unit="epoch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    run_start = time.perf_counter()
    with ExitStack() as stack:
        flop_counter = stack.enter_context(FlopCounterMode(display=False))
        for epoch in epoch_bar:
            epoch_start = time.perf_counter()
            model.train()
            running = 0.0
            train_correct = train_total = 0
            train_loss_sum, train_correct_per_class, train_count_per_class = init_per_class_counters(
                n_classes, device,
            )
            for eeg, emg, y_soft, y_hard in tqdm(train_dl, desc="train", leave=False):
                eeg, emg = eeg.to(device), emg.to(device)
                y_soft = y_soft.to(device)
                y_hard = y_hard.to(device)
                opt.zero_grad()
                logits = model(eeg, emg)
                loss = soft_cross_entropy(logits, y_soft)
                loss.backward()
                opt.step()
                running += loss.item() * y_hard.size(0)
                pred = logits.argmax(1)
                train_correct += (pred == y_hard).sum().item()
                train_total += y_hard.size(0)
                update_per_class_counters(
                    logits, y_hard, pred,
                    train_loss_sum, train_correct_per_class, train_count_per_class,
                )

            train_loss = running / len(train_ds)
            train_acc = train_correct / train_total if train_total else 0.0
            train_loss_per_label, train_acc_per_label = per_class_metrics_from_counters(
                train_loss_sum, train_correct_per_class, train_count_per_class, idx_to_label,
            )

        
            val_loss, val_acc, val_loss_per_label, val_acc_per_label = evaluate_fusion_split(
                model,
                val_dl,
                len(val_ds),
                n_classes=n_classes,
                idx_to_label=idx_to_label,
                desc="val",
            )
            test_loss, test_acc, test_loss_per_label, test_acc_per_label = evaluate_fusion_split(
                model,
                test_dl,
                len(test_ds),
                n_classes=n_classes,
                idx_to_label=idx_to_label,
                desc="test",
            )

            epoch_num = epoch + 1
            epoch_notes: list[str] = []

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["test_acc"].append(test_acc)
            for label in label_to_idx:
                history["train_loss_per_label"][label].append(train_loss_per_label[label])
                history["val_loss_per_label"][label].append(val_loss_per_label[label])
                history["test_loss_per_label"][label].append(test_loss_per_label[label])
                history["train_acc_per_label"][label].append(train_acc_per_label[label])
                history["val_acc_per_label"][label].append(val_acc_per_label[label])
                history["test_acc_per_label"][label].append(test_acc_per_label[label])

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = {
                    "epoch": epoch_num,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                write_checkpoint(
                    run_dir / "best.pt",
                    kind="best",
                    epoch_num=epoch_num,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    state_dict=best_state,
                    log=False,
                )
                epoch_notes.append(f"saved best.pt (epoch {epoch_num})")

            smoothed_stop = smoothed_tail(history[stop_metric_key], early_stopping_smooth_window)
            if smoothed_stop is not None:
                if early_stopping_metric == "loss":
                    stop_improved = smoothed_stop < best_smoothed_stop
                else:
                    stop_improved = smoothed_stop > best_smoothed_stop
                if stop_improved:
                    best_smoothed_stop = smoothed_stop
                    best_smoothed_stop_epoch = epoch_num
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

            if save_interval > 0 and epoch_num % save_interval == 0:
                ckpt_name = f"epoch_{epoch_num:04d}.pt"
                write_checkpoint(
                    run_dir / ckpt_name,
                    kind="epoch",
                    epoch_num=epoch_num,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    log=False,
                )
                epoch_notes.append(f"saved {ckpt_name}")

            epoch_time = time.perf_counter() - epoch_start
            epoch_bar.write(
                format_epoch_summary(
                    epoch_num,
                    epochs,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                    best_acc=best_acc,
                    best_epoch=best_metrics["epoch"],
                    epoch_time=epoch_time,
                    epochs_without_improve=epochs_without_improve if early_stopping_patience > 0 else None,
                    smoothed_stop=smoothed_stop if early_stopping_patience > 0 else None,
                    notes=epoch_notes or None,
                )
            )
            epoch_bar.write("")
            epoch_bar.write("")

            if (
                early_stopping_patience > 0
                and smoothed_stop is not None
                and epochs_without_improve >= early_stopping_patience
            ):
                metric_label = "val loss" if early_stopping_metric == "loss" else "val acc"
                epoch_bar.write(
                    f"early stopping at epoch {epoch_num}: "
                    f"no smoothed {metric_label} improvement for {early_stopping_patience} epochs "
                    f"(window={early_stopping_smooth_window}, "
                    f"best_smooth={best_smoothed_stop:.4f} @ epoch {best_smoothed_stop_epoch}; "
                    f"best_acc={best_acc:.4f} @ epoch {best_metrics['epoch']})"
                )
                break

    compute_elapsed = time.perf_counter() - run_start
    total_flops = flop_counter.get_total_flops()
    epochs_completed = len(history["train_loss"])

    last_epoch = epochs_completed
    write_checkpoint(
        run_dir / "last.pt",
        kind="last",
        epoch_num=last_epoch,
        train_loss=history["train_loss"][-1],
        train_acc=history["train_acc"][-1],
        val_loss=history["val_loss"][-1],
        val_acc=history["val_acc"][-1],
    )

    if best_state is not None:
        write_checkpoint(
            run_dir / "best.pt",
            kind="best",
            epoch_num=best_metrics["epoch"],
            train_loss=best_metrics["train_loss"],
            train_acc=best_metrics["train_acc"],
            val_loss=best_metrics["val_loss"],
            val_acc=best_metrics["val_acc"],
            state_dict=best_state,
        )
        model.load_state_dict(best_state)

    plot_training_history(
        history,
        run_dir / "training_history.png",
        idx_to_label=idx_to_label,
        best_epoch=best_metrics["epoch"] if best_state is not None else None,
    )

    run_elapsed = time.perf_counter() - run_start
    tqdm.write("")
    tqdm.write(
        f"training finished: {epochs_completed} epoch(s) in {format_duration(run_elapsed)} | "
        f"total {format_flops(total_flops)}"
    )
    if compute_elapsed > 0 and total_flops > 0:
        tqdm.write(f"  avg throughput: {format_flops(int(total_flops / compute_elapsed))}/s")
    tqdm.write("")

    return model, history, run_dir


def run_training(
    *,
    architecture: str = MODEL_ARCHITECTURE,
    splits,
    run_dir: Path | None = None,
    continue_from: Path | None = None,
    seed: int = SEED,
    deterministic: bool = TORCH_DETERMINISTIC,
    train_kwargs: dict | None = None,
) -> dict:
    """Train one model on the given splits; return model, history, and run metadata."""
    seed_everything(seed, deterministic)
    continued_from: dict | None = None
    kwargs = dict(train_kwargs or {})

    if continue_from is not None:
        ckpt_path = Path(continue_from)
        model, label_to_idx, model_config, continued_from = load_training_checkpoint(ckpt_path)
        if run_dir is None:
            run_dir = create_run_dir(continued=True)
        print(f"continuing from {ckpt_path}")
        print(
            f"  source run: {continued_from['source_run_dir']} "
            f"(epoch {continued_from['source_epoch']}, "
            f"val_acc={continued_from['source_val_acc']:.4f}, "
            f"kind={continued_from['source_kind']})"
        )
    else:
        if run_dir is None:
            run_dir = create_run_dir()
        else:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=False)
        model, label_to_idx, model_config = construct_model(splits, architecture=architecture)

    print(f"window T={model_config['T']} samples, classes={model_config['n_classes']}")
    print("\n\n")
    print(f"starting training on device {device} at {run_dir}")
    print("\n")

    model, history, run_dir = train(
        model,
        splits,
        label_to_idx,
        model_config,
        run_dir=run_dir,
        continued_from=continued_from,
        **kwargs,
    )

    best_epoch = 0
    best_val_acc = 0.0
    if history["val_acc"]:
        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1

    return {
        "model": model,
        "history": history,
        "run_dir": run_dir,
        "label_to_idx": label_to_idx,
        "model_config": model_config,
        "architecture": architecture,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "continued_from": continued_from,
    }


# --- main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fusion EEG/EMG model.")
    parser.add_argument(
        "--model",
        choices=sorted(ARCHITECTURES),
        default=MODEL_ARCHITECTURE,
        help=f"model architecture (default: {MODEL_ARCHITECTURE})",
    )
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_training",
        help="Load the latest checkpoint and train in a new -continued run folder",
    )
    args = parser.parse_args()

    SPLITS_DIR = Path(__file__).resolve().parent / "splits"
    splits = load_dataset_splits(SPLITS_DIR)

    continue_from: Path | None = None
    if args.continue_training:
        source_run = latest_run_dir()
        if source_run is None:
            raise SystemExit("No checkpoint runs found under model/checkpoints; train from scratch first.")
        continue_from = latest_checkpoint_in_run(source_run)

    result = run_training(
        architecture=args.model,
        splits=splits,
        continue_from=continue_from,
    )

    print("\n")
    print(f"artifacts saved under {result['run_dir']}")
