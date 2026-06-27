from __future__ import annotations

import argparse
import os
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import default_label_to_idx, load_dataset_splits

# ---
SEED = 42
TORCH_DETERMINISTIC = False
RUN_DIR_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

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
ACTIVE_EEG_INDICES = [i for i, use in enumerate(_EEG_CHANNEL_USE) if use]
ACTIVE_EMG_INDICES = [16 + i for i, use in enumerate(_EMG_CHANNEL_USE) if use]

if not ACTIVE_EEG_INDICES:
    raise ValueError("At least one EEG channel must be enabled")
if not ACTIVE_EMG_INDICES:
    raise ValueError("At least one EMG channel must be enabled")


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
CHECKPOINT_SAVE_INTERVAL = 30
EARLY_STOPPING_METRIC = "loss"  # "loss" or "acc" — stop signal only, not best.pt selection
EARLY_STOPPING_SMOOTH_WINDOW = 4


# --- architecture
class ModalityBranch(nn.Module):
    """EEGNet Block 1: temporal conv -> depthwise spatial conv.

    Input:  (B, 1, C, T)
    Output: (B, D*F1, 1, T)   -- spatial axis collapsed, time preserved
    """

    def __init__(self, n_channels: int, F1: int, D: int, kernel_length: int):
        super().__init__()
        # temporal conv: 'same' padding so T is preserved. one shared kernel
        # across all channels, F1 of them.
        self.temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            padding=(0, kernel_length // 2), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # depthwise spatial conv: kernel (C, 1), valid padding -> collapses
        # channel axis to 1. groups=F1 ties each spatial filter to one temporal
        # map. depth multiplier D via out_channels = D*F1.
        self.spatial = nn.Conv2d(
            F1, D * F1, (n_channels, 1),
            groups=F1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(D * F1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.temporal(x))          # (B, F1, C, T)
        x = self.bn2(self.spatial(x))           # (B, D*F1, 1, T)
        x = F.elu(x)
        return x


class TimeAvgPool(nn.Module):
    """Pool along the time axis to a fixed length without AdaptiveAvgPool2d.

    MPS does not implement adaptive pooling when input length is not divisible
    by the target length (pytorch#96056). Trim trailing samples if needed, then
    use fixed-kernel average pooling.
    """

    def __init__(self, out_len: int):
        super().__init__()
        self.out_len = out_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 1, T)
        t = x.shape[-1]
        out = self.out_len
        if t == out:
            return x
        if t < out:
            return F.interpolate(x, size=(1, out), mode="linear", align_corners=False)

        trim = t - (t % out)
        x = x[..., :trim]
        stride = trim // out
        return F.avg_pool2d(x, kernel_size=(1, stride), stride=(1, stride))


class IntermediateFusionEEGNet(nn.Module):
    """Two EEGNet Block-1 branches (EEG, EMG) fused before the separable conv.

    Fusion = concat along feature-map axis once both branches are
    (B, D*F1, 1, T). The shared separable conv then learns cross-modal
    temporal summaries and mixes EEG+EMG feature maps together.
    """

    def __init__(
        self,
        n_eeg: int,
        n_emg: int,
        n_classes: int,
        T: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 32,
        kern_eeg: int = 128,   # half the sampling rate per the paper
        kern_emg: int = 128,   # tune: EMG carries higher-freq content
        sep_kernel: int = 16,
        p_drop: float = 0.5,
    ):
        super().__init__()
        self.eeg_branch = ModalityBranch(n_eeg, F1, D, kern_eeg)
        self.emg_branch = ModalityBranch(n_emg, F1, D, kern_emg)

        fused_maps = 2 * (D * F1)  # concat of both branches

        pool1_out = max(1, T // 4)
        pool2_out = max(1, pool1_out // 8)
        self.pool1_out = pool1_out
        self.pool2_out = pool2_out

        # fixed pools handle short windows and avoid MPS adaptive-pool limits
        self.pool1 = TimeAvgPool(pool1_out)
        self.drop1 = nn.Dropout(p_drop)

        # --- Block 2: separable conv on the FUSED maps ---
        # depthwise temporal part: per-map (1, sep_kernel) summary, 'same' pad
        self.sep_depth = nn.Conv2d(
            fused_maps, fused_maps, (1, sep_kernel),
            padding=(0, sep_kernel // 2), groups=fused_maps, bias=False,
        )
        # pointwise: mix all fused maps -> F2 (this is where EEG and EMG
        # feature maps actually combine)
        self.sep_point = nn.Conv2d(fused_maps, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = TimeAvgPool(pool2_out)
        self.drop2 = nn.Dropout(p_drop)

        self.classifier = nn.Linear(F2 * pool2_out, n_classes)

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        e = self.eeg_branch(eeg)   # (B, D*F1, 1, T)
        m = self.emg_branch(emg)   # (B, D*F1, 1, T)

        # align time length in case kernels/padding differ by a sample
        t = min(e.shape[-1], m.shape[-1])
        e, m = e[..., :t], m[..., :t]

        x = torch.cat([e, m], dim=1)   # (B, 2*D*F1, 1, T) <-- intermediate fusion
        x = self.drop1(self.pool1(x))

        x = self.sep_point(self.sep_depth(x))
        x = F.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))

        x = torch.flatten(x, 1)
        return self.classifier(x)

    def forward_embeddings(self, eeg: torch.Tensor, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flattened fusion embeddings without dropout (use under model.eval())."""
        e = self.eeg_branch(eeg)
        m = self.emg_branch(emg)
        t = min(e.shape[-1], m.shape[-1])
        e, m = e[..., :t], m[..., :t]

        x = torch.cat([e, m], dim=1)
        x = self.pool1(x)

        x = self.sep_point(self.sep_depth(x))
        pre_pool2 = F.elu(self.bn3(x))
        post_pool2 = self.pool2(pre_pool2)

        return {
            "pre_pool2": torch.flatten(pre_pool2, start_dim=1),
            "classifier_input": torch.flatten(post_pool2, start_dim=1),
        }


# --- dataset adapter
class FusionDataset(torch.utils.data.Dataset):
    """Wraps the split so __getitem__ returns (eeg, emg, label) batched-ready."""

    def __init__(self, base_dataset, indices, label_to_idx):
        self.base = base_dataset
        self.indices = list(indices)
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        sample = self.base[self.indices[i]]
        x = sample["x"]
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
                                
        eeg = x[:, ACTIVE_EEG_INDICES].T.unsqueeze(0)  # (1, C_eeg, T)
        emg = x[:, ACTIVE_EMG_INDICES].T.unsqueeze(0)  # (1, C_emg, T)
        y = self.label_to_idx[sample["label"]]
        return eeg, emg, torch.tensor(y, dtype=torch.long)


def build_label_map(base_dataset, indices) -> dict:
    label_to_idx = default_label_to_idx()
    seen = {base_dataset[i]["label"] for i in indices}
    unknown = seen - set(label_to_idx)
    if unknown:
        raise ValueError(f"Unknown labels in training data: {sorted(unknown)}")
    return label_to_idx


# --- train
def construct_model(splits):
    label_to_idx = build_label_map(splits.dataset, splits.train.indices)
    n_classes = len(label_to_idx)
    print(f"classes ({n_classes}): {', '.join(label_to_idx)}")

    # infer shapes from one sample
    eeg0, emg0, _ = FusionDataset(splits.dataset, splits.train.indices, label_to_idx)[0]
    n_eeg, T = eeg0.shape[1], eeg0.shape[2]
    n_emg = emg0.shape[1]

    print("\n\n")
    print("--- example sample ---")
    print("eeg: ", eeg0)
    print("\n")
    print("emg: ", emg0)
    print("------")
    print("\n\n")

    model = IntermediateFusionEEGNet(
        n_eeg=n_eeg, n_emg=n_emg, n_classes=n_classes, T=T,
        p_drop=0.5
    ).to(device)

    return model, label_to_idx, {
        "n_eeg": n_eeg,
        "n_emg": n_emg,
        "n_classes": n_classes,
        "T": T,
        "active_eeg_indices": ACTIVE_EEG_INDICES,
        "active_emg_indices": ACTIVE_EMG_INDICES,
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

    model = IntermediateFusionEEGNet(
        n_eeg=model_config["n_eeg"],
        n_emg=model_config["n_emg"],
        n_classes=model_config["n_classes"],
        T=model_config["T"],
    )
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

    print("\n")
    print(f"saved checkpoint ({kind}, epoch {epoch}) -> {path}")
    print("\n")


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


def update_per_class_counters(
    logits: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    loss_sum: torch.Tensor,
    correct: torch.Tensor,
    count: torch.Tensor,
) -> None:
    loss_per_sample = F.cross_entropy(logits, y, reduction="none")
    ones = torch.ones_like(y, dtype=loss_sum.dtype)
    count.scatter_add_(0, y, ones)
    loss_sum.scatter_add_(0, y, loss_per_sample)
    correct.scatter_add_(0, y, (pred == y).to(loss_sum.dtype))


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
    labels = [idx_to_label[i] for i in sorted(idx_to_label)]

    fig, ((ax_loss, ax_acc), (ax_loss_label, ax_acc_label)) = plt.subplots(2, 2, figsize=(12, 10))

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
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training and validation loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train acc")
    ax_acc.plot(epochs, history["val_acc"], label="val acc")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Training and validation accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)

    train_loss_per_label = history["train_loss_per_label"]
    val_loss_per_label = history["val_loss_per_label"]
    train_acc_per_label = history["train_acc_per_label"]
    val_acc_per_label = history["val_acc_per_label"]

    label_colors = _per_label_colors(len(labels))
    for color, label in zip(label_colors, labels):
        ax_loss_label.plot(
            epochs, train_loss_per_label[label], color=color, linestyle="-", label=f"train {label}",
        )
        ax_loss_label.plot(
            epochs, val_loss_per_label[label], color=color, linestyle=":", label=f"val {label}",
        )
        ax_acc_label.plot(
            epochs, train_acc_per_label[label], color=color, linestyle="-", label=f"train {label}",
        )
        ax_acc_label.plot(
            epochs, val_acc_per_label[label], color=color, linestyle=":", label=f"val {label}",
        )

    ax_loss_label.set_xlabel("epoch")
    ax_loss_label.set_ylabel("loss")
    ax_loss_label.set_title("Per-label training and validation loss")
    ax_loss_label.legend(fontsize=7, ncol=2)
    ax_loss_label.grid(True, alpha=0.3)

    ax_acc_label.set_xlabel("epoch")
    ax_acc_label.set_ylabel("accuracy")
    ax_acc_label.set_title("Per-label training and validation accuracy")
    ax_acc_label.set_ylim(0.0, 1.0)
    ax_acc_label.legend(fontsize=7, ncol=2)
    ax_acc_label.grid(True, alpha=0.3)

    mark_best_epoch(ax_loss)
    mark_best_epoch(ax_acc, with_label=True)
    mark_best_epoch(ax_loss_label)
    mark_best_epoch(ax_acc_label)
    for ax in (ax_loss, ax_acc, ax_loss_label, ax_acc_label):
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

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # class weights for imbalance (inverse frequency)
    counts = torch.zeros(n_classes)
    for i in train_ds.indices:
        counts[label_to_idx[splits.dataset[i]["label"]]] += 1
    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    weights = (counts.sum() / (safe_counts * n_classes)).to(device)

    crit = nn.CrossEntropyLoss(weight=weights)

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
        "train_acc": [],
        "val_acc": [],
        "train_loss_per_label": {label: [] for label in label_to_idx},
        "val_loss_per_label": {label: [] for label in label_to_idx},
        "train_acc_per_label": {label: [] for label in label_to_idx},
        "val_acc_per_label": {label: [] for label in label_to_idx},
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
        )

    epoch_bar = tqdm(range(epochs), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        train_correct = train_total = 0
        train_loss_sum, train_correct_per_class, train_count_per_class = init_per_class_counters(
            n_classes, device,
        )
        for eeg, emg, y in tqdm(train_dl, desc="train", leave=False):
            eeg, emg, y = eeg.to(device), emg.to(device), y.to(device)
            opt.zero_grad()
            logits = model(eeg, emg)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            pred = logits.argmax(1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
            update_per_class_counters(
                logits, y, pred,
                train_loss_sum, train_correct_per_class, train_count_per_class,
            )

        train_loss = running / len(train_ds)
        train_acc = train_correct / train_total if train_total else 0.0
        train_loss_per_label, train_acc_per_label = per_class_metrics_from_counters(
            train_loss_sum, train_correct_per_class, train_count_per_class, idx_to_label,
        )

        
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            correct = total = 0
            val_loss_sum, val_correct_per_class, val_count_per_class = init_per_class_counters(
                n_classes, device,
            )
            for eeg, emg, y in tqdm(val_dl, desc="val", leave=False):
                eeg, emg, y = eeg.to(device), emg.to(device), y.to(device)
                logits = model(eeg, emg)
                val_running += crit(logits, y).item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                update_per_class_counters(
                    logits, y, pred,
                    val_loss_sum, val_correct_per_class, val_count_per_class,
                )
            val_loss = val_running / len(val_ds) if len(val_ds) else 0.0
            acc = correct / total if total else 0.0
            val_loss_per_label, val_acc_per_label = per_class_metrics_from_counters(
                val_loss_sum, val_correct_per_class, val_count_per_class, idx_to_label,
            )

        epoch_num = epoch + 1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(acc)
        for label in label_to_idx:
            history["train_loss_per_label"][label].append(train_loss_per_label[label])
            history["val_loss_per_label"][label].append(val_loss_per_label[label])
            history["train_acc_per_label"][label].append(train_acc_per_label[label])
            history["val_acc_per_label"][label].append(val_acc_per_label[label])

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch_num,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": acc,
            }
            write_checkpoint(
                run_dir / "best.pt",
                kind="best",
                epoch_num=epoch_num,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=acc,
                state_dict=best_state,
            )

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
            write_checkpoint(
                run_dir / f"epoch_{epoch_num:04d}.pt",
                kind="epoch",
                epoch_num=epoch_num,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=acc,
            )

        epoch_time = time.perf_counter() - epoch_start
        postfix = {
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_acc": f"{acc:.4f}",
            "best": f"{best_acc:.4f}",
            "epoch_s": f"{epoch_time:.1f}s",
        }
        if early_stopping_patience > 0:
            postfix["no_improve"] = epochs_without_improve
            if smoothed_stop is not None:
                postfix["stop_smooth"] = f"{smoothed_stop:.4f}"
        epoch_bar.set_postfix(**postfix)

        if (
            early_stopping_patience > 0
            and smoothed_stop is not None
            and epochs_without_improve >= early_stopping_patience
        ):
            metric_label = "val loss" if early_stopping_metric == "loss" else "val acc"
            print(
                f"\nearly stopping at epoch {epoch_num}: "
                f"no smoothed {metric_label} improvement for {early_stopping_patience} epochs "
                f"(window={early_stopping_smooth_window}, "
                f"best_smooth={best_smoothed_stop:.4f} @ epoch {best_smoothed_stop_epoch}; "
                f"best_acc={best_acc:.4f} @ epoch {best_metrics['epoch']})"
            )
            break

    last_epoch = len(history["train_loss"])
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

    return model, history, run_dir


# --- main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fusion EEGNet model.")
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_training",
        help="Load the latest checkpoint and train in a new -continued run folder",
    )
    args = parser.parse_args()

    SPLITS_DIR = Path(__file__).resolve().parent / "splits"
    seed_everything(SEED, TORCH_DETERMINISTIC)
    splits = load_dataset_splits(SPLITS_DIR)

    continued_from: dict | None = None
    if args.continue_training:
        source_run = latest_run_dir()
        if source_run is None:
            raise SystemExit("No checkpoint runs found under model/checkpoints; train from scratch first.")
        ckpt_path = latest_checkpoint_in_run(source_run)
        model, label_to_idx, model_config, continued_from = load_training_checkpoint(ckpt_path)
        run_dir = create_run_dir(continued=True)
        print(f"continuing from {ckpt_path}")
        print(
            f"  source run: {continued_from['source_run_dir']} "
            f"(epoch {continued_from['source_epoch']}, "
            f"val_acc={continued_from['source_val_acc']:.4f}, "
            f"kind={continued_from['source_kind']})"
        )
    else:
        run_dir = create_run_dir()
        model, label_to_idx, model_config = construct_model(splits)

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
    )

    print("\n")
    print(f"artifacts saved under {run_dir}")
