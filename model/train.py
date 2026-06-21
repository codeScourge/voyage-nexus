from __future__ import annotations

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

from data import load_dataset_splits

# ---
SEED = 42
TORCH_DETERMINISTIC = False
RUN_DIR_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

# --- channel selection (set False to exclude from training)
EEG1 = 0
EEG2 = 1
EEG3 = 0
EEG4 = 1
EEG5 = 1
EEG6 = 0
EEG7 = 1
EEG8 = 1
EEG9 = 1
EEG10 = 1
EEG11 = 0
EEG12 = 1
EEG13 = 0
EEG14 = 0
EEG15 = 0
EEG16 = 0

EMG1 = 1
EMG2 = 0
EMG3 = 1
EMG4 = 1
EMG5 = 0
EMG6 = 1
EMG7 = 0
EMG8 = 1
EMG9 = 0
EMG10 = 1
EMG11 = 0
EMG12 = 0
EMG13 = 0
EMG14 = 1
EMG15 = 0
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
        F2: int = 16,
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
    labels = sorted({base_dataset[i]["label"] for i in indices})
    return {lab: idx for idx, lab in enumerate(labels)}


# --- train
def construct_model(splits):
    label_to_idx = build_label_map(splits.dataset, splits.train.indices)
    n_classes = len(label_to_idx)

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


def create_run_dir(root: Path = CHECKPOINT_DIR) -> Path:
    base_name = new_run_dir_name()
    run_dir = root / base_name
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
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
        },
        path,
    )

    print("\n")
    print(f"saved checkpoint ({kind}, epoch {epoch}) -> {path}")
    print("\n")


def plot_training_history(history: dict[str, list[float]], path: Path) -> None:
    
    epochs = range(1, len(history["train_loss"]) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

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
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

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
    early_stopping_patience: int = 15,
    num_workers=0,
    pin_memory=False,
):
    n_classes = len(label_to_idx)
    model = model.to(device)

    train_ds = FusionDataset(splits.dataset, splits.train.indices, label_to_idx)
    val_ds = FusionDataset(splits.dataset, splits.val.indices, label_to_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # class weights for imbalance (inverse frequency)
    counts = torch.zeros(n_classes)
    for i in train_ds.indices:
        counts[label_to_idx[splits.dataset[i]["label"]]] += 1
    weights = (counts.sum() / (counts * n_classes)).to(device)

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
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    epochs_without_improve = 0

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
        )

    epoch_bar = tqdm(range(epochs), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        train_correct = train_total = 0
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

        train_loss = running / len(train_ds)
        train_acc = train_correct / train_total if train_total else 0.0

        
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            correct = total = 0
            for eeg, emg, y in tqdm(val_dl, desc="val", leave=False):
                eeg, emg, y = eeg.to(device), emg.to(device), y.to(device)
                logits = model(eeg, emg)
                val_running += crit(logits, y).item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            val_loss = val_running / len(val_ds) if len(val_ds) else 0.0
            acc = correct / total if total else 0.0

        epoch_num = epoch + 1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(acc)

        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": epoch_num,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": acc,
            }
            epochs_without_improve = 0
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
        epoch_bar.set_postfix(**postfix)

        if early_stopping_patience > 0 and epochs_without_improve >= early_stopping_patience:
            print(
                f"\nearly stopping at epoch {epoch_num}: "
                f"no val acc improvement for {early_stopping_patience} epochs "
                f"(best={best_acc:.4f} @ epoch {best_metrics['epoch']})"
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

    plot_training_history(history, run_dir / "training_history.png")

    return model, history, run_dir


# --- main

if __name__ == "__main__":
    SPLITS_DIR = Path(__file__).resolve().parent / "splits"
    seed_everything(SEED, TORCH_DETERMINISTIC)
    splits = load_dataset_splits(SPLITS_DIR)
    # print(splits)

    run_dir = create_run_dir()
    untrained_model, label_to_idx, model_config = construct_model(splits)
    print(f"window T={model_config['T']} samples, classes={model_config['n_classes']}")

    print("\n\n")
    print(f"starting training on device {device} at {run_dir}")
    print("\n")

    model, history, run_dir = train(untrained_model, splits, label_to_idx, model_config, run_dir=run_dir)

    print("\n")
    print(f"'artifacts saved under {run_dir}")
