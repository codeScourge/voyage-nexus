"""
pretrain_tmspd_eegnet.py
========================
The "downstream" arm, step 1: supervised-pretrain the SHIPPED
``IntermediateFusionEEGNet`` (imported from train.py -- single source of truth)
on T-MSPD, and save a checkpoint whose montage-INDEPENDENT layers
(temporal convs + bn1 + Block-2 separable conv) can be warm-started into a Voyage
model by ``transfer_init`` in train_voyage_from_tmspd.py.

Two things that MUST hold for this to be valid, both handled here:

1. COMMON 1 kHz on both modalities. IntermediateFusionEEGNet fuses by cropping to
   min(T) across branches -- a padding safety, NOT a rate adapter. EEG@250 +
   EMG@1000 would silently truncate the EMG feature map to the EEG length. So we
   load BOTH at 1 kHz (eeg_fs_out=1000), which also matches the Voyage rig (1 kHz
   both modalities) and the model's kern_eeg=500 (= half the sample rate).

2. INPUT NORMALIZATION matches Voyage. train.py's FusionDataset z-scores each
   channel per window, for BOTH modalities. So we pass norm_eeg/norm_emg=pertrial,
   not the "global" EMG default, so the conv front-ends see the same input
   statistics they'll see on Voyage.

The EEG channel count used here does NOT affect what transfers: the spatial conv
is channel-specific and is re-init at transfer time regardless. We use full 64ch
by default for the richest supervised signal; --eeg-channels periauric is the
montage-honest alternative.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

from train import IntermediateFusionEEGNet, get_device, seed_everything, SEED
from tmspd_loader import load_tmspd_dataset

device = get_device()


def make_loaders(Xe, Xm, y, groups, *, batch=32, val_frac=0.2, seed=SEED):
    """Subject-grouped train/val so the val split isn't leaked within-subject."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(gss.split(Xe, y, groups))

    def ds(idx):
        return TensorDataset(
            torch.from_numpy(Xe[idx]).unsqueeze(1).float(),   # (N,1,Ceeg,T)
            torch.from_numpy(Xm[idx]).unsqueeze(1).float(),   # (N,1,Cemg,T)
            torch.from_numpy(y[idx]).long())

    return (DataLoader(ds(tr), batch_size=batch, shuffle=True, drop_last=True),
            DataLoader(ds(va), batch_size=batch, shuffle=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="T-MSPD root containing {mode}/S{nn}/EEG|sEMG/...")
    ap.add_argument("--mode", default="overt speech",
                    choices=["overt speech", "silent speech", "imagined speech"])
    ap.add_argument("--eeg-channels", default="all",
                    help='"all" (64ch) | "periauric" (9ch) | comma-separated names')
    ap.add_argument("--subjects", default="1-30", help="inclusive range, e.g. 1-30")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    seed_everything(a.seed, deterministic=False)
    lo, hi = (int(x) for x in a.subjects.split("-"))
    chans = (a.eeg_channels if a.eeg_channels in ("all", "periauric")
             else a.eeg_channels.split(","))

    # COMMON 1 kHz + Voyage-matched per-trial normalization
    Xe, Xm, y, groups = load_tmspd_dataset(
        a.root, mode=a.mode, subjects=range(lo, hi + 1),
        eeg_channels=chans, eeg_fs_out=1000, emg_fs_out=1000,
        norm_eeg="pertrial", norm_emg="pertrial")

    n_eeg, T = Xe.shape[1], Xe.shape[2]
    n_emg = Xm.shape[1]
    n_classes = int(np.max(y)) + 1
    assert Xm.shape[2] == T, f"EEG T={T} != EMG T={Xm.shape[2]} (both must be 1 kHz)"
    print(f"T-MSPD[{a.mode}]: Xe{Xe.shape} Xm{Xm.shape} y{y.shape} "
          f"classes={n_classes} subjects={len(np.unique(groups))}")

    tr_dl, va_dl = make_loaders(Xe, Xm, y, groups, seed=a.seed)

    # IDENTICAL architecture hyperparams to train.construct_model (p_drop=0.5,
    # everything else default) so the transferable layer shapes match exactly.
    model = IntermediateFusionEEGNet(n_eeg=n_eeg, n_emg=n_emg,
                                     n_classes=n_classes, T=T, p_drop=0.5).to(device)

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes),
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best, best_state = 0.0, None
    for ep in range(a.epochs):
        model.train()
        for eeg, emg, yb in tr_dl:
            eeg, emg, yb = eeg.to(device), emg.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(eeg, emg), yb)
            loss.backward()
            opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for eeg, emg, yb in va_dl:
                eeg, emg, yb = eeg.to(device), emg.to(device), yb.to(device)
                pred = model(eeg, emg).argmax(1)
                c += (pred == yb).sum().item()
                t += yb.numel()
        acc = c / max(t, 1)
        if acc >= best:
            best = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"  ep {ep + 1:3d}/{a.epochs}  val_acc={acc:.3f}  best={best:.3f}")

    out = Path(a.out or f"tmspd_pretrain_{a.mode.replace(' ', '_')}_"
                        f"{datetime.now():%Y%m%d_%H%M%S}.pt")
    torch.save({"model_state_dict": best_state or model.state_dict(),
                "model_config": {"n_eeg": n_eeg, "n_emg": n_emg,
                                 "n_classes": n_classes, "T": T, "fs": 1000,
                                 "eeg_channels": a.eeg_channels, "mode": a.mode,
                                 "arch": "IntermediateFusionEEGNet"},
                "best_val_acc": best}, out)
    print(f"saved -> {out}  (best T-MSPD val acc {best:.3f})")


if __name__ == "__main__":
    main()
