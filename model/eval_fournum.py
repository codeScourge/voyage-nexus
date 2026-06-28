"""
eval_fournum.py
===============
Phase 9. Four-number held-out report for a SPECIFIC IntermediateFusionEEGNet
checkpoint:
    balanced accuracy | Cohen's kappa | macro-F1 (support>0 classes) | chance

Self-contained on purpose: it takes an explicit --checkpoint so you can score
each arm x seed unambiguously, instead of depending on whichever run is "latest".
It rebuilds the model from the checkpoint's own model_config and reuses train.py's
FusionDataset, so channel selection and per-window normalization match training
exactly. IntermediateFusionEEGNet is native here -- no val.py import swap needed.

Usage:
    python eval_fournum.py --checkpoint checkpoints/<run_dir>/best.pt
    python eval_fournum.py --checkpoint checkpoints/<run_dir>/best.pt --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

import train as base


def evaluate(checkpoint, split="test", batch=64):
    ckpt = torch.load(checkpoint, map_location=base.device)
    cfg = ckpt["model_config"]
    l2i = ckpt["label_to_idx"]

    model = base.IntermediateFusionEEGNet(
        n_eeg=cfg["n_eeg"], n_emg=cfg["n_emg"],
        n_classes=cfg["n_classes"], T=cfg["T"], p_drop=0.5).to(base.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    splits = base.load_dataset_splits(Path(base.__file__).resolve().parent / "splits")
    if not hasattr(splits, split):
        print(f"[warn] splits has no '{split}' set; falling back to 'val'")
        split = "val"
    subset = getattr(splits, split)

    ds = base.FusionDataset(splits.dataset, subset.indices, l2i)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)

    yt, yp = [], []
    with torch.no_grad():
        for eeg, emg, y in dl:
            logits = model(eeg.to(base.device), emg.to(base.device))
            yp.append(logits.argmax(1).cpu().numpy())
            yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)

    present = sorted(set(yt.tolist()))                       # support>0 classes only
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "n": int(len(yt)),
        "n_classes_slots": int(cfg["n_classes"]),            # softmax width
        "n_classes_present": len(present),                   # populated classes
        "balanced_acc": float(balanced_accuracy_score(yt, yp)),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=present, average="macro",
                                   zero_division=0)),
        "chance": 1.0 / max(len(present), 1),                # 1/populated, not 1/slots
        "support": {int(c): int((yt == c).sum()) for c in present},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", help="test | val | train")
    a = ap.parse_args()
    r = evaluate(a.checkpoint, a.split)
    print(f"\n{r['checkpoint']}   [{r['split']}, n={r['n']}]")
    print(f"  balanced_acc {r['balanced_acc']:.4f}")
    print(f"  cohen_kappa  {r['cohen_kappa']:.4f}")
    print(f"  macro_f1     {r['macro_f1']:.4f}   (support>0 classes)")
    print(f"  chance       {r['chance']:.4f}   "
          f"(1/{r['n_classes_present']} populated; {r['n_classes_slots']} slots)")
    print(f"  support      {r['support']}")
