"""
train_voyage_from_tmspd.py
==========================
The "downstream" arm, step 2: build the Voyage ``IntermediateFusionEEGNet``
EXACTLY as train.py does (same channel flags, same shapes, same splits), warm-
start the montage-INDEPENDENT layers from a T-MSPD checkpoint, then run the
IDENTICAL train.py training loop and checkpoint format.

Run BOTH arms through this one script so they are identical except for --init:
    DIRECT     : python train_voyage_from_tmspd.py --seed 42
    DOWNSTREAM : python train_voyage_from_tmspd.py --seed 42 --init tmspd_pretrain_*.pt
Same data, same loop, same seed handling, same checkpoint format -- only the init
differs, so the held-out delta between the two best.pt files is exactly what
T-MSPD pretraining bought the shipped model.

What transfer_init copies vs re-inits IS the montage ceiling, made explicit:
  COPIED  : eeg/emg_branch.temporal + bn1, sep_depth, sep_point, bn3
  RE-INIT : eeg/emg_branch.spatial + bn2 (channel-count-specific) and classifier
            (class/T-specific) -- the channel-mixing layer cannot cross montages.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import train as base


def transfer_init(model, ckpt_path, *, verbose=True):
    """Copy only shape-matching tensors from a T-MSPD checkpoint into `model`;
    leave everything else at its fresh random init."""
    src = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    tgt = model.state_dict()
    loaded, skipped = [], []
    for k, v in tgt.items():
        if k in src and src[k].shape == v.shape:
            tgt[k] = src[k].clone()
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(tgt)
    if verbose:
        def groupset(keys):
            return sorted({".".join(k.split(".")[:2]) for k in keys})
        print(f"[transfer_init] copied {len(loaded)} tensors from {ckpt_path}")
        print(f"[transfer_init]   warm-started layers: {groupset(loaded)}")
        print(f"[transfer_init]   re-init (montage/class-specific): {groupset(skipped)}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None,
                    help="T-MSPD pretrain checkpoint (.pt). OMIT for the DIRECT "
                         "arm (random init). Provide it for the DOWNSTREAM arm.")
    ap.add_argument("--seed", type=int, default=base.SEED)
    a = ap.parse_args()

    base.seed_everything(a.seed, base.TORCH_DETERMINISTIC)
    splits_dir = Path(base.__file__).resolve().parent / "splits"
    splits = base.load_dataset_splits(splits_dir)

    run_dir = base.create_run_dir()
    model, label_to_idx, model_config = base.construct_model(splits)
    model_config["seed"] = a.seed
    if a.init:
        model_config["init_from"] = a.init
        transfer_init(model, a.init)
    else:
        print("[direct] random init (no --init) -- this is the baseline arm")

    model, history, run_dir = base.train(
        model, splits, label_to_idx, model_config, run_dir=run_dir)
    print(f"artifacts under {run_dir}")


if __name__ == "__main__":
    main()
