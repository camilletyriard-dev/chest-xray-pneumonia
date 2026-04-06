"""
utils.py — Reproducibility, class-weight computation, and misc helpers.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    return device


def compute_class_weights(
    data_dir:    str,
    split:       str = "train",
    classes:     list[str] = ("NORMAL", "PNEUMONIA"),
    device:      torch.device | None = None,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a split directory.

    Weights are scaled so that the loss contribution of each class is
    proportional to its inverse frequency, balancing majority/minority classes.

    Args:
        data_dir: Root of the re-split dataset.
        split:    Which split to count (default ``"train"``).
        classes:  Ordered list of class subdirectory names.
        device:   Move the tensor to this device before returning.

    Returns:
        Float tensor of shape ``(len(classes),)``.

    Example::

        weights = compute_class_weights("data/chest_xray_split")
        criterion = nn.CrossEntropyLoss(weight=weights)
    """
    counts = [
        len(os.listdir(os.path.join(data_dir, split, cls)))
        for cls in classes
    ]
    total = sum(counts)
    weights = torch.tensor([total / c for c in counts], dtype=torch.float)

    for cls, cnt, w in zip(classes, counts, weights):
        print(f"  {cls}: {cnt} samples → weight {w:.4f}")

    if device is not None:
        weights = weights.to(device)
    return weights


def load_checkpoint(
    model:          nn.Module,
    checkpoint_path: str,
    device:         torch.device,
) -> dict:
    """
    Load a checkpoint saved by ``train.train()`` into ``model``.

    Returns the full checkpoint dict (contains epoch, val_loss, val_acc).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(
        f"[INFO] Loaded checkpoint from '{checkpoint_path}' "
        f"(epoch {ckpt.get('epoch', '?')}, "
        f"val_acc={ckpt.get('val_acc', float('nan'))*100:.2f}%)"
    )
    return ckpt
