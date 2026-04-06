"""
dataset.py — ChestXrayDataset and data loading utilities.
"""

import os
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = ["NORMAL", "PNEUMONIA"]

# Training-set statistics (precomputed; recompute with scripts/compute_stats.py
# if you use a different split or image size)
TRAIN_MEAN = 0.4812
TRAIN_STD  = 0.2204


# ── Dataset ──────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    Chest X-ray dataset (Kaggle: paultimothymooney/chest-xray-pneumonia).

    Expects the root directory to contain sub-folders::

        root_dir/
          train/  NORMAL/  PNEUMONIA/
          val/    NORMAL/  PNEUMONIA/
          test/   NORMAL/  PNEUMONIA/

    Labels: NORMAL → 0, PNEUMONIA → 1.
    Images are stored as (path, label) tuples and loaded lazily.
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir  = root_dir
        self.split     = split
        self.transform = transform

        split_dir = os.path.join(root_dir, split)
        self.images: list[tuple[str, int]] = []

        for label, cls in enumerate(CLASSES):
            class_folder = os.path.join(split_dir, cls)
            for filename in sorted(os.listdir(class_folder)):
                self.images.append((os.path.join(class_folder, filename), label))

    # ── dunder ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path, label = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    # ── helpers ──────────────────────────────────────────────────────────────

    def class_counts(self) -> dict[str, int]:
        """Return {class_name: count} for this split."""
        counts = {cls: 0 for cls in CLASSES}
        for _, label in self.images:
            counts[CLASSES[label]] += 1
        return counts


# ── Transforms ───────────────────────────────────────────────────────────────

def build_transforms(image_size: int = 128,
                     mean: float = TRAIN_MEAN,
                     std: float  = TRAIN_STD,
                     augment: bool = False) -> transforms.Compose:
    """
    Build a torchvision transform pipeline.

    Args:
        image_size: Target spatial resolution (square).
        mean / std: Pixel-level statistics computed from the training split.
        augment:    If True, add random horizontal flip and small rotation
                    (for training only).
    """
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
    ]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(ops)


# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir:   str,
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_tf = build_transforms(image_size, augment=False)  # keep reproducible
    eval_tf  = build_transforms(image_size, augment=False)

    train_set = ChestXrayDataset(data_dir, "train", train_tf)
    val_set   = ChestXrayDataset(data_dir, "val",   eval_tf)
    test_set  = ChestXrayDataset(data_dir, "test",  eval_tf)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True,  **common)
    val_loader   = DataLoader(val_set,   shuffle=False, **common)
    test_loader  = DataLoader(test_set,  shuffle=False, **common)

    return train_loader, val_loader, test_loader


# ── Data preparation ──────────────────────────────────────────────────────────

def resplit_dataset(
    raw_dir:  str,
    out_dir:  str,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> None:
    """
    Re-stratify the original Kaggle split (80 / 10 / 10) with a fixed seed.

    The original Kaggle split is poorly balanced across train/val/test.
    This function pools all images per class and creates fresh, reproducible
    splits, preserving the class ratio across all three sets.

    Args:
        raw_dir:  Root of the unzipped Kaggle dataset (contains train/val/test).
        out_dir:  Destination root for the re-split dataset.
        train_frac / val_frac: Proportions (test = 1 - train - val).
        seed: Random seed for reproducibility.
    """
    if os.path.exists(out_dir):
        print(f"[INFO] Re-split dataset already exists at '{out_dir}'. Skipping.")
        return

    rng = random.Random(seed)

    for split in ("train", "val", "test"):
        for cls in CLASSES:
            Path(os.path.join(out_dir, split, cls)).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        all_files: list[tuple[str, str]] = []
        for split in ("train", "val", "test"):
            folder = os.path.join(raw_dir, split, cls)
            if os.path.isdir(folder):
                all_files.extend(
                    (f, folder) for f in os.listdir(folder)
                )

        all_files.sort()          # deterministic order before shuffle
        rng.shuffle(all_files)

        n = len(all_files)
        train_end = int(n * train_frac)
        val_end   = int(n * (train_frac + val_frac))

        splits = {
            "train": all_files[:train_end],
            "val":   all_files[train_end:val_end],
            "test":  all_files[val_end:],
        }

        for split_name, file_list in splits.items():
            for fname, src_folder in file_list:
                shutil.copy(
                    os.path.join(src_folder, fname),
                    os.path.join(out_dir, split_name, cls, fname),
                )

    print(f"[INFO] Dataset re-split complete → '{out_dir}'")
