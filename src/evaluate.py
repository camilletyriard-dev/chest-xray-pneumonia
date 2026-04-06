"""
evaluate.py — Evaluation, visualisation, and result reporting utilities.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import evaluate


# ── Confusion matrix ──────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (all_true_labels, all_predicted_labels) as NumPy arrays."""
    model.eval()
    all_labels, all_preds = [], []
    for images, labels in loader:
        images = images.to(device)
        preds  = model(images).argmax(dim=1).cpu()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list[str] = ("NORMAL", "PNEUMONIA"),
    title:       str       = "Confusion Matrix",
    save_path:   str | None = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True",      fontsize=12)
    plt.title(title,        fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history:    dict[str, list],
    title:      str       = "Training Curves",
    baseline:   dict | None = None,
    save_path:  str | None  = None,
) -> None:
    """
    Four-panel training-curve plot.

    Args:
        history:   Dict returned by ``train.train()``.
        title:     Figure suptitle.
        baseline:  Optional dict with keys ``val_acc``, ``normal_acc``,
                   ``pneumonia_acc`` (scalar floats) drawn as dashed reference.
        save_path: If given, save the figure here.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ── Loss ─────────────────────────────────────────────────────────────────
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train",      lw=2)
    axes[0, 0].plot(epochs, history["val_loss"],   "r-", label="Validation", lw=2)
    axes[0, 0].set(title="Loss",     xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # ── Overall accuracy ─────────────────────────────────────────────────────
    axes[0, 1].plot(epochs, [a * 100 for a in history["train_acc"]], "b-", label="Train",      lw=2)
    axes[0, 1].plot(epochs, [a * 100 for a in history["val_acc"]],   "r-", label="Validation", lw=2)
    if baseline:
        axes[0, 1].axhline(
            baseline["val_acc"] * 100, color="grey", ls="--", lw=1.5,
            label=f"Baseline ({baseline['val_acc']*100:.2f}%)", alpha=0.7,
        )
    axes[0, 1].set(title="Overall Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # ── Training class accuracy ───────────────────────────────────────────────
    axes[1, 0].plot(epochs, [a * 100 for a in history["train_normal_acc"]],    "g-", label="NORMAL",    lw=2)
    axes[1, 0].plot(epochs, [a * 100 for a in history["train_pneumonia_acc"]], "m-", label="PNEUMONIA", lw=2)
    axes[1, 0].set(title="Training Class-wise Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # ── Validation class accuracy ─────────────────────────────────────────────
    axes[1, 1].plot(epochs, [a * 100 for a in history["val_normal_acc"]],    "g-", label="NORMAL",    lw=2)
    axes[1, 1].plot(epochs, [a * 100 for a in history["val_pneumonia_acc"]], "m-", label="PNEUMONIA", lw=2)
    if baseline:
        axes[1, 1].axhline(baseline["normal_acc"] * 100,    color="darkgreen",   ls="--", lw=1.5,
                           label=f"Baseline N ({baseline['normal_acc']*100:.1f}%)", alpha=0.6)
        axes[1, 1].axhline(baseline["pneumonia_acc"] * 100, color="darkmagenta", ls="--", lw=1.5,
                           label=f"Baseline P ({baseline['pneumonia_acc']*100:.1f}%)", alpha=0.6)
    axes[1, 1].set(title="Validation Class-wise Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Full evaluation report ────────────────────────────────────────────────────

def full_report(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    split_name:  str       = "test",
    class_names: list[str] = ("NORMAL", "PNEUMONIA"),
    cm_title:    str | None = None,
    cm_save_path: str | None = None,
) -> dict[str, float]:
    """
    Print a formatted performance report and plot a confusion matrix.

    Returns:
        dict with ``loss``, ``acc``, ``normal_acc``, ``pneumonia_acc``.
    """
    loss, acc, class_acc = evaluate(model, loader, criterion, device)
    y_true, y_pred       = get_predictions(model, loader, device)

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET RESULTS")
    print(f"{'='*60}")
    print(f"  Loss:           {loss:.4f}")
    print(f"  Accuracy:       {acc*100:.2f}%")
    print(f"  NORMAL:         {class_acc[0]*100:.2f}%")
    print(f"  PNEUMONIA:      {class_acc[1]*100:.2f}%")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        title=cm_title or f"{split_name.capitalize()} Confusion Matrix",
        save_path=cm_save_path,
    )

    return {
        "loss":          loss,
        "acc":           acc,
        "normal_acc":    class_acc[0].item(),
        "pneumonia_acc": class_acc[1].item(),
    }


# ── Comparison table ─────────────────────────────────────────────────────────

def print_comparison(results: dict[str, dict]) -> None:
    """
    Print a markdown-style comparison table.

    Args:
        results: ``{run_name: {"acc": …, "normal_acc": …, "pneumonia_acc": …}}``
    """
    header = f"{'Model':<30} {'Overall':>10} {'NORMAL':>10} {'PNEUMONIA':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<30} {r['acc']*100:>9.2f}%"
            f" {r['normal_acc']*100:>9.2f}%"
            f" {r['pneumonia_acc']*100:>11.2f}%"
        )
    print("=" * len(header))
