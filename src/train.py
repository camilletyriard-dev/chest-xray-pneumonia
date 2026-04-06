"""
train.py — Training and validation loops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Single-epoch helpers ──────────────────────────────────────────────────────

def _class_accuracy(
    preds:  torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """Return per-class accuracy as a 1-D tensor of length ``num_classes``."""
    correct = torch.zeros(num_classes, device=preds.device)
    total   = torch.zeros(num_classes, device=preds.device)
    for c in range(num_classes):
        mask = labels == c
        total[c]   += mask.sum()
        correct[c] += ((preds == labels) & mask).sum()
    return correct / total.clamp(min=1)


def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
) -> tuple[float, float, torch.Tensor]:
    """
    Run one full training pass.

    Returns:
        (epoch_loss, epoch_accuracy, class_accuracy_tensor)
    """
    model.train()
    total_loss = total_correct = total_samples = 0
    all_preds = all_labels = None

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds   = logits.argmax(dim=1)
        n       = images.size(0)
        total_samples  += n
        total_loss     += loss.item() * n
        total_correct  += (preds == labels).sum().item()

        all_preds  = preds  if all_preds  is None else torch.cat([all_preds,  preds])
        all_labels = labels if all_labels is None else torch.cat([all_labels, labels])

    class_acc = _class_accuracy(all_preds, all_labels)
    return total_loss / total_samples, total_correct / total_samples, class_acc


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, torch.Tensor]:
    """
    Evaluate model on a DataLoader.

    Returns:
        (loss, accuracy, class_accuracy_tensor)
    """
    model.eval()
    total_loss = total_correct = total_samples = 0
    all_preds = all_labels = None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        preds  = logits.argmax(dim=1)
        n      = images.size(0)
        total_samples += n
        total_loss    += loss.item() * n
        total_correct += (preds == labels).sum().item()

        all_preds  = preds  if all_preds  is None else torch.cat([all_preds,  preds])
        all_labels = labels if all_labels is None else torch.cat([all_labels, labels])

    class_acc = _class_accuracy(all_preds, all_labels)
    return total_loss / total_samples, total_correct / total_samples, class_acc


# ── Full training loop ────────────────────────────────────────────────────────

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    criterion:    nn.Module,
    optimizer:    torch.optim.Optimizer,
    num_epochs:   int   = 30,
    device:       torch.device | str = "cpu",
    checkpoint_path: str | None = None,
    use_wandb:    bool  = False,
) -> dict[str, list]:
    """
    Full training loop with best-model checkpointing.

    Args:
        model:            The network to train (moved to ``device`` in-place).
        train_loader:     Training DataLoader.
        val_loader:       Validation DataLoader.
        criterion:        Loss function.
        optimizer:        Optimiser.
        num_epochs:       Number of full passes over the training data.
        device:           ``"cuda"`` / ``"cpu"`` or a ``torch.device``.
        checkpoint_path:  If given, save the best checkpoint here.
        use_wandb:        Log metrics to Weights & Biases.

    Returns:
        History dict with keys::

            train_loss, train_acc,
            val_loss,   val_acc,
            train_normal_acc, train_pneumonia_acc,
            val_normal_acc,   val_pneumonia_acc
    """
    device = torch.device(device)
    model.to(device)

    history: dict[str, list] = {k: [] for k in (
        "train_loss", "train_acc", "val_loss", "val_acc",
        "train_normal_acc", "train_pneumonia_acc",
        "val_normal_acc",   "val_pneumonia_acc",
    )}
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc, tr_cls = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_cls = evaluate(model, val_loader, criterion, device)

        # --- store -------------------------------------------------------
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["train_normal_acc"].append(tr_cls[0].item())
        history["train_pneumonia_acc"].append(tr_cls[1].item())
        history["val_normal_acc"].append(vl_cls[0].item())
        history["val_pneumonia_acc"].append(vl_cls[1].item())

        # --- print -------------------------------------------------------
        print(
            f"Epoch {epoch:>3}/{num_epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.2f}%  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.2f}%  "
            f"[N={vl_cls[0]*100:.1f}%  P={vl_cls[1]*100:.1f}%]"
        )

        # --- checkpoint --------------------------------------------------
        if vl_loss < best_val_loss and checkpoint_path:
            best_val_loss = vl_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": vl_loss,
                    "val_acc":  vl_acc,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best checkpoint (val_loss={vl_loss:.4f})")

        # --- wandb -------------------------------------------------------
        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss":          tr_loss,
                    "train/acc":           tr_acc,
                    "train/normal_acc":    tr_cls[0].item(),
                    "train/pneumonia_acc": tr_cls[1].item(),
                    "val/loss":            vl_loss,
                    "val/acc":             vl_acc,
                    "val/normal_acc":      vl_cls[0].item(),
                    "val/pneumonia_acc":   vl_cls[1].item(),
                })
            except ImportError:
                pass

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return history
