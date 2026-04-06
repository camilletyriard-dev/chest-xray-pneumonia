#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end chest X-ray pneumonia classification pipeline.

Usage
-----
    # Full pipeline (data prep → baseline → improved → final evaluation)
    python run_pipeline.py --data_dir data/chest_xray_split --run all

    # Individual stages
    python run_pipeline.py --data_dir data/chest_xray_split --run baseline
    python run_pipeline.py --data_dir data/chest_xray_split --run improved
    python run_pipeline.py --data_dir data/chest_xray_split --run evaluate

    # Enable Weights & Biases logging
    python run_pipeline.py --data_dir data/chest_xray_split --run all --use_wandb

Pipeline stages
---------------
1. Data preparation  — re-split Kaggle data into reproducible 80/10/10 splits.
2. Baseline training — 3-layer CNN with SGD and standard cross-entropy.
3. Improved training — BatchNorm + Adam + class-weighted loss.
4. Final evaluation  — test-set report comparing both models.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src import (
    set_seed, get_device,
    resplit_dataset, build_dataloaders,
    BaselineCNN, ImprovedCNN,
    train, full_report, plot_training_curves,
    compute_class_weights, load_checkpoint, print_comparison,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chest X-ray pneumonia classifier")
    p.add_argument("--data_dir",    default="data/chest_xray_split",
                   help="Root of the re-split dataset (will be created if absent).")
    p.add_argument("--raw_dir",     default="data/chest_xray",
                   help="Root of the original Kaggle dataset (only needed for --run data).")
    p.add_argument("--output_dir",  default="results",
                   help="Directory for checkpoints and figures.")
    p.add_argument("--run", choices=["data", "baseline", "improved", "evaluate", "all"],
                   default="all")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--image_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--use_wandb",   action="store_true")
    return p.parse_args()


# ── Stage helpers ─────────────────────────────────────────────────────────────

def stage_data(args) -> None:
    print("\n" + "="*60)
    print("  STAGE 1 — DATA PREPARATION")
    print("="*60)
    resplit_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.data_dir,
        train_frac=0.80,
        val_frac=0.10,
        seed=args.seed,
    )


def stage_baseline(args, device: torch.device) -> dict:
    print("\n" + "="*60)
    print("  STAGE 2 — BASELINE CNN  (SGD, standard CE)")
    print("="*60)

    train_loader, val_loader, _ = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size
    )

    model     = BaselineCNN(image_size=args.image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(args.output_dir, "baseline_best.pth")
    history   = train(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=args.epochs, device=device,
        checkpoint_path=ckpt_path, use_wandb=args.use_wandb,
    )

    plot_training_curves(
        history, title="Baseline CNN — Training Curves",
        save_path=os.path.join(args.output_dir, "baseline_curves.png"),
    )
    return history


def stage_improved(args, device: torch.device, baseline_history: dict | None = None) -> dict:
    print("\n" + "="*60)
    print("  STAGE 3 — IMPROVED CNN  (Adam + BatchNorm + class weights)")
    print("="*60)

    train_loader, val_loader, _ = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size
    )

    print("\n[class weights]")
    weights   = compute_class_weights(args.data_dir, device=device)
    model     = ImprovedCNN(image_size=args.image_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(args.output_dir, "improved_best.pth")

    baseline_ref = None
    if baseline_history:
        best_idx   = baseline_history["val_loss"].index(min(baseline_history["val_loss"]))
        baseline_ref = {
            "val_acc":       baseline_history["val_acc"][best_idx],
            "normal_acc":    baseline_history["val_normal_acc"][best_idx],
            "pneumonia_acc": baseline_history["val_pneumonia_acc"][best_idx],
        }

    history = train(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=args.epochs, device=device,
        checkpoint_path=ckpt_path, use_wandb=args.use_wandb,
    )

    plot_training_curves(
        history,
        title="Improved CNN — Training Curves (BatchNorm + Adam + Class Weights)",
        baseline=baseline_ref,
        save_path=os.path.join(args.output_dir, "improved_curves.png"),
    )
    return history


def stage_evaluate(args, device: torch.device) -> None:
    print("\n" + "="*60)
    print("  STAGE 4 — FINAL TEST-SET EVALUATION")
    print("="*60)

    _, _, test_loader = build_dataloaders(args.data_dir, args.image_size, args.batch_size)
    weights = compute_class_weights(args.data_dir, device=device)

    all_results: dict[str, dict] = {}

    for model_name, ModelClass, criterion_cls, criterion_kwargs in [
        ("Baseline CNN",  BaselineCNN, nn.CrossEntropyLoss, {}),
        ("Improved CNN",  ImprovedCNN, nn.CrossEntropyLoss, {"weight": weights}),
    ]:
        ckpt_path = os.path.join(
            args.output_dir,
            f"{'baseline' if 'Baseline' in model_name else 'improved'}_best.pth"
        )
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path} — skipping {model_name}")
            continue

        model = ModelClass(image_size=args.image_size)
        load_checkpoint(model, ckpt_path, device)

        criterion = criterion_cls(**criterion_kwargs)
        results   = full_report(
            model, test_loader, criterion, device,
            split_name=model_name,
            cm_title=f"{model_name} — Test Confusion Matrix",
            cm_save_path=os.path.join(
                args.output_dir,
                f"{'baseline' if 'Baseline' in model_name else 'improved'}_test_cm.png"
            ),
        )
        all_results[model_name] = results

    if len(all_results) >= 2:
        print("\n[SUMMARY COMPARISON]")
        print_comparison(all_results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    run = args.run
    baseline_history = None

    if run in ("data", "all"):
        stage_data(args)

    if run in ("baseline", "all"):
        baseline_history = stage_baseline(args, device)

    if run in ("improved", "all"):
        stage_improved(args, device, baseline_history)

    if run in ("evaluate", "all"):
        stage_evaluate(args, device)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
