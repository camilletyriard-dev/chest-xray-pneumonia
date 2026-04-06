# Chest X-ray Pneumonia Classification

> **Deep learning for clinical decision support under class imbalance**
> — ablation study of Batch Normalization, Adam optimisation, and class-weighted loss.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)

---

## Overview

This project develops and systematically evaluates convolutional neural networks for binary pneumonia detection from chest X-rays, using the [Kaggle Chest X-Ray (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset.

The core research question is: **how do individual design choices — class-weighted loss, batch normalisation, and the Adam optimiser — each contribute to performance under realistic clinical class imbalance?** Each improvement is ablated independently before being combined into a final model.

### Key results

| Model | Val Acc | NORMAL Acc | PNEUMONIA Acc |
|---|---|---|---|
| Baseline CNN (SGD) | 96.07% | 90.51% | 97.90% |
| + Adam only | ~96.5% | — | — |
| + BatchNorm only | 97.95% | — | — |
| + Class weights only | 93.16% | 97.47% | — |
| **Improved CNN (all three)** | **97.61%** | **91.19%** | **97.90%** |
| Improved CNN (test set) | ~96.8% | ~91.2% | ~97.9% |

> **Clinical note:** the class-weights experiment reveals a core tension in medical AI — optimising for the minority class (NORMAL / true negative) at the cost of overall accuracy. In a screening context, this trade-off requires explicit clinical input, not just a loss-function default.

---

## Dataset & re-splitting

The original Kaggle split is poorly stratified (only 16 validation images). We re-split the full corpus into **80 / 10 / 10** (train / val / test) with a fixed seed, preserving the ~74% PNEUMONIA / ~26% NORMAL class ratio across all three sets.

```
Full dataset:  5,856 images
  Train:  4,684  (NORMAL: 1,213 | PNEUMONIA: 3,471)
  Val:      585  (NORMAL:   152 | PNEUMONIA:   433)
  Test:     587  (NORMAL:   154 | PNEUMONIA:   433)
```

---

## Architecture

### BaselineCNN
```
Input (1 × 128 × 128)
  └─ Conv(1→16, 3×3) → ReLU → MaxPool(2×2)    # → 16 × 64 × 64
  └─ Conv(16→32, 3×3) → ReLU → MaxPool(2×2)   # → 32 × 32 × 32
  └─ Conv(32→64, 3×3) → ReLU → MaxPool(2×2)   # → 64 × 16 × 16
  └─ Flatten → Linear(16384, 2)
```

Trained with **SGD (lr=0.001)** and **CrossEntropyLoss** for 30 epochs.

### ImprovedCNN
Identical backbone with **Batch Normalization** inserted after every conv layer. Trained with **Adam (lr=0.001)** and **class-weighted CrossEntropyLoss**.

---

## Installation

```bash
git clone https://github.com/camilletyriard/chest-xray-pneumonia.git
cd chest-xray-pneumonia
pip install -r requirements.txt
```

---

## Data download

1. Create a [Kaggle API token](https://www.kaggle.com/settings) and place `kaggle.json` in `~/.config/kaggle/`.
2. Download and unzip the dataset:

```bash
kaggle datasets download paultimothymooney/chest-xray-pneumonia \
    --unzip -p data/chest_xray
```

---

## Quick start

```bash
# 1. Prepare data (re-split into 80/10/10)
python run_pipeline.py --raw_dir data/chest_xray --run data

# 2. Train baseline + improved + evaluate test set
python run_pipeline.py --data_dir data/chest_xray_split --run all

# 3. Individual stages
python run_pipeline.py --data_dir data/chest_xray_split --run baseline
python run_pipeline.py --data_dir data/chest_xray_split --run improved
python run_pipeline.py --data_dir data/chest_xray_split --run evaluate
```

Results (checkpoints + plots) are saved to `results/`.

### Optional: Weights & Biases logging

```bash
pip install wandb
wandb login
python run_pipeline.py --data_dir data/chest_xray_split --run all --use_wandb
```

---

## Repository structure

```
chest-xray-pneumonia/
├── run_pipeline.py          # End-to-end pipeline (entry point)
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── dataset.py           # ChestXrayDataset, DataLoaders, re-split logic
│   ├── models.py            # BaselineCNN, ImprovedCNN, build_model()
│   ├── train.py             # train(), evaluate(), train_one_epoch()
│   ├── evaluate.py          # full_report(), confusion matrix, curve plots
│   └── utils.py             # set_seed(), compute_class_weights(), load_checkpoint()
│
└── results/                 # Checkpoints and figures (git-ignored)
    ├── baseline_best.pth
    ├── improved_best.pth
    ├── baseline_curves.png
    ├── improved_curves.png
    └── *_test_cm.png
```

---

## Reproducing the ablation study

Each single-improvement ablation can be run by adjusting the training config:

```python
from src import set_seed, get_device, build_dataloaders, BaselineCNN, ImprovedCNN, train
import torch.nn as nn

set_seed(42)
device = get_device()
train_loader, val_loader, _ = build_dataloaders("data/chest_xray_split")

# Ablation A: Adam only (SimpleModel + Adam)
model     = BaselineCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history   = train(model, train_loader, val_loader, criterion, optimizer, device=device)

# Ablation B: BatchNorm only (ImprovedModel + SGD, no class weights)
model     = ImprovedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
history   = train(model, train_loader, val_loader, criterion, optimizer, device=device)
```

---

## Design choices & clinical discussion

### Class imbalance (~74% pneumonia)
The dataset reflects real screening distributions. Three strategies were evaluated:
- **No correction (baseline):** High overall accuracy; NORMAL recall suppressed (minority class under-detected — a dangerous false negative in screening).
- **Class weights:** Dramatically improved NORMAL recall (+7pp) at the cost of overall accuracy (−3pp) and increased PNEUMONIA false positives. Reveals the inherent accuracy–equity trade-off.
- **BatchNorm + Adam:** Improved both overall accuracy and minority-class accuracy without the explicit fairness penalty.

### Batch Normalization
Normalising activations per mini-batch accelerates convergence, acts as a mild regulariser, and reduces sensitivity to weight initialisation — explaining its consistent +1.9pp gain over the baseline.

### Adam vs SGD
Adam's adaptive per-parameter learning rates allow faster convergence on this relatively small dataset, explaining the improved validation stability without hyperparameter tuning.

---

## Citation

```bibtex
@misc{tyriard2025chestxray,
  author       = {Tyriard, Camille},
  title        = {Chest X-ray Pneumonia Classification: Ablation of BatchNorm, Adam, and Class Weights},
  year         = {2025},
  institution  = {University College London},
  note         = {MSc Artificial Intelligence, Deep Learning coursework}
}
```
