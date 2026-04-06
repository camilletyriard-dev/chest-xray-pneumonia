"""
chest-xray-pneumonia/src — modular deep learning library.
"""

from .dataset  import ChestXrayDataset, build_dataloaders, build_transforms, resplit_dataset
from .models   import BaselineCNN, ImprovedCNN, build_model
from .train    import train, evaluate, train_one_epoch
from .evaluate import full_report, plot_training_curves, plot_confusion_matrix, print_comparison
from .utils    import set_seed, get_device, compute_class_weights, load_checkpoint

__all__ = [
    "ChestXrayDataset", "build_dataloaders", "build_transforms", "resplit_dataset",
    "BaselineCNN", "ImprovedCNN", "build_model",
    "train", "evaluate", "train_one_epoch",
    "full_report", "plot_training_curves", "plot_confusion_matrix", "print_comparison",
    "set_seed", "get_device", "compute_class_weights", "load_checkpoint",
]
