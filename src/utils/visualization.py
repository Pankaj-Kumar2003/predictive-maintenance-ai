"""
visualization.py
────────────────
Visualization utilities for model evaluation and exploratory data analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot multi-task training and validation loss curves and F1 metrics."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # F1 score curve
    if "val_f1" in history:
        axes[1].plot(epochs, history["val_f1"], label="Val F1 (Failure)", color="#2ca02c", lw=2)
        axes[1].set_title("Validation F1 Score")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot normalized or raw confusion matrix for binary failure detection."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Failure"])
    ax.set_yticklabels(["Normal", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot ROC Curve with calculated AUC."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Failure Detection")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_rul_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_id: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot true vs predicted Remaining Useful Life (RUL) trajectory."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_true, label="True RUL", color="black", linestyle="--", lw=2)
    ax.plot(y_pred, label="Predicted RUL", color="#d62728", lw=2)
    title = f"RUL Prediction Trajectory (Unit {unit_id})" if unit_id else "RUL Prediction Trajectory"
    ax.set_title(title)
    ax.set_xlabel("Cycle / Window Index")
    ax.set_ylabel("Remaining Useful Life (Cycles)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"RUL plot saved to {save_path}")
    plt.close()
