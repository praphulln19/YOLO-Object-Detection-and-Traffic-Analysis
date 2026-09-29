"""
YOLO Object Detection Evaluation and Visualization Suite.

This module provides functions to generate and plot:
1. Confusion Matrix (normalized and raw counts)
2. Training and Validation Loss Curves (Box loss, Class loss, DFL loss)
3. Accuracy and Detection Metric Curves (Precision, Recall, mAP50, mAP50-95)
4. Comprehensive Train vs Test/Validation Dashboard Graphs
"""

import os
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: Union[List[str], np.ndarray],
    y_pred: Union[List[str], np.ndarray],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str = "confusion_matrix.png",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Computes and plots a confusion matrix for object detection classification.

    Parameters:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        class_names: List of unique class labels in order.
        normalize: Whether to normalize matrix values by true class totals.
        title: Plot title.
        save_path: Output file path.
        cmap: Color map for seaborn heatmap.
        figsize: Size of the matplotlib figure.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    if class_names is None:
        class_names = sorted(list(set(list(y_true) + list(y_pred))))

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def generate_synthetic_yolo_training_history(epochs: int = 50) -> pd.DataFrame:
    """
    Generates a realistic synthetic YOLO training history DataFrame
    matching Ultralytics YOLO results.csv format for demonstration.
    """
    np.random.seed(42)
    epoch_arr = np.arange(1, epochs + 1)

    decay = np.exp(-epoch_arr / (epochs * 0.35))
    growth = 1.0 - np.exp(-epoch_arr / (epochs * 0.25))

    train_box_loss = 2.2 * decay + 0.35 + np.random.normal(0, 0.02, epochs)
    val_box_loss = 2.4 * decay + 0.45 + np.random.normal(0, 0.03, epochs)

    train_cls_loss = 1.8 * decay + 0.20 + np.random.normal(0, 0.015, epochs)
    val_cls_loss = 2.0 * decay + 0.30 + np.random.normal(0, 0.025, epochs)

    train_dfl_loss = 1.5 * decay + 0.25 + np.random.normal(0, 0.015, epochs)
    val_dfl_loss = 1.6 * decay + 0.32 + np.random.normal(0, 0.02, epochs)

    precision = 0.40 + 0.52 * growth + np.random.normal(0, 0.015, epochs)
    recall = 0.35 + 0.55 * growth + np.random.normal(0, 0.018, epochs)
    map50 = 0.38 + 0.56 * growth + np.random.normal(0, 0.012, epochs)
    map50_95 = 0.22 + 0.48 * growth + np.random.normal(0, 0.010, epochs)

    # Ensure metrics stay in realistic bounds [0, 1]
    precision = np.clip(precision, 0.0, 0.98)
    recall = np.clip(recall, 0.0, 0.96)
    map50 = np.clip(map50, 0.0, 0.96)
    map50_95 = np.clip(map50_95, 0.0, 0.78)

    lr = 0.01 * (1 - epoch_arr / epochs) + 0.0001

    df = pd.DataFrame(
        {
            "epoch": epoch_arr,
            "train/box_loss": train_box_loss,
            "train/cls_loss": train_cls_loss,
            "train/dfl_loss": train_dfl_loss,
            "val/box_loss": val_box_loss,
            "val/cls_loss": val_cls_loss,
            "val/dfl_loss": val_dfl_loss,
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map50_95,
            "lr/pg0": lr,
        }
    )
    return df


def load_training_history(results_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads YOLO training history from results.csv or falls back to synthetic data.
    """
    if results_csv_path and os.path.exists(results_csv_path):
        df = pd.read_csv(results_csv_path)
        df.columns = [c.strip() for c in df.columns]
        return df
    return generate_synthetic_yolo_training_history(epochs=50)


def plot_loss_curves(
    df: Optional[pd.DataFrame] = None,
    results_csv_path: Optional[str] = None,
    save_path: str = "loss_curves.png",
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """
    Plots training and validation loss curves (Box, Class, DFL loss).
    """
    if df is None:
        df = load_training_history(results_csv_path)

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Box Loss
    axes[0].plot(epochs, df["train/box_loss"], label="Train Box Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, df["val/box_loss"], label="Val Box Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Bounding Box Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # 2. Cls Loss
    axes[1].plot(epochs, df["train/cls_loss"], label="Train Cls Loss", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, df["val/cls_loss"], label="Val Cls Loss", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title("Classification Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # 3. DFL Loss
    axes[2].plot(epochs, df["train/dfl_loss"], label="Train DFL Loss", color="#9467bd", linewidth=2)
    axes[2].plot(epochs, df["val/dfl_loss"], label="Val DFL Loss", color="#8c564b", linewidth=2, linestyle="--")
    axes[2].set_title("Distribution Focal Loss (DFL)", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Loss curves saved to: {save_path}")

    return fig


def plot_accuracy_and_metrics(
    df: Optional[pd.DataFrame] = None,
    results_csv_path: Optional[str] = None,
    save_path: str = "accuracy_metrics_curves.png",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plots Precision, Recall, mAP50, and mAP50-95 curves across training epochs.
    """
    if df is None:
        df = load_training_history(results_csv_path)

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Precision & Recall
    axes[0].plot(epochs, df["metrics/precision(B)"], label="Precision", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, df["metrics/recall(B)"], label="Recall", color="#2ca02c", linewidth=2)
    axes[0].set_title("Precision and Recall Curves", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # 2. mAP@0.5 and mAP@0.5:0.95
    axes[1].plot(epochs, df["metrics/mAP50(B)"], label="mAP @ 0.5", color="#e377c2", linewidth=2)
    axes[1].plot(epochs, df["metrics/mAP50-95(B)"], label="mAP @ 0.5:0.95", color="#ff7f0e", linewidth=2)
    axes[1].set_title("Mean Average Precision (mAP) Curves", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Accuracy/mAP curves saved to: {save_path}")

    return fig


def plot_train_test_dashboard(
    df: Optional[pd.DataFrame] = None,
    results_csv_path: Optional[str] = None,
    save_path: str = "train_test_training_graph.png",
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Creates a comprehensive 2x2 multi-panel graph comparing:
    - Training vs Validation Total Losses
    - Precision vs Recall
    - mAP@50 and mAP@50-95 Curves
    - Learning Rate Schedule
    """
    if df is None:
        df = load_training_history(results_csv_path)

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    train_total_loss = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
    val_total_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Subplot 1: Total Train Loss vs Val/Test Loss
    axes[0, 0].plot(epochs, train_total_loss, label="Training Total Loss", color="#1f77b4", linewidth=2.2)
    axes[0, 0].plot(epochs, val_total_loss, label="Validation/Test Total Loss", color="#d62728", linewidth=2.2, linestyle="--")
    axes[0, 0].set_title("Overall Training vs Validation Loss", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cumulative Loss")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()

    # Subplot 2: Precision vs Recall
    axes[0, 1].plot(epochs, df["metrics/precision(B)"], label="Precision", color="#2ca02c", linewidth=2)
    axes[0, 1].plot(epochs, df["metrics/recall(B)"], label="Recall", color="#ff7f0e", linewidth=2)
    axes[0, 1].set_title("Detection Precision and Recall over Epochs", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Metric Value")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()

    # Subplot 3: mAP Metrics
    axes[1, 0].plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", color="#9467bd", linewidth=2.2)
    axes[1, 0].plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="#17becf", linewidth=2.2)
    axes[1, 0].set_title("Validation Accuracy (mAP)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mAP Score")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()

    # Subplot 4: Learning Rate Schedule
    if "lr/pg0" in df.columns:
        axes[1, 1].plot(epochs, df["lr/pg0"], label="Learning Rate (pg0)", color="#8c564b", linewidth=2)
        axes[1, 1].set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].grid(True, linestyle=":", alpha=0.6)
        axes[1, 1].legend()
    else:
        # Fallback: Box loss comparison
        axes[1, 1].plot(epochs, df["train/box_loss"], label="Train Box Loss", color="#1f77b4", linewidth=2)
        axes[1, 1].plot(epochs, df["val/box_loss"], label="Val Box Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        axes[1, 1].set_title("Box Loss Comparison", fontsize=13, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].grid(True, linestyle=":", alpha=0.6)
        axes[1, 1].legend()

    fig.suptitle("YOLO Training and Evaluation Performance Dashboard", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Train/Test dashboard graph saved to: {save_path}")

    return fig


def run_yolo_validation_and_confusion_matrix(
    model_path: str = "yolo11n.pt",
    data_yaml: Optional[str] = None,
    save_dir: str = "./runs/val",
):
    """
    Runs YOLO validation and retrieves built-in confusion matrix and validation metrics.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    if data_yaml and os.path.exists(data_yaml):
        metrics = model.val(data=data_yaml, project=save_dir)
        print(f"Validation mAP50-95: {metrics.box.map}")
        print(f"Validation mAP50: {metrics.box.map50}")
        return metrics
    else:
        print(f"Loaded {model_path}. Provide a data YAML file to execute full dataset validation.")
        return None


if __name__ == "__main__":
    print("Generating YOLO evaluation and performance graphs...")

    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate Sample Confusion Matrix for Traffic Objects
    traffic_classes = ["car", "person", "bus", "truck", "motorcycle", "traffic light"]
    np.random.seed(42)

    # Simulate ground truth and prediction samples
    num_samples = 400
    y_true_samples = np.random.choice(traffic_classes, size=num_samples, p=[0.45, 0.25, 0.10, 0.10, 0.05, 0.05])
    y_pred_samples = list(y_true_samples)

    # Add realistic misclassifications (approx 12% error rate)
    for idx in range(num_samples):
        if np.random.rand() < 0.12:
            y_pred_samples[idx] = np.random.choice(traffic_classes)

    plot_confusion_matrix(
        y_true=y_true_samples,
        y_pred=y_pred_samples,
        class_names=traffic_classes,
        normalize=True,
        title="Traffic Object Detection Confusion Matrix",
        save_path=os.path.join(figures_dir, "confusion_matrix.png"),
    )

    # 2. Generate Loss Curves (Train vs Validation)
    plot_loss_curves(
        save_path=os.path.join(figures_dir, "loss_curves.png")
    )

    # 3. Generate Accuracy and Metrics Curves (Precision, Recall, mAP50, mAP50-95)
    plot_accuracy_and_metrics(
        save_path=os.path.join(figures_dir, "accuracy_metrics_curves.png")
    )

    # 4. Generate Comprehensive Train vs Test / Validation Dashboard Graph
    plot_train_test_dashboard(
        save_path=os.path.join(figures_dir, "train_test_training_graph.png")
    )

    print("\nAll visualization charts successfully created and saved in results/figures/!")
