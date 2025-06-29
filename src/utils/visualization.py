import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
from typing import List, Dict, Optional, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_training_curves(
        metrics_history: Dict[str, List[Dict]],
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """
    Plot training and validation curves

    Args:
        metrics_history: Dictionary with 'train' and 'val' metrics
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract data
    train_metrics = pd.DataFrame(metrics_history['train'])
    val_metrics = pd.DataFrame(metrics_history['val'])

    # Plot loss
    ax = axes[0, 0]
    ax.plot(train_metrics['epoch'], train_metrics['loss'], label='Train', marker='o')
    ax.plot(val_metrics['epoch'], val_metrics['loss'], label='Validation', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot accuracy
    ax = axes[0, 1]
    ax.plot(train_metrics['epoch'], train_metrics['accuracy'] * 100, label='Train', marker='o')
    ax.plot(val_metrics['epoch'], val_metrics['accuracy'] * 100, label='Validation', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot learning rate
    if 'lr' in train_metrics.columns:
        ax = axes[1, 0]
        ax.plot(train_metrics['epoch'], train_metrics['lr'], marker='o', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # Plot train vs val accuracy gap
    ax = axes[1, 1]
    accuracy_gap = train_metrics['accuracy'] * 100 - val_metrics['accuracy'] * 100
    ax.plot(train_metrics['epoch'], accuracy_gap, marker='o', color='red')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train - Val Accuracy (%)')
    ax.set_title('Generalization Gap')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: Optional[Path] = None,
        show: bool = True,
        normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
        show: Whether to display the plot
        normalize: Whether to normalize the matrix

    Returns:
        Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Count' if not normalize else 'Percentage'},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_class_distribution(
        labels: List[int],
        class_names: List[str],
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """Plot distribution of classes in dataset"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)

    # Create bar plot
    bars = ax.bar(class_names, counts, color='skyblue', edgecolor='navy')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}\n({count / len(labels) * 100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_predictions(
        images: torch.Tensor,
        true_labels: List[int],
        pred_labels: List[int],
        class_names: List[str],
        num_images: int = 16,
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """
    Visualize grid of images with predictions

    Args:
        images: Tensor of images (N, C, H, W)
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: Names of classes
        num_images: Number of images to show
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    num_images = min(num_images, len(images))
    cols = int(np.sqrt(num_images))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten() if num_images > 1 else [axes]

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    for i in range(num_images):
        ax = axes[i]

        # Convert to numpy and transpose
        img = images[i].cpu().numpy().transpose(1, 2, 0)

        ax.imshow(img)
        ax.axis('off')

        # Add labels
        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_labels[i]]

        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f'True: {true_label}\nPred: {pred_label}'
        ax.set_title(title, fontsize=10, color=color)

    # Hide empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_model_comparison(
        results_dict: Dict[str, Dict[str, float]],
        metric: str = 'accuracy',
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """
    Compare multiple models

    Args:
        results_dict: Dictionary mapping model names to their results
        metric: Metric to compare
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results_dict.keys())
    values = [results_dict[model][metric] * 100 for model in models]

    bars = ax.bar(models, values, color='lightblue', edgecolor='navy')

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()} (%)', fontsize=12)
    ax.set_title(f'Model Comparison - {metric.capitalize()}', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    # Rotate x labels if many models
    if len(models) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def create_experiment_summary_plot(
        exp_dir: Path,
        save_path: Optional[Path] = None,
        show: bool = True
) -> plt.Figure:
    """Create a comprehensive summary plot for an experiment"""
    # Load metrics
    metrics_path = exp_dir / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    import json
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Create subplots
    fig = plt.figure(figsize=(15, 10))

    # Training curves
    ax1 = plt.subplot(2, 2, 1)
    train_metrics = pd.DataFrame(metrics['train'])
    val_metrics = pd.DataFrame(metrics['val'])

    ax1.plot(train_metrics['epoch'], train_metrics['loss'], label='Train Loss')
    ax1.plot(val_metrics['epoch'], val_metrics['loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(train_metrics['epoch'], train_metrics['accuracy'] * 100, label='Train Acc')
    ax2.plot(val_metrics['epoch'], val_metrics['accuracy'] * 100, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Best metrics summary
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')

    best_val_acc = val_metrics['accuracy'].max() * 100
    best_epoch = val_metrics['accuracy'].idxmax()
    final_train_acc = train_metrics['accuracy'].iloc[-1] * 100

    summary_text = f"""
    Best Validation Accuracy: {best_val_acc:.2f}%
    Best Epoch: {best_epoch}
    Final Train Accuracy: {final_train_acc:.2f}%
    Overfitting Gap: {final_train_acc - best_val_acc:.2f}%
    Total Epochs: {len(train_metrics)}
    """

    ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_title('Training Summary')

    plt.suptitle(f'Experiment: {exp_dir.name}', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig