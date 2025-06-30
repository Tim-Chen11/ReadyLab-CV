import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)


class MetricTracker:
    """Track and compute metrics during training"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, name: str, value: float):
        """Update a metric with a new value"""
        self.metrics[name].append(value)

    def avg(self, name: str) -> float:
        """Get average value of a metric"""
        values = self.metrics.get(name, [])
        return np.mean(values) if values else 0.0

    def get_averages(self) -> Dict[str, float]:
        """Get all metric averages"""
        return {name: self.avg(name) for name in self.metrics}

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int = 5,
        class_names: Optional[List[str]] = None
) -> Dict:
    """
    Calculate comprehensive metrics for classification

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        class_names: Names of classes for report

    Returns:
        Dictionary containing various metrics
    """
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred)
    }

    # Per-class metrics
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f'class_{i}'
        mask = y_true == i

        if mask.sum() > 0:  # Only calculate if class exists in true labels
            metrics[f'{class_name}_precision'] = precision_score(y_true == i, y_pred == i, zero_division=0)
            metrics[f'{class_name}_recall'] = recall_score(y_true == i, y_pred == i, zero_division=0)
            metrics[f'{class_name}_f1'] = f1_score(y_true == i, y_pred == i, zero_division=0)
            metrics[f'{class_name}_support'] = mask.sum()

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Classification report as string
    if class_names:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names
        )

    return metrics


def top_k_accuracy(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5
) -> float:
    """
    Calculate top-k accuracy

    Args:
        outputs: Model outputs (logits)
        targets: True labels
        k: k value for top-k accuracy

    Returns:
        Top-k accuracy as percentage
    """
    with torch.no_grad():
        batch_size = targets.size(0)

        # Get top k predictions
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()

        # Compare with targets
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        # Calculate accuracy
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return (correct_k.mul_(100.0 / batch_size)).item()


def calculate_class_weights(
        labels: List[int],
        num_classes: int,
        method: str = 'inverse_frequency'
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets

    Args:
        labels: List of labels in dataset
        num_classes: Total number of classes
        method: Weighting method ('inverse_frequency' or 'effective_number')

    Returns:
        Tensor of class weights
    """
    # Count occurrences
    counts = np.bincount(labels, minlength=num_classes)

    if method == 'inverse_frequency':
        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes

    elif method == 'effective_number':
        # Effective number of samples
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
        weights = weights / weights.sum() * num_classes

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """Early stopping helper"""

    def __init__(
            self,
            patience: int = 10,
            min_delta: float = 0.0,
            mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training

        Args:
            score: Current score to check

        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset the early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate for all parameter groups"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr