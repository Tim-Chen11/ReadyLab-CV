import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing
    Helps prevent overconfidence in predictions
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses learning on hard examples
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MixUpCrossEntropy(nn.Module):
    """
    Loss function for MixUp augmentation
    Computes loss for mixed targets
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, y_a, y_b, lam):
        loss_a = F.cross_entropy(pred, y_a)
        loss_b = F.cross_entropy(pred, y_b)
        return lam * loss_a + (1 - lam) * loss_b


class CutMixCrossEntropy(nn.Module):
    """
    Loss function for CutMix augmentation
    Similar to MixUp but for CutMix
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, y_a, y_b, lam):
        loss_a = F.cross_entropy(pred, y_a)
        loss_b = F.cross_entropy(pred, y_b)
        return lam * loss_a + (1 - lam) * loss_b


class WeightedCrossEntropy(nn.Module):
    """
    Weighted cross entropy for handling class imbalance
    """

    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        if self.class_weights is not None:
            weight = self.class_weights.to(inputs.device)
        else:
            weight = None
        return F.cross_entropy(inputs, targets, weight=weight)


def get_loss_function(loss_name='cross_entropy', **kwargs):
    """
    Factory function to get loss function by name

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance
    """
    loss_dict = {
        'cross_entropy': nn.CrossEntropyLoss,
        'label_smoothing': LabelSmoothingCrossEntropy,
        'focal': FocalLoss,
        'weighted_ce': WeightedCrossEntropy,
        'mixup': MixUpCrossEntropy,
        'cutmix': CutMixCrossEntropy,
    }

    if loss_name not in loss_dict:
        raise ValueError(f"Unknown loss function: {loss_name}")

    return loss_dict[loss_name](**kwargs)


# Example usage for calculating class weights
def calculate_class_weights(dataset):
    """
    Calculate class weights for weighted loss

    Args:
        dataset: Dataset instance with label information

    Returns:
        torch.Tensor: Class weights
    """
    from collections import Counter

    # Count labels
    labels = []
    for _, label, _ in dataset:
        labels.append(label)

    # Calculate weights (inverse frequency)
    label_counts = Counter(labels)
    total = len(labels)
    num_classes = len(label_counts)

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)