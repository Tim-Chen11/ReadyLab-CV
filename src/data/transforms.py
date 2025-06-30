from torchvision import transforms
import torch
from typing import Tuple


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get training data augmentation pipeline

    Args:
        input_size: Target image size

    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        # Resize with some randomness
        transforms.RandomResizedCrop(
            input_size,
            scale=(0.8, 1.0),  # Zoom in/out
            ratio=(0.9, 1.1),  # Aspect ratio variation
        ),

        # Horizontal flip (makes sense for products)
        transforms.RandomHorizontalFlip(p=0.5),

        # Small rotation (products might be slightly tilted)
        transforms.RandomRotation(degrees=10),

        # Color augmentation (different lighting conditions)
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        # Random perspective (simulates different camera angles)
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation)

    Args:
        input_size: Target image size

    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        # Center crop after resize
        transforms.Resize(int(input_size * 1.14)),  # Resize to slightly larger
        transforms.CenterCrop(input_size),  # Then center crop

        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get inference transforms (same as validation)
    """
    return get_val_transforms(input_size)


# Model-specific transform configurations
MODEL_CONFIGS = {
    'efficientnet-b0': {'input_size': 224},
    'efficientnet-b1': {'input_size': 240},
    'efficientnet-b2': {'input_size': 260},
    'efficientnet-b3': {'input_size': 300},
    'efficientnet-b4': {'input_size': 380},
    'efficientnet-b5': {'input_size': 456},
    'efficientnet-b6': {'input_size': 528},
    'efficientnet-b7': {'input_size': 600},
    'convnext-tiny': {'input_size': 224},
    'convnext-small': {'input_size': 224},
    'convnext-base': {'input_size': 224},
    'convnext-large': {'input_size': 224},
    'convnext-tiny-384': {'input_size': 384},
    'resnet18': {'input_size': 224},
    'resnet34': {'input_size': 224},
    'resnet50': {'input_size': 224},
    'resnet101': {'input_size': 224},
    'resnet152': {'input_size': 224},
    'resnext50': {'input_size': 224},
    'resnext101': {'input_size': 224},
}


def get_transforms_for_model(model_name: str, is_training: bool = True) -> transforms.Compose:
    """
    Get appropriate transforms for a specific model

    Args:
        model_name: Name of the model
        is_training: Whether to include augmentations

    Returns:
        Transform pipeline
    """
    config = MODEL_CONFIGS.get(model_name, {'input_size': 224})
    input_size = config['input_size']

    if is_training:
        return get_train_transforms(input_size)
    else:
        return get_val_transforms(input_size)


class MixUpTransform:
    """
    MixUp augmentation for training
    Reference: https://arxiv.org/abs/1710.09412
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch

        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)

        Returns:
            mixed_images, labels_a, labels_b, lam
        """
        batch_size = images.size(0)

        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()

        # Random shuffle for mixing
        index = torch.randperm(batch_size)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]

        # Return mixed images and both label sets
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


class CutMixTransform:
    """
    CutMix augmentation for training
    Reference: https://arxiv.org/abs/1905.04899
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch
        """
        batch_size, _, height, width = images.size()

        # Sample lambda
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()

        # Random shuffle for mixing
        index = torch.randperm(batch_size)

        # Create random box
        cut_ratio = torch.sqrt(1 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        # Random center point
        cx = torch.randint(width, (1,)).item()
        cy = torch.randint(height, (1,)).item()

        # Box boundaries
        x1 = max(0, cx - cut_w // 2)
        x2 = min(width, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(height, cy + cut_h // 2)

        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1) / (width * height))

        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


# Test augmentation strength levels
def get_augmentation_levels(level: str = 'medium') -> dict:
    """
    Get augmentation parameters for different strength levels

    Args:
        level: 'light', 'medium', or 'heavy'

    Returns:
        Dictionary of augmentation parameters
    """
    levels = {
        'light': {
            'rotation': 5,
            'brightness': 0.1,
            'contrast': 0.1,
            'saturation': 0.1,
            'hue': 0.02,
            'perspective': 0.1,
            'scale': (0.9, 1.0),
        },
        'medium': {
            'rotation': 10,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.05,
            'perspective': 0.2,
            'scale': (0.8, 1.0),
        },
        'heavy': {
            'rotation': 15,
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.1,
            'perspective': 0.3,
            'scale': (0.7, 1.0),
        }
    }

    return levels.get(level, levels['medium'])


if __name__ == "__main__":
    # Test transforms
    import numpy as np
    from PIL import Image

    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Test train transforms
    train_transform = get_train_transforms()
    transformed = train_transform(dummy_image)
    print(f"Train transform output shape: {transformed.shape}")

    # Test val transforms
    val_transform = get_val_transforms()
    transformed = val_transform(dummy_image)
    print(f"Val transform output shape: {transformed.shape}")

    # Test model-specific transforms
    for model_name in ['efficientnet-b2', 'convnext-tiny-384', 'resnet50']:
        transform = get_transforms_for_model(model_name, is_training=True)
        config = MODEL_CONFIGS.get(model_name, {})
        print(f"{model_name}: input_size={config.get('input_size', 224)}")