from torchvision import transforms
import torch
from typing import Tuple, Dict, Optional
import numpy as np

# Import augmentation configs from model configs
from ..models.model_configs import AUGMENTATION_CONFIGS, TRAINING_CONFIGS


def get_train_transforms(
        input_size: int = 224,
        augmentation_level: str = 'medium'
) -> transforms.Compose:
    """
    Get training data augmentation pipeline

    Args:
        input_size: Target image size
        augmentation_level: Augmentation strength ('light', 'medium', 'heavy')

    Returns:
        Composed transform pipeline
    """
    # Get augmentation parameters
    aug_params = AUGMENTATION_CONFIGS.get(augmentation_level, AUGMENTATION_CONFIGS['medium'])

    transform_list = [
        # Resize with some randomness
        transforms.RandomResizedCrop(
            input_size,
            scale=aug_params['scale'],
            ratio=(0.9, 1.1),  # Aspect ratio variation
        ),

        # Horizontal flip (makes sense for products)
        transforms.RandomHorizontalFlip(p=aug_params['h_flip_p']),

        # Rotation
        transforms.RandomRotation(degrees=aug_params['rotation']),

        # Color augmentation
        transforms.ColorJitter(
            brightness=aug_params['brightness'],
            contrast=aug_params['contrast'],
            saturation=aug_params['saturation'],
            hue=aug_params['hue']
        ),

        # Perspective transformation
        transforms.RandomPerspective(
            distortion_scale=aug_params['perspective'],
            p=0.3
        ),

        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    return transforms.Compose(transform_list)


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


def get_transforms_for_model(
        model_name: str,
        is_training: bool = True,
        augmentation_level: str = 'medium'
) -> transforms.Compose:
    """
    Get appropriate transforms for a specific model

    Args:
        model_name: Name of the model
        is_training: Whether to include augmentations
        augmentation_level: Strength of augmentations for training

    Returns:
        Transform pipeline
    """
    # Get model-specific input size from configs
    config = TRAINING_CONFIGS.get(model_name, TRAINING_CONFIGS['resnet50'])
    input_size = config['input_size']

    if is_training:
        return get_train_transforms(input_size, augmentation_level)
    else:
        return get_val_transforms(input_size)


class MixUpTransform:
    """
    MixUp augmentation for training
    Reference: https://arxiv.org/abs/1710.09412
    """

    def __init__(self, alpha: float = 1.0, num_classes: int = 5):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(
            self,
            images: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
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
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        # Random shuffle for mixing
        index = torch.randperm(batch_size).to(images.device)

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

    def __init__(self, alpha: float = 1.0, num_classes: int = 5):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(
            self,
            images: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch
        """
        batch_size, _, height, width = images.size()

        # Sample lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        # Random shuffle for mixing
        index = torch.randperm(batch_size).to(images.device)

        # Create random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        # Random center point
        cx = np.random.randint(width)
        cy = np.random.randint(height)

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


class RandAugmentTransform:
    """
    RandAugment for automatic augmentation policy
    Simplified version for product images
    """

    def __init__(self, n: int = 2, m: int = 10):
        """
        Args:
            n: Number of augmentation transformations to apply
            m: Magnitude of transformations
        """
        self.n = n
        self.m = m

        # Define augmentation pool suitable for product images
        self.augmentations = [
            lambda img, mag: transforms.functional.rotate(img, mag * 3),
            lambda img, mag: transforms.functional.adjust_brightness(img, 1 + mag * 0.05),
            lambda img, mag: transforms.functional.adjust_contrast(img, 1 + mag * 0.05),
            lambda img, mag: transforms.functional.adjust_saturation(img, 1 + mag * 0.05),
            lambda img, mag: transforms.functional.adjust_sharpness(img, 1 + mag * 0.1),
        ]

    def __call__(self, img):
        """Apply RandAugment to an image"""
        # Randomly select n augmentations
        selected_augs = np.random.choice(self.augmentations, self.n, replace=False)

        for aug in selected_augs:
            img = aug(img, self.m)

        return img


def get_advanced_train_transforms(
        input_size: int = 224,
        use_randaugment: bool = False,
        randaugment_n: int = 2,
        randaugment_m: int = 10
) -> transforms.Compose:
    """
    Get advanced training transforms with optional RandAugment

    Args:
        input_size: Target image size
        use_randaugment: Whether to use RandAugment
        randaugment_n: Number of augmentations
        randaugment_m: Magnitude of augmentations

    Returns:
        Transform pipeline
    """
    transform_list = [
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
    ]

    if use_randaugment:
        transform_list.append(RandAugmentTransform(n=randaugment_n, m=randaugment_m))

    transform_list.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(transform_list)


# Denormalization for visualization
class DeNormalize:
    """Denormalize tensor for visualization"""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor: Normalized image tensor
        Returns:
            Denormalized tensor
        """
        return tensor * self.std + self.mean


def test_augmentations(
        image_path: str,
        model_name: str = 'efficientnet-b2',
        num_samples: int = 8
):
    """
    Test and visualize augmentations

    Args:
        image_path: Path to test image
        model_name: Model name for transforms
        num_samples: Number of augmented samples to generate
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    # Load image
    img = Image.open(image_path).convert('RGB')

    # Get transforms
    transform = get_transforms_for_model(model_name, is_training=True)

    # Generate augmented samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_samples):
        augmented = transform(img)

        # Denormalize for visualization
        denorm = DeNormalize()
        augmented = denorm(augmented)
        augmented = torch.clamp(augmented, 0, 1)

        # Convert to numpy
        augmented = augmented.permute(1, 2, 0).numpy()

        axes[i].imshow(augmented)
        axes[i].axis('off')
        axes[i].set_title(f'Augmented {i + 1}')

    plt.suptitle(f'Augmentation samples for {model_name}')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test transforms
    import numpy as np
    from PIL import Image

    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Test different augmentation levels
    for level in ['light', 'medium', 'heavy']:
        transform = get_train_transforms(augmentation_level=level)
        transformed = transform(dummy_image)
        print(f"{level} augmentation output shape: {transformed.shape}")

    # Test model-specific transforms
    for model_name in ['efficientnet-b2', 'convnext-tiny-384', 'resnet50']:
        transform = get_transforms_for_model(model_name, is_training=True)
        config = TRAINING_CONFIGS.get(model_name, {})
        print(f"{model_name}: input_size={config.get('input_size', 224)}")