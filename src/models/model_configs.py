# Model configurations separated from factory for better organization

MODEL_REGISTRY = {
    # EfficientNet family
    'efficientnet-b0': 'efficientnet_b0',
    'efficientnet-b1': 'efficientnet_b1',
    'efficientnet-b2': 'efficientnet_b2',
    'efficientnet-b3': 'efficientnet_b3',
    'efficientnet-b4': 'efficientnet_b4',
    'efficientnet-b5': 'efficientnet_b5',
    'efficientnet-b6': 'efficientnet_b6',
    'efficientnet-b7': 'efficientnet_b7',

    # ConvNeXt V2 family
    'convnext-tiny': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'convnext-small': 'convnextv2_small.fcmae_ft_in22k_in1k',
    'convnext-base': 'convnextv2_base.fcmae_ft_in22k_in1k',
    'convnext-tiny-384': 'convnextv2_tiny.fcmae_ft_in22k_in1k_384',

    # ResNet family
    'resnet18': 'resnet18',
    'resnet34': 'resnet34',
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
    'resnet152': 'resnet152',

    # ResNeXt family
    'resnext50': 'resnext50_32x4d',
    'resnext101': 'resnext101_32x8d',

    # Vision Transformer family
    'vit-tiny': 'vit_tiny_patch16_224',
    'vit-small': 'vit_small_patch16_224',
    'vit-base': 'vit_base_patch16_224',

    # MobileNet family
    'mobilenet-v3-small': 'mobilenetv3_small_100',
    'mobilenet-v3-large': 'mobilenetv3_large_100',
}

# Training configurations for each model
TRAINING_CONFIGS = {
    # EfficientNet configs
    'efficientnet-b0': {
        'input_size': 224,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'epochs': 30,
        'warmup_epochs': 3,
        'min_lr': 1e-6,
    },
    'efficientnet-b1': {
        'input_size': 240,
        'batch_size': 96,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'epochs': 30,
        'warmup_epochs': 3,
        'min_lr': 1e-6,
    },
    'efficientnet-b2': {
        'input_size': 260,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'epochs': 30,
        'warmup_epochs': 3,
        'min_lr': 1e-6,
    },
    'efficientnet-b3': {
        'input_size': 300,
        'batch_size': 48,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'epochs': 30,
        'warmup_epochs': 3,
        'min_lr': 1e-6,
    },
    'efficientnet-b4': {
        'input_size': 380,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'epochs': 30,
        'warmup_epochs': 3,
        'min_lr': 1e-6,
    },

    # ConvNeXt configs - these models benefit from different hyperparameters
    'convnext-tiny': {
        'input_size': 224,
        'batch_size': 64,
        'learning_rate': 5e-4,
        'weight_decay': 0.05,  # Higher weight decay for ConvNeXt
        'epochs': 30,
        'warmup_epochs': 5,
        'min_lr': 1e-6,
        'drop_path_rate': 0.1,
    },
    'convnext-base': {
        'input_size': 224,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'weight_decay': 0.05,
        'epochs': 30,
        'warmup_epochs': 5,
        'min_lr': 1e-6,
        'drop_path_rate': 0.2,
    },
    'convnext-tiny-384': {
        'input_size': 384,
        'batch_size': 16,  # Smaller batch due to larger image size
        'learning_rate': 5e-4,
        'weight_decay': 0.05,
        'epochs': 30,
        'warmup_epochs': 5,
        'min_lr': 1e-6,
        'drop_path_rate': 0.1,
    },

    # ResNet configs
    'resnet50': {
        'input_size': 224,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 30,
        'warmup_epochs': 3,
        'min_lr': 1e-6,
    },
    'resnext50': {
        'input_size': 224,
        'batch_size': 48,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 30,
        'warmup_epochs': 3,
        'min_lr': 1e-6,
    },

    # Vision Transformer configs
    'vit-tiny': {
        'input_size': 224,
        'batch_size': 128,
        'learning_rate': 1e-4,  # Lower LR for ViT
        'weight_decay': 0.05,
        'epochs': 30,
        'warmup_epochs': 5,
        'min_lr': 1e-6,
        'drop_rate': 0.1,
    },
}

# Augmentation configurations
AUGMENTATION_CONFIGS = {
    'light': {
        'rotation': 5,
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.02,
        'perspective': 0.1,
        'scale': (0.9, 1.0),
        'h_flip_p': 0.5,
    },
    'medium': {
        'rotation': 10,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.05,
        'perspective': 0.2,
        'scale': (0.8, 1.0),
        'h_flip_p': 0.5,
    },
    'heavy': {
        'rotation': 15,
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.3,
        'hue': 0.1,
        'perspective': 0.3,
        'scale': (0.7, 1.0),
        'h_flip_p': 0.5,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
    }
}

# Model-specific fine-tuning strategies
FINETUNE_CONFIGS = {
    'efficientnet': {
        'freeze_stages': ['stem', 'blocks.0', 'blocks.1'],  # Freeze early layers
        'unfreeze_epoch': 10,  # Unfreeze all layers after this epoch
        'discriminative_lr': True,  # Use different LRs for different layers
        'lr_mult': [0.1, 0.3, 0.5, 0.7, 1.0],  # LR multipliers from bottom to top
    },
    'convnext': {
        'freeze_stages': ['stem', 'stages.0'],
        'unfreeze_epoch': 10,
        'discriminative_lr': True,
        'lr_mult': [0.1, 0.5, 1.0],
    },
    'resnet': {
        'freeze_stages': ['conv1', 'bn1', 'layer1'],
        'unfreeze_epoch': 10,
        'discriminative_lr': False,
    },
    'vit': {
        'freeze_stages': ['patch_embed', 'blocks.0', 'blocks.1'],
        'unfreeze_epoch': 15,
        'discriminative_lr': True,
        'lr_mult': [0.1, 0.5, 1.0],
    }
}


def get_model_family(model_name: str) -> str:
    """Get the family of a model (e.g., 'efficientnet', 'convnext')"""
    if 'efficientnet' in model_name:
        return 'efficientnet'
    elif 'convnext' in model_name:
        return 'convnext'
    elif 'resnext' in model_name:
        return 'resnet'  # ResNeXt uses similar strategy as ResNet
    elif 'resnet' in model_name:
        return 'resnet'
    elif 'vit' in model_name:
        return 'vit'
    elif 'mobilenet' in model_name:
        return 'efficientnet'  # Similar strategy
    else:
        return 'default'


def get_optimal_batch_size(model_name: str, gpu_memory_gb: int = 8) -> int:
    """
    Get optimal batch size based on model and GPU memory

    Args:
        model_name: Name of the model
        gpu_memory_gb: GPU memory in GB

    Returns:
        Recommended batch size
    """
    base_config = TRAINING_CONFIGS.get(model_name, TRAINING_CONFIGS['resnet50'])
    base_batch_size = base_config['batch_size']

    # Adjust based on GPU memory (assuming base is for 8GB)
    memory_multiplier = gpu_memory_gb / 8.0

    # Some models scale better than others
    if 'efficientnet' in model_name:
        memory_multiplier *= 1.2  # EfficientNets are memory efficient
    elif 'vit' in model_name:
        memory_multiplier *= 0.8  # ViTs use more memory

    adjusted_batch_size = int(base_batch_size * memory_multiplier)

    # Round to nearest power of 2 or multiple of 8
    return max(8, min(256, (adjusted_batch_size // 8) * 8))


# Default config for unknown models
DEFAULT_CONFIG = {
    'input_size': 224,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 30,
    'warmup_epochs': 3,
    'min_lr': 1e-6,
}