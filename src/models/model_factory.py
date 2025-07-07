import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating different model architectures"""
    
    # Model registry with timm model names
    MODEL_REGISTRY = {
        # EfficientNet family
        'efficientnet-b0': 'efficientnet_b0',
        'efficientnet-b1': 'efficientnet_b1',
        'efficientnet-b2': 'efficientnet_b2',
        'efficientnet-b3': 'efficientnet_b3',
        'efficientnet-b4': 'efficientnet_b4',
        
        # ConvNeXt family
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
    }
    
    # Default configurations for each model
    DEFAULT_CONFIGS = {
        # EfficientNet configs
        'efficientnet-b0': {
            'input_size': 224,
            'batch_size': 128,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 30,
        },
        'efficientnet-b1': {
            'input_size': 240,
            'batch_size': 96,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 30,
        },
        'efficientnet-b2': {
            'input_size': 260,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 30,
        },
        'efficientnet-b3': {
            'input_size': 300,
            'batch_size': 48,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 30,
        },
        'efficientnet-b4': {
            'input_size': 380,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 30,
        },
        
        # ConvNeXt configs
        'convnext-tiny': {
            'input_size': 224,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'weight_decay': 0.05,
            'epochs': 30,
        },
        'convnext-small': {
            'input_size': 224,
            'batch_size': 48,
            'learning_rate': 5e-4,
            'weight_decay': 0.05,
            'epochs': 30,
        },
        'convnext-base': {
            'input_size': 224,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 0.05,
            'epochs': 30,
        },
        'convnext-tiny-384': {
            'input_size': 384,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 0.05,
            'epochs': 30,
        },
        
        # ResNet configs
        'resnet50': {
            'input_size': 224,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 30,
        },
        'resnext50': {
            'input_size': 224,
            'batch_size': 48,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 30,
        },
    }
    
    @classmethod
    def create_model(
        cls,
        model_name: str,
        num_classes: int = 5,  # 5 decades: 1960s-2000s
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None
    ) -> nn.Module:
        """
        Create a model instance
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            checkpoint_path: Path to load checkpoint from
        
        Returns:
            Model instance
        """
        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(cls.MODEL_REGISTRY.keys())}")
        
        timm_model_name = cls.MODEL_REGISTRY[model_name]
        
        # Create model
        model = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        logger.info(f"Created model: {model_name} (timm: {timm_model_name})")
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return model
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict:
        """Get default configuration for a model"""
        if model_name not in cls.DEFAULT_CONFIGS:
            logger.warning(f"No default config for {model_name}, using base config")
            return cls.DEFAULT_CONFIGS.get('resnet50').copy()
        
        return cls.DEFAULT_CONFIGS[model_name].copy()
    
    @classmethod
    def list_available_models(cls) -> list:
        """List all available model architectures"""
        return list(cls.MODEL_REGISTRY.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """Get detailed information about a model"""
        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create a temporary model to get info
        model = cls.create_model(model_name, pretrained=False)
        
        info = {
            'name': model_name,
            'timm_name': cls.MODEL_REGISTRY[model_name],
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'default_config': cls.get_model_config(model_name),
        }
        
        # Clean up
        del model
        
        return info


class ModelWithFeatures(nn.Module):
    """Wrapper to extract features from any model"""
    
    def __init__(self, base_model: nn.Module, feature_layer: str = 'global_pool'):
        super().__init__()
        self.base_model = base_model
        self.feature_layer = feature_layer
        
        # Register hook to extract features
        self.features = None
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook to extract features"""
        def hook(module, input, output):
            self.features = output
        
        # Find the layer
        for name, module in self.base_model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook)
                break
    
    def forward(self, x):
        """Forward pass returning both logits and features"""
        logits = self.base_model(x)
        return logits, self.features


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
    
    Returns:
        Optimizer instance
    """
    optimizer_name = config.get('optimizer', 'adamw')
    learning_rate = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
    
    Returns:
        Scheduler instance
    """
    scheduler_name = config.get('scheduler', 'cosine')
    epochs = config.get('epochs', 30)
    
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    elif scheduler_name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def freeze_backbone(model: nn.Module, freeze_ratio: float = 0.5):
    """
    Freeze early layers of the model
    
    Args:
        model: Model instance
        freeze_ratio: Ratio of layers to freeze (0.0 to 1.0)
    """
    # Get all named parameters
    all_params = list(model.named_parameters())
    num_to_freeze = int(len(all_params) * freeze_ratio)
    
    # Freeze early layers
    for i, (name, param) in enumerate(all_params):
        if i < num_to_freeze:
            param.requires_grad = False
            logger.debug(f"Froze layer: {name}")
        else:
            param.requires_grad = True
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Froze {num_to_freeze}/{len(all_params)} layers")
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


if __name__ == "__main__":
    # Test model creation
    print("Testing ModelFactory...\n")
    
    # List available models
    print("Available models:")
    for model_name in ModelFactory.list_available_models():
        print(f"  - {model_name}")
    
    # Test creating a model
    print("\nCreating EfficientNet-B2...")
    model = ModelFactory.create_model('efficientnet-b2', num_classes=5)
    
    # Get model info
    info = ModelFactory.get_model_info('efficientnet-b2')
    print(f"\nModel info:")
    print(f"  Parameters: {info['num_parameters']:,}")
    print(f"  Default config: {info['default_config']}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 260, 260)
    output = model(dummy_input)
    print(f"\nForward pass successful! Output shape: {output.shape}")
    
    # Test optimizer and scheduler creation
    config = ModelFactory.get_model_config('efficientnet-b2')
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    print(f"\nCreated optimizer: {type(optimizer).__name__}")
    print(f"Created scheduler: {type(scheduler).__name__}")