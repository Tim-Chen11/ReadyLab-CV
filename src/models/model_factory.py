import torch
import torch.nn as nn
import timm
from typing import Dict, Optional, Union, List
import logging

# Import configurations from separate file
from .model_configs import MODEL_REGISTRY, TRAINING_CONFIGS, get_model_family, FINETUNE_CONFIGS
from .base_model import ModelWithFeatures

logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):
    """Multi-task head for decade, cluster, and device type prediction"""
    
    def __init__(
        self, 
        in_features: int,
        num_decade_classes: int = 5,
        num_cluster_classes: int = 10,
        num_device_classes: int = 2,  # phone or calculator
        hidden_dim: int = 512,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.num_decade_classes = num_decade_classes
        self.num_cluster_classes = num_cluster_classes
        self.num_device_classes = num_device_classes
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads
        self.decade_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_decade_classes)
        )
        
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_cluster_classes)
        )
        
        # New device type head (phone vs calculator)
        self.device_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_device_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly"""
        for m in [self.shared_features, self.decade_head, self.cluster_head, self.device_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-task head"""
        shared_features = self.shared_features(x)
        
        decade_logits = self.decade_head(shared_features)
        cluster_logits = self.cluster_head(shared_features)
        device_logits = self.device_head(shared_features)
        
        return {
            'decade': decade_logits,
            'cluster': cluster_logits,
            'device': device_logits
        }


class MultiTaskModel(nn.Module):
    """Wrapper to convert single-task model to multi-task"""
    
    def __init__(
        self,
        backbone: nn.Module,
        num_decade_classes: int = 5,
        num_cluster_classes: int = 10,
        num_device_classes: int = 2,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.backbone = backbone
        
        # Get the number of features from the backbone
        # This works for most timm models
        if hasattr(backbone, 'num_features'):
            in_features = backbone.num_features
        elif hasattr(backbone, 'classifier'):
            if isinstance(backbone.classifier, nn.Linear):
                in_features = backbone.classifier.in_features
            else:
                # For more complex classifiers, take the last linear layer
                in_features = None
                for module in reversed(list(backbone.classifier.modules())):
                    if isinstance(module, nn.Linear):
                        in_features = module.in_features
                        break
                if in_features is None:
                    raise ValueError("Could not determine backbone output features")
        else:
            raise ValueError("Could not determine backbone output features")
        
        # Replace the classifier with identity to get features
        if hasattr(backbone, 'classifier'):
            backbone.classifier = nn.Identity()
        elif hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        else:
            # Try to find and replace the last linear layer
            for name, module in backbone.named_modules():
                if isinstance(module, nn.Linear) and 'classifier' in name.lower():
                    setattr(backbone, name.split('.')[-1], nn.Identity())
                    break
        
        # Create multi-task head
        self.multitask_head = MultiTaskHead(
            in_features=in_features,
            num_decade_classes=num_decade_classes,
            num_cluster_classes=num_cluster_classes,
            num_device_classes=num_device_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        logger.info(f"Created multi-task model with {in_features} backbone features")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        features = self.backbone(x)
        return self.multitask_head(features)
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with softmax"""
        with torch.no_grad():
            logits = self.forward(x)
            return {
                'decade': torch.softmax(logits['decade'], dim=1),
                'cluster': torch.softmax(logits['cluster'], dim=1),
                'device': torch.softmax(logits['device'], dim=1)
            }


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(
        self, 
        decade_weight: float = 1.0,
        cluster_weight: float = 1.0,
        device_weight: float = 1.0,
        loss_type: str = 'cross_entropy',
        loss_params: Optional[Dict] = None
    ):
        super().__init__()
        
        self.decade_weight = decade_weight
        self.cluster_weight = cluster_weight
        self.device_weight = device_weight
        
        # Import losses module to access all loss functions
        from ..training.losses import get_loss_function
        
        # Create loss functions for each task
        loss_params = loss_params or {}
        self.decade_criterion = get_loss_function(loss_type, **loss_params)
        self.cluster_criterion = get_loss_function(loss_type, **loss_params)
        self.device_criterion = get_loss_function(loss_type, **loss_params)
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate combined loss"""
        decade_loss = self.decade_criterion(predictions['decade'], targets['decade'])
        cluster_loss = self.cluster_criterion(predictions['cluster'], targets['cluster'])
        device_loss = self.device_criterion(predictions['device'], targets['device'])
        
        total_loss = (self.decade_weight * decade_loss + 
                     self.cluster_weight * cluster_loss +
                     self.device_weight * device_loss)
        
        return {
            'total_loss': total_loss,
            'decade_loss': decade_loss,
            'cluster_loss': cluster_loss,
            'device_loss': device_loss
        }


class ModelFactory:
    """Factory class for creating different model architectures"""

    @classmethod
    def create_model(
            cls,
            model_name: str,
            num_classes: Union[int, Dict[str, int]] = 5,  # Can be int or dict for multi-task
            pretrained: bool = True,
            checkpoint_path: Optional[str] = None,
            return_features: bool = False,
            multi_task: bool = False,
            multitask_config: Optional[Dict] = None
    ) -> nn.Module:
        """
        Create a model instance

        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes (int) or dict with task names for multi-task
            pretrained: Whether to use pretrained weights
            checkpoint_path: Path to load checkpoint from
            return_features: Wrap model to return features
            multi_task: Whether to create multi-task model
            multitask_config: Configuration for multi-task head

        Returns:
            Model instance
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")

        timm_model_name = MODEL_REGISTRY[model_name]

        if multi_task:
            # Multi-task model creation
            if isinstance(num_classes, dict):
                num_decade_classes = num_classes.get('decade', 5)
                num_cluster_classes = num_classes.get('cluster', 10)
                num_device_classes = num_classes.get('device', 2)
            else:
                # Assume single number is for decades, cluster and device classes need to be specified
                num_decade_classes = num_classes
                num_cluster_classes = multitask_config.get('num_cluster_classes', 10) if multitask_config else 10
                num_device_classes = multitask_config.get('num_device_classes', 2) if multitask_config else 2
            
            # Create backbone with dummy classifier (will be replaced)
            backbone = timm.create_model(
                timm_model_name,
                pretrained=pretrained,
                num_classes=1000  # Use original pretrained classes initially
            )
            
            # Create multi-task wrapper
            multitask_config = multitask_config or {}
            model = MultiTaskModel(
                backbone=backbone,
                num_decade_classes=num_decade_classes,
                num_cluster_classes=num_cluster_classes,
                num_device_classes=num_device_classes,
                hidden_dim=multitask_config.get('hidden_dim', 512),
                dropout_rate=multitask_config.get('dropout_rate', 0.3)
            )
            
            logger.info(f"Created multi-task model: {model_name}")
            logger.info(f"Decade classes: {num_decade_classes}, Cluster classes: {num_cluster_classes}, Device classes: {num_device_classes}")
            
        else:
            # Single-task model creation (original behavior)
            if isinstance(num_classes, dict):
                num_classes = num_classes.get('decade', 5)  # Default to decade task
            
            model = timm.create_model(
                timm_model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            
            # Wrap with feature extractor if requested
            if return_features:
                model = ModelWithFeatures(model, num_classes=num_classes)
            
            logger.info(f"Created single-task model: {model_name}")

        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Load checkpoint if provided
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded checkpoint from {checkpoint_path}")
                except RuntimeError as e:
                    logger.warning(f"Could not load full checkpoint due to architecture mismatch: {e}")
                    logger.info("Attempting to load compatible layers only...")
                    
                    # Load compatible layers only
                    model_dict = model.state_dict()
                    checkpoint_dict = checkpoint['model_state_dict']
                    
                    # Filter compatible layers
                    compatible_dict = {
                        k: v for k, v in checkpoint_dict.items() 
                        if k in model_dict and model_dict[k].shape == v.shape
                    }
                    
                    model_dict.update(compatible_dict)
                    model.load_state_dict(model_dict)
                    logger.info(f"Loaded {len(compatible_dict)} compatible layers from checkpoint")
            else:
                model.load_state_dict(checkpoint)

        return model

    @classmethod
    def create_multitask_loss(
        cls, 
        decade_weight: float = 1.0, 
        cluster_weight: float = 1.0,
        device_weight: float = 1.0,
        loss_type: str = 'cross_entropy',
        loss_params: Optional[Dict] = None
    ) -> MultiTaskLoss:
        """Create multi-task loss function"""
        return MultiTaskLoss(
            decade_weight=decade_weight,
            cluster_weight=cluster_weight,
            device_weight=device_weight,
            loss_type=loss_type,
            loss_params=loss_params
        )

    @classmethod
    def get_model_config(cls, model_name: str, multi_task: bool = False) -> Dict:
        """Get default configuration for a model"""
        if model_name not in TRAINING_CONFIGS:
            logger.warning(f"No default config for {model_name}, using base config")
            config = TRAINING_CONFIGS.get('resnet50', {}).copy()
        else:
            config = TRAINING_CONFIGS[model_name].copy()
        
        # Add multi-task specific configurations
        if multi_task:
            config.update({
                'multi_task': True,
                'decade_weight': 1.0,
                'cluster_weight': 1.0,
                'multitask_hidden_dim': 512,
                'multitask_dropout': 0.3
            })
        
        return config

    @classmethod
    def get_finetune_config(cls, model_name: str) -> Dict:
        """Get fine-tuning configuration for a model"""
        model_family = get_model_family(model_name)
        return FINETUNE_CONFIGS.get(model_family, {}).copy()

    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model architectures"""
        return list(MODEL_REGISTRY.keys())

    @classmethod
    def get_model_info(cls, model_name: str, multi_task: bool = False) -> Dict:
        """Get detailed information about a model"""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        # Create a temporary model to get info
        num_classes = {'decade': 5, 'cluster': 10} if multi_task else 5
        model = cls.create_model(model_name, num_classes=num_classes, pretrained=False, multi_task=multi_task)

        info = {
            'name': model_name,
            'timm_name': MODEL_REGISTRY[model_name],
            'multi_task': multi_task,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'default_config': cls.get_model_config(model_name, multi_task=multi_task),
            'finetune_config': cls.get_finetune_config(model_name),
        }

        # Clean up
        del model

        return info


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
    # Handle multi-task models - only freeze backbone
    if isinstance(model, MultiTaskModel):
        target_model = model.backbone
        logger.info("Freezing backbone layers in multi-task model")
    else:
        target_model = model
    
    # Get all named parameters
    all_params = list(target_model.named_parameters())
    num_to_freeze = int(len(all_params) * freeze_ratio)

    # Freeze early layers
    for i, (name, param) in enumerate(all_params):
        if i < num_to_freeze:
            param.requires_grad = False
            logger.debug(f"Froze layer: {name}")
        else:
            param.requires_grad = True

    # Count parameters for the whole model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Froze {num_to_freeze}/{len(all_params)} backbone layers")
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")


if __name__ == "__main__":
    # Test model creation
    print("Testing ModelFactory with Multi-task Support...\n")

    # List available models
    print("Available models:")
    for model_name in ModelFactory.list_available_models():
        print(f"  - {model_name}")

    # Test creating a single-task model
    print("\nCreating single-task EfficientNet-B2...")
    single_model = ModelFactory.create_model('efficientnet-b2', num_classes=5)
    
    # Test creating a multi-task model
    print("\nCreating multi-task EfficientNet-B2...")
    multi_model = ModelFactory.create_model(
        'efficientnet-b2', 
        num_classes={'decade': 5, 'cluster': 8},
        multi_task=True,
        multitask_config={
            'hidden_dim': 512,
            'dropout_rate': 0.3
        }
    )

    # Get model info
    single_info = ModelFactory.get_model_info('efficientnet-b2', multi_task=False)
    multi_info = ModelFactory.get_model_info('efficientnet-b2', multi_task=True)
    
    print(f"\nSingle-task model parameters: {single_info['num_parameters']:,}")
    print(f"Multi-task model parameters: {multi_info['num_parameters']:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 260, 260)
    
    print("\nTesting single-task forward pass...")
    single_output = single_model(dummy_input)
    print(f"Single-task output shape: {single_output.shape}")
    
    print("\nTesting multi-task forward pass...")
    multi_output = multi_model(dummy_input)
    print(f"Multi-task output shapes:")
    for task, output in multi_output.items():
        print(f"  {task}: {output.shape}")

    # Test multi-task loss
    print("\nTesting multi-task loss...")
    loss_fn = ModelFactory.create_multitask_loss(decade_weight=1.0, cluster_weight=0.8)
    
    targets = {
        'decade': torch.randint(0, 5, (2,)),
        'cluster': torch.randint(0, 8, (2,))
    }
    
    losses = loss_fn(multi_output, targets)
    print(f"Loss components:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")

    print("\n✅ All tests passed!")