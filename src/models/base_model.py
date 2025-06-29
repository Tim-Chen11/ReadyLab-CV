import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations"""
        pass

    def freeze_backbone(self, freeze_ratio: float = 0.5) -> None:
        """Freeze early layers of the model"""
        pass

    def unfreeze_all(self) -> None:
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True


class ModelWithFeatures(BaseModel):
    """Wrapper to add feature extraction to any model"""

    def __init__(self, base_model: nn.Module, feature_layer: str = None, num_classes: int = 5):
        super().__init__(num_classes)
        self.base_model = base_model
        self.feature_layer = feature_layer
        self.features = None

        if feature_layer:
            self._register_hook()

    def _register_hook(self):
        """Register forward hook to extract features"""

        def hook(module, input, output):
            self.features = output.detach()

        # Find the layer
        for name, module in self.base_model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook)
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.base_model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification layer"""
        # Forward pass to populate features
        _ = self.forward(x)
        return self.features if self.features is not None else x

    def freeze_backbone(self, freeze_ratio: float = 0.5) -> None:
        """Freeze early layers"""
        all_params = list(self.base_model.named_parameters())
        num_to_freeze = int(len(all_params) * freeze_ratio)

        for i, (name, param) in enumerate(all_params):
            if i < num_to_freeze:
                param.requires_grad = False


class EnsembleModel(BaseModel):
    """Ensemble multiple models for better performance"""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models"""
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Weighted average
        stacked = torch.stack(outputs, dim=0)
        weights = self.weights.to(x.device).view(-1, 1, 1)
        return (stacked * weights).sum(dim=0)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get concatenated features from all models"""
        features = []
        for model in self.models:
            if hasattr(model, 'get_features'):
                features.append(model.get_features(x))

        return torch.cat(features, dim=1) if features else x