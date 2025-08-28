#!/usr/bin/env python
"""
Inference script for trained models
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from src.models.model_factory import ModelFactory
from src.data.transforms import get_transforms_for_model
from src.utils.helpers import get_device


class ModelInference:
    """Class for model inference"""
    
    def __init__(
        self, 
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        multi_task: bool = False
    ):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            multi_task: Whether the model is multi-task
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or get_device()
        self.multi_task = multi_task
        
        # Load checkpoint (handle PyTorch 2.6+ weights_only change)
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Trying with weights_only=False due to: {e}")
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint.get('config', {})
        
        # Check if checkpoint indicates multi-task
        if 'multi_task' in self.checkpoint:
            self.multi_task = self.checkpoint['multi_task']
        
        # Create model
        self._create_model()
        
        # Load transforms
        self.transform = get_transforms_for_model(
            self.config.get('model_name', 'efficientnet-b0'),
            is_training=False
        )
        
        # Get class names from checkpoint or use defaults
        if 'class_names' in self.checkpoint:
            # Use class names from checkpoint if available
            if self.multi_task:
                self.decades = self.checkpoint['class_names'].get('decade', ['1960s', '1970s', '1980s', '1990s', '2000s'])
                self.clusters = self.checkpoint['class_names'].get('cluster', [0, 1, 2, 3, 4])
                self.devices = self.checkpoint['class_names'].get('device', ['calculator', 'phone'])
            else:
                self.decades = self.checkpoint['class_names']
                if not isinstance(self.decades, list):
                    self.decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
                self.clusters = [0, 1, 2, 3, 4]
                self.devices = ['calculator', 'phone']
        else:
            # Use default labels
            self.decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
            self.clusters = [0, 1, 2, 3, 4]
            self.devices = ['calculator', 'phone']
        
        print(f"Model configured for {len(self.decades)} decade classes")
        print(f"Decade labels: {self.decades}")
        if self.multi_task:
            print(f"Cluster labels: {self.clusters}")
            print(f"Device labels: {self.devices}")
    
    def _create_model(self):
        """Create and load model"""
        model_name = self.config.get('model_name', 'efficientnet-b0')
        
        if self.multi_task:
            # Multi-task model
            num_classes = {
                'decade': self.config.get('num_decade_classes', 5),
                'cluster': self.config.get('num_cluster_classes', 5),
                'device': self.config.get('num_device_classes', 2)
            }
            self.model = ModelFactory.create_model(
                model_name,
                num_classes=num_classes,
                multi_task=True,
                multitask_config={
                    'hidden_dim': self.config.get('multitask_hidden_dim', 512),
                    'dropout_rate': self.config.get('multitask_dropout', 0.3),
                    'num_device_classes': self.config.get('num_device_classes', 2)
                },
                pretrained=False
            )
        else:
            # Single-task model
            num_classes = self.config.get('num_classes', 5)
            self.model = ModelFactory.create_model(
                model_name,
                num_classes=num_classes,
                pretrained=False
            )
        
        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded {model_name} model from {self.checkpoint_path}")
        print(f"Multi-task mode: {self.multi_task}")
    
    def predict_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict on a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with predictions
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
        
        if self.multi_task:
            # Multi-task predictions
            decade_probs = F.softmax(output['decade'], dim=1)
            cluster_probs = F.softmax(output['cluster'], dim=1)
            
            # Check if device output exists
            if 'device' in output:
                device_probs = F.softmax(output['device'], dim=1)
                device_prob, device_idx = device_probs.max(1)
            else:
                device_probs = None
                device_prob = None
                device_idx = None
            
            # Get top predictions
            decade_prob, decade_idx = decade_probs.max(1)
            cluster_prob, cluster_idx = cluster_probs.max(1)
            
            # Get top-3 for each task
            decade_top3_probs, decade_top3_idx = decade_probs.topk(3, dim=1)
            cluster_top3_probs, cluster_top3_idx = cluster_probs.topk(3, dim=1)
            
            result = {
                'decade': {
                    'prediction': self.decades[decade_idx.item()],
                    'confidence': decade_prob.item(),
                    'top3': [
                        {
                            'class': self.decades[idx],
                            'confidence': prob
                        }
                        for idx, prob in zip(
                            decade_top3_idx[0].tolist(),
                            decade_top3_probs[0].tolist()
                        )
                    ]
                },
                'cluster': {
                    'prediction': self.clusters[cluster_idx.item()],
                    'confidence': cluster_prob.item(),
                    'top3': [
                        {
                            'class': self.clusters[idx],
                            'confidence': prob
                        }
                        for idx, prob in zip(
                            cluster_top3_idx[0].tolist(),
                            cluster_top3_probs[0].tolist()
                        )
                    ]
                }
            }
            
            # Add device prediction if available
            if device_probs is not None:
                result['device'] = {
                    'prediction': self.devices[device_idx.item()],
                    'confidence': device_prob.item()
                }
        else:
            # Single-task predictions
            probs = F.softmax(output, dim=1)
            prob, idx = probs.max(1)
            
            # Get top-3 predictions
            top3_probs, top3_idx = probs.topk(3, dim=1)
            
            result = {
                'decade': {
                    'prediction': self.decades[idx.item()],
                    'confidence': prob.item(),
                    'top3': [
                        {
                            'class': self.decades[i],
                            'confidence': p
                        }
                        for i, p in zip(top3_idx[0].tolist(), top3_probs[0].tolist())
                    ],
                    'all_probabilities': {
                        self.decades[i]: p for i, p in enumerate(probs[0].tolist())
                    }
                }
            }
        
        return result
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                result['image_path'] = str(image_path)
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'status': 'error',
                    'error': str(e)
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('images', nargs='+', help='Image paths to predict on')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--multi-task', action='store_true',
                        help='Use multi-task model')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = ModelInference(
        checkpoint_path=args.checkpoint,
        device=get_device(args.device),
        multi_task=args.multi_task
    )
    
    # Run predictions
    if len(args.images) == 1 and Path(args.images[0]).is_dir():
        # Directory of images
        image_dir = Path(args.images[0])
        image_paths = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.jpeg')) + \
                     list(image_dir.glob('*.png'))
        print(f"Found {len(image_paths)} images in {image_dir}")
    else:
        # Individual images
        image_paths = [Path(p) for p in args.images]
    
    # Predict
    results = inference.predict_batch(image_paths)
    
    # Display results
    for result in results:
        print(f"\nImage: {result['image_path']}")
        if result['status'] == 'success':
            if 'decade' in result:
                decade_info = result['decade']
                print(f"  Decade: {decade_info['prediction']} "
                      f"(confidence: {decade_info['confidence']:.2%})")
                print("  Top 3 decades:")
                for i, pred in enumerate(decade_info['top3'], 1):
                    print(f"    {i}. {pred['class']}: {pred['confidence']:.2%}")
            
            if 'cluster' in result:
                cluster_info = result['cluster']
                print(f"  Cluster: {cluster_info['prediction']} "
                      f"(confidence: {cluster_info['confidence']:.2%})")
        else:
            print(f"  Error: {result['error']}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()