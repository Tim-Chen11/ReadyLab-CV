#!/usr/bin/env python3
"""
Script to display ALL data stored in training checkpoints
"""

import torch
import argparse
from pathlib import Path
import json
import numpy as np
from pprint import pprint
import sys


def format_size(num_bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def print_nested_dict(d, indent=0):
    """Recursively print nested dictionary with proper indentation"""
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_nested_dict(value, indent + 1)
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                print("  " * indent + f"{key}: [{len(value)} items]")
                # Show first and last item if many
                if len(value) > 2:
                    print("  " * (indent + 1) + f"First item:")
                    print_nested_dict(value[0], indent + 2)
                    print("  " * (indent + 1) + f"Last item:")
                    print_nested_dict(value[-1], indent + 2)
                else:
                    for i, item in enumerate(value):
                        print("  " * (indent + 1) + f"Item {i}:")
                        print_nested_dict(item, indent + 2)
            elif len(value) > 10:
                print("  " * indent + f"{key}: [List with {len(value)} items - showing first 5 and last 5]")
                print("  " * (indent + 1) + f"First 5: {value[:5]}")
                print("  " * (indent + 1) + f"Last 5: {value[-5:]}")
            else:
                print("  " * indent + f"{key}: {value}")
        elif isinstance(value, np.ndarray):
            print("  " * indent + f"{key}: numpy array with shape {value.shape}")
        elif isinstance(value, torch.Tensor):
            print("  " * indent + f"{key}: tensor with shape {list(value.shape)}")
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                print("  " * indent + f"{key}: {value:.6f}")
            else:
                print("  " * indent + f"{key}: {value}")
        else:
            # For other types, show type and str representation
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            print("  " * indent + f"{key}: {value_str}")


def analyze_checkpoint(checkpoint_path):
    """Load and analyze all data in checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    print("=" * 100)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Get file size
    file_size = checkpoint_path.stat().st_size
    print(f"Checkpoint file size: {format_size(file_size)}")
    print("=" * 100)
    
    # Display all top-level keys
    print("\n### TOP-LEVEL KEYS IN CHECKPOINT ###")
    print("-" * 50)
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            if 'state_dict' in key:
                # For state dicts, show number of parameters
                num_params = len(checkpoint[key])
                total_size = sum(p.numel() * p.element_size() if hasattr(p, 'numel') else 0 
                               for p in checkpoint[key].values())
                print(f"  - {key}: Dictionary with {num_params} parameters (~{format_size(total_size)})")
            else:
                print(f"  - {key}: Dictionary with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], list):
            print(f"  - {key}: List with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], (int, float)):
            print(f"  - {key}: {checkpoint[key]}")
        elif checkpoint[key] is None:
            print(f"  - {key}: None")
        else:
            print(f"  - {key}: {type(checkpoint[key]).__name__}")
    
    # Detailed breakdown of each component
    print("\n" + "=" * 100)
    print("### DETAILED CHECKPOINT CONTENTS ###")
    print("=" * 100)
    
    # 1. Basic Information
    print("\n1. BASIC INFORMATION")
    print("-" * 50)
    basic_keys = ['epoch', 'best_metric', 'multi_task']
    for key in basic_keys:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")
    
    # 2. Configuration
    if 'config' in checkpoint:
        print("\n2. TRAINING CONFIGURATION")
        print("-" * 50)
        print_nested_dict(checkpoint['config'])
    
    # 3. Validation Metrics
    if 'val_metrics' in checkpoint:
        print("\n3. VALIDATION METRICS (Final Epoch)")
        print("-" * 50)
        val_metrics = checkpoint['val_metrics']
        
        # Group metrics by type
        accuracy_metrics = {k: v for k, v in val_metrics.items() if 'accuracy' in k}
        loss_metrics = {k: v for k, v in val_metrics.items() if 'loss' in k}
        precision_metrics = {k: v for k, v in val_metrics.items() if 'precision' in k}
        recall_metrics = {k: v for k, v in val_metrics.items() if 'recall' in k}
        f1_metrics = {k: v for k, v in val_metrics.items() if 'f1' in k}
        other_metrics = {k: v for k, v in val_metrics.items() 
                        if not any(x in k for x in ['accuracy', 'loss', 'precision', 'recall', 'f1', 'confusion_matrix', 'support'])}
        
        if accuracy_metrics:
            print("\n  Accuracy Metrics:")
            for k, v in sorted(accuracy_metrics.items()):
                print(f"    • {k}: {v:.6f} ({v*100:.2f}%)")
        
        if loss_metrics:
            print("\n  Loss Metrics:")
            for k, v in sorted(loss_metrics.items()):
                print(f"    • {k}: {v:.6f}")
        
        if precision_metrics:
            print("\n  Precision Metrics:")
            for k, v in sorted(precision_metrics.items()):
                print(f"    • {k}: {v:.6f}")
        
        if recall_metrics:
            print("\n  Recall Metrics:")
            for k, v in sorted(recall_metrics.items()):
                print(f"    • {k}: {v:.6f}")
        
        if f1_metrics:
            print("\n  F1-Score Metrics:")
            for k, v in sorted(f1_metrics.items()):
                print(f"    • {k}: {v:.6f}")
        
        if other_metrics:
            print("\n  Other Metrics:")
            for k, v in sorted(other_metrics.items()):
                if isinstance(v, (int, float)):
                    print(f"    • {k}: {v:.6f}")
                else:
                    print(f"    • {k}: {type(v).__name__}")
        
        # Confusion Matrix
        if 'confusion_matrix' in val_metrics:
            print("\n  Confusion Matrix:")
            cm = val_metrics['confusion_matrix']
            if isinstance(cm, list):
                cm = np.array(cm)
                print(f"    • Shape: {cm.shape}")
                print("    • Matrix:")
                for row in cm:
                    print(f"      {row}")
    
    # 4. Metrics History
    if 'metrics_history' in checkpoint:
        print("\n4. METRICS HISTORY")
        print("-" * 50)
        history = checkpoint['metrics_history']
        
        if 'train' in history and history['train']:
            print(f"\n  Training History: {len(history['train'])} epochs recorded")
            print("  Sample from first epoch:")
            print_nested_dict(history['train'][0], indent=2)
            if len(history['train']) > 1:
                print("  Sample from last epoch:")
                print_nested_dict(history['train'][-1], indent=2)
        
        if 'val' in history and history['val']:
            print(f"\n  Validation History: {len(history['val'])} epochs recorded")
            print("  Sample from first epoch:")
            print_nested_dict(history['val'][0], indent=2)
            if len(history['val']) > 1:
                print("  Sample from last epoch:")
                print_nested_dict(history['val'][-1], indent=2)
    
    # 5. Model Architecture Info
    if 'model_state_dict' in checkpoint:
        print("\n5. MODEL ARCHITECTURE")
        print("-" * 50)
        state_dict = checkpoint['model_state_dict']
        
        # Count parameters by layer type
        layer_summary = {}
        total_params = 0
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                num_params = param.numel()
                total_params += num_params
                
                # Group by layer type
                layer_type = name.split('.')[0]
                if layer_type not in layer_summary:
                    layer_summary[layer_type] = {'count': 0, 'params': 0, 'layers': []}
                layer_summary[layer_type]['count'] += 1
                layer_summary[layer_type]['params'] += num_params
                layer_summary[layer_type]['layers'].append(name)
        
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Total Size: ~{format_size(total_params * 4)}")  # Assuming float32
        print("\n  Layer Summary:")
        for layer_type, info in sorted(layer_summary.items()):
            print(f"    • {layer_type}: {info['count']} tensors, {info['params']:,} parameters")
            # Show first few layer names
            if len(info['layers']) <= 3:
                for layer in info['layers']:
                    print(f"        - {layer}: {state_dict[layer].shape}")
            else:
                print(f"        - {info['layers'][0]}: {state_dict[info['layers'][0]].shape}")
                print(f"        ... {len(info['layers']) - 2} more layers ...")
                print(f"        - {info['layers'][-1]}: {state_dict[info['layers'][-1]].shape}")
    
    # 6. Optimizer State
    if 'optimizer_state_dict' in checkpoint:
        print("\n6. OPTIMIZER STATE")
        print("-" * 50)
        opt_state = checkpoint['optimizer_state_dict']
        
        if 'state' in opt_state:
            print(f"  Number of parameter groups with state: {len(opt_state['state'])}")
        
        if 'param_groups' in opt_state:
            print(f"  Number of parameter groups: {len(opt_state['param_groups'])}")
            for i, group in enumerate(opt_state['param_groups']):
                print(f"\n  Parameter Group {i}:")
                for key, value in group.items():
                    if key != 'params':  # Skip the params list as it's usually very long
                        print(f"    • {key}: {value}")
    
    # 7. Scheduler State
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        print("\n7. SCHEDULER STATE")
        print("-" * 50)
        scheduler_state = checkpoint['scheduler_state_dict']
        print_nested_dict(scheduler_state, indent=1)
    
    # 8. Class Names
    if 'class_names' in checkpoint and checkpoint['class_names'] is not None:
        print("\n8. CLASS NAMES")
        print("-" * 50)
        class_names = checkpoint['class_names']
        if isinstance(class_names, dict):
            for task, names in class_names.items():
                print(f"  {task}: {names}")
        else:
            print(f"  {class_names}")
    
    # 9. Any other keys not covered
    covered_keys = {'epoch', 'best_metric', 'multi_task', 'config', 'val_metrics', 
                   'metrics_history', 'model_state_dict', 'optimizer_state_dict', 
                   'scheduler_state_dict', 'class_names'}
    other_keys = set(checkpoint.keys()) - covered_keys
    
    if other_keys:
        print("\n9. OTHER DATA")
        print("-" * 50)
        for key in sorted(other_keys):
            value = checkpoint[key]
            if isinstance(value, (int, float, str, bool)):
                print(f"  {key}: {value}")
            elif value is None:
                print(f"  {key}: None")
            else:
                print(f"  {key}: {type(value).__name__}")
                if hasattr(value, '__len__'):
                    print(f"    Length: {len(value)}")


def main():
    parser = argparse.ArgumentParser(description='Display ALL data from training checkpoints')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint file')
    parser.add_argument('--experiment', type=str, help='Name of experiment folder')
    parser.add_argument('--exp-dir', type=str, default='experiments', help='Base experiments directory')
    parser.add_argument('--best', action='store_true', help='Load best checkpoint (default)')
    parser.add_argument('--last', action='store_true', help='Load last checkpoint')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    checkpoint_path = None
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.experiment:
        exp_path = Path(args.exp_dir) / args.experiment / 'checkpoints'
        if args.last:
            checkpoint_path = exp_path / 'last_checkpoint.pth'
        else:  # Default to best
            checkpoint_path = exp_path / 'best_checkpoint.pth'
            if not checkpoint_path.exists():
                checkpoint_path = exp_path / 'last_checkpoint.pth'
    else:
        print("Please specify --experiment or --checkpoint")
        return
    
    if not checkpoint_path or not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Analyze the checkpoint
    analyze_checkpoint(checkpoint_path)
    
    print("\n" + "=" * 100)
    print("END OF COMPLETE CHECKPOINT ANALYSIS")
    print("=" * 100)


if __name__ == '__main__':
    main()