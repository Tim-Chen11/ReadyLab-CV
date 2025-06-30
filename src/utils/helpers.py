import torch
import numpy as np
import random
import os
from pathlib import Path
import json
import yaml
from typing import Dict, Any, Optional, List
import hashlib
import shutil
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get torch device

    Args:
        gpu_id: Specific GPU ID to use

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'total_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
        'trainable_mb': trainable_params * 4 / 1024 / 1024
    }


def save_config(config: Dict[str, Any], save_path: Path, format: str = 'json'):
    """
    Save configuration to file

    Args:
        config: Configuration dictionary
        save_path: Path to save file
        format: File format ('json' or 'yaml')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    elif format == 'yaml':
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown config format: {config_path.suffix}")

    return config


def merge_configs(base_config: Dict, update_config: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries

    Args:
        base_config: Base configuration
        update_config: Configuration with updates

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in update_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_experiment_name(
        model_name: str,
        dataset_name: str = 'decade',
        timestamp: bool = True
) -> str:
    """
    Generate experiment name

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        timestamp: Whether to include timestamp

    Returns:
        Experiment name
    """
    name_parts = [model_name, dataset_name]

    if timestamp:
        name_parts.append(get_timestamp())

    return '_'.join(name_parts)


def compute_file_hash(file_path: Path, hash_algo: str = 'md5') -> str:
    """
    Compute hash of a file

    Args:
        file_path: Path to file
        hash_algo: Hash algorithm to use

    Returns:
        Hex digest of file hash
    """
    hash_func = getattr(hashlib, hash_algo)()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def create_experiment_structure(base_dir: Path, experiment_name: str) -> Dict[str, Path]:
    """
    Create directory structure for an experiment

    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment

    Returns:
        Dictionary mapping directory names to paths
    """
    exp_dir = base_dir / experiment_name

    dirs = {
        'root': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',
        'logs': exp_dir / 'logs',
        'visualizations': exp_dir / 'visualizations',
        'predictions': exp_dir / 'predictions',
        'configs': exp_dir / 'configs'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def backup_code(src_dir: Path, backup_dir: Path, extensions: List[str] = None):
    """
    Backup source code to experiment directory

    Args:
        src_dir: Source directory
        backup_dir: Backup destination
        extensions: List of file extensions to backup
    """
    if extensions is None:
        extensions = ['.py', '.yaml', '.yml', '.json', '.txt', '.md']

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    for file_path in src_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            relative_path = file_path.relative_to(src_dir)
            backup_path = backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage

    Returns:
        Dictionary with memory usage in MB
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0}

    return {
        'allocated': torch.cuda.memory_allocated() / 1024 / 1024,
        'reserved': torch.cuda.memory_reserved() / 1024 / 1024
    }


def clean_checkpoint(checkpoint_path: Path, keep_keys: List[str] = None):
    """
    Clean checkpoint file by keeping only specified keys

    Args:
        checkpoint_path: Path to checkpoint
        keep_keys: Keys to keep (default: model_state_dict only)
    """
    if keep_keys is None:
        keep_keys = ['model_state_dict']

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cleaned_checkpoint = {k: v for k, v in checkpoint.items() if k in keep_keys}

    # Save cleaned checkpoint
    output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_cleaned.pth"
    torch.save(cleaned_checkpoint, output_path)

    # Print size reduction
    original_size = checkpoint_path.stat().st_size / 1024 / 1024
    new_size = output_path.stat().st_size / 1024 / 1024
    print(f"Cleaned checkpoint: {original_size:.1f}MB → {new_size:.1f}MB")

    return output_path


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


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.val_loss_min = val_loss