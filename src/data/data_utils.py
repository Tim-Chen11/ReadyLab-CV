import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import logging

from .url_dataset import URLDataset, CachedDataset, BaseDataset
from .transforms import get_transforms_for_model
from ..training.metrics import calculate_class_weights

logger = logging.getLogger(__name__)


def create_data_loaders(
        config: Dict,
        data_dir: Optional[Path] = None,
        use_subset: bool = False,
        subset_fraction: float = 0.1
) -> Tuple[DataLoader, DataLoader, torch.Tensor, List[str]]:
    """
    Create train and validation data loaders

    Args:
        config: Configuration dictionary
        data_dir: Data directory path
        use_subset: Whether to use a subset for quick testing
        subset_fraction: Fraction of data to use if use_subset is True

    Returns:
        train_loader, val_loader, class_weights, class_names
    """
    if data_dir is None:
        data_dir = Path(config.get('data_dir', '../data'))

    # Get transforms
    train_transform = get_transforms_for_model(
        config['model_name'],
        is_training=True,
        augmentation_level=config.get('augmentation_level', 'medium')
    )
    val_transform = get_transforms_for_model(
        config['model_name'],
        is_training=False
    )

    # Choose dataset class
    dataset_class = CachedDataset if config.get('use_cached', False) else URLDataset

    # Create datasets
    if dataset_class == URLDataset:
        train_dataset = URLDataset(
            split_file=data_dir / 'splits' / 'train.json',
            transform=train_transform,
            cache_dir=data_dir / 'cache' / 'images',
            max_retries=config.get('max_download_retries', 3),
            timeout=config.get('download_timeout', 10)
        )
        val_dataset = URLDataset(
            split_file=data_dir / 'splits' / 'val.json',
            transform=val_transform,
            cache_dir=data_dir / 'cache' / 'images',
            max_retries=config.get('max_download_retries', 3),
            timeout=config.get('download_timeout', 10)
        )
    else:
        train_dataset = CachedDataset(
            split_file=data_dir / 'splits' / 'train.json',
            images_dir=data_dir / 'cache' / 'images',
            transform=train_transform,
            verify_images=True
        )
        val_dataset = CachedDataset(
            split_file=data_dir / 'splits' / 'val.json',
            images_dir=data_dir / 'cache' / 'images',
            transform=val_transform,
            verify_images=True
        )

    # Create subset if requested
    if use_subset:
        from .url_dataset import create_subset_dataset
        train_dataset = create_subset_dataset(train_dataset, subset_fraction)
        val_dataset = create_subset_dataset(val_dataset, subset_fraction)
        logger.info(f"Using subset with {subset_fraction * 100}% of data")

    # Calculate class weights if needed
    class_weights = None
    if config.get('use_class_weights', False):
        labels = train_dataset.get_labels()
        class_weights = calculate_class_weights(
            labels,
            train_dataset.num_classes,
            method=config.get('class_weight_method', 'inverse_frequency')
        )
        logger.info(f"Class weights: {class_weights.numpy()}")

    # Create sampler if using weighted sampling
    train_sampler = None
    if config.get('use_weighted_sampling', False):
        train_sampler = create_weighted_sampler(train_dataset)
        shuffle = False  # Don't shuffle when using sampler
    else:
        shuffle = True

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.get('num_workers', 4) > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        persistent_workers=config.get('num_workers', 4) > 0
    )

    return train_loader, val_loader, class_weights, train_dataset.decades


def create_weighted_sampler(dataset: BaseDataset) -> WeightedRandomSampler:
    """
    Create a weighted sampler for balanced batch sampling

    Args:
        dataset: Dataset instance

    Returns:
        WeightedRandomSampler instance
    """
    # Get all labels
    labels = dataset.get_labels()

    # Count occurrences
    class_counts = Counter(labels)

    # Calculate weights for each sample
    weights = []
    for label in labels:
        weight = 1.0 / class_counts[label]
        weights.append(weight)

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    logger.info(f"Created weighted sampler with class counts: {dict(class_counts)}")

    return sampler


def analyze_dataset_splits(data_dir: Path) -> Dict:
    """
    Analyze train/val/test splits

    Args:
        data_dir: Data directory containing splits

    Returns:
        Dictionary with analysis results
    """
    splits_dir = data_dir / 'splits'
    analysis = {}

    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f'{split_name}.json'

        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue

        with open(split_file, 'r') as f:
            data = json.load(f)

        # Analyze split
        split_analysis = {
            'total_images': len(data),
            'unique_products': len(set(item['product_id'] for item in data)),
            'decades': defaultdict(int),
            'classifications': defaultdict(int),
            'countries': defaultdict(int),
            'images_per_product': defaultdict(int)
        }

        # Count by various attributes
        product_image_count = defaultdict(int)

        for item in data:
            split_analysis['decades'][item['decade']] += 1
            split_analysis['classifications'][item.get('classification', 'unknown')] += 1
            split_analysis['countries'][item.get('country', 'unknown')] += 1
            product_image_count[item['product_id']] += 1

        # Statistics on images per product
        image_counts = list(product_image_count.values())
        split_analysis['images_per_product'] = {
            'mean': np.mean(image_counts),
            'std': np.std(image_counts),
            'min': min(image_counts),
            'max': max(image_counts),
            'distribution': Counter(image_counts)
        }

        # Convert defaultdicts to regular dicts
        split_analysis['decades'] = dict(split_analysis['decades'])
        split_analysis['classifications'] = dict(split_analysis['classifications'])
        split_analysis['countries'] = dict(split_analysis['countries'])

        analysis[split_name] = split_analysis

    # Check for data leakage
    if 'train' in analysis and 'val' in analysis:
        train_products = set()
        val_products = set()

        with open(splits_dir / 'train.json', 'r') as f:
            train_data = json.load(f)
            train_products = set(item['product_id'] for item in train_data)

        with open(splits_dir / 'val.json', 'r') as f:
            val_data = json.load(f)
            val_products = set(item['product_id'] for item in val_data)

        overlap = train_products.intersection(val_products)
        if overlap:
            logger.warning(f"Found {len(overlap)} products in both train and val splits!")
            analysis['data_leakage'] = {
                'train_val_overlap': len(overlap),
                'overlapping_products': list(overlap)[:10]  # Show first 10
            }

    return analysis


def create_test_loader(
        config: Dict,
        data_dir: Optional[Path] = None,
        batch_size: Optional[int] = None
) -> Tuple[DataLoader, List[str]]:
    """
    Create test data loader

    Args:
        config: Configuration dictionary
        data_dir: Data directory path
        batch_size: Batch size (uses config value if None)

    Returns:
        test_loader, class_names
    """
    if data_dir is None:
        data_dir = Path(config.get('data_dir', '../data'))

    if batch_size is None:
        batch_size = config.get('batch_size', 32)

    # Get test transform (same as validation)
    test_transform = get_transforms_for_model(
        config['model_name'],
        is_training=False
    )

    # Choose dataset class
    dataset_class = CachedDataset if config.get('use_cached', False) else URLDataset

    # Create test dataset
    if dataset_class == URLDataset:
        test_dataset = URLDataset(
            split_file=data_dir / 'splits' / 'test.json',
            transform=test_transform,
            cache_dir=data_dir / 'cache' / 'images'
        )
    else:
        test_dataset = CachedDataset(
            split_file=data_dir / 'splits' / 'test.json',
            images_dir=data_dir / 'cache' / 'images',
            transform=test_transform
        )

    # Create loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return test_loader, test_dataset.decades


def prepare_data_for_training(
        config: Dict,
        download_if_missing: bool = True,
        verify_splits: bool = True
) -> bool:
    """
    Prepare data for training by checking splits and optionally downloading

    Args:
        config: Configuration dictionary
        download_if_missing: Whether to download missing images
        verify_splits: Whether to verify data splits

    Returns:
        True if data is ready, False otherwise
    """
    data_dir = Path(config.get('data_dir', '../data'))

    # Check if splits exist
    splits_dir = data_dir / 'splits'
    required_splits = ['train.json', 'val.json', 'test.json']

    missing_splits = []
    for split_file in required_splits:
        if not (splits_dir / split_file).exists():
            missing_splits.append(split_file)

    if missing_splits:
        logger.error(f"Missing split files: {missing_splits}")
        logger.error(f"Please run the data preparation pipeline first")
        return False

    # Verify splits if requested
    if verify_splits:
        logger.info("Analyzing dataset splits...")
        analysis = analyze_dataset_splits(data_dir)

        # Check for issues
        if 'data_leakage' in analysis:
            logger.warning("Data leakage detected between splits!")
            return False

        # Log split statistics
        for split_name, split_stats in analysis.items():
            if isinstance(split_stats, dict) and 'total_images' in split_stats:
                logger.info(f"{split_name}: {split_stats['total_images']} images, "
                            f"{split_stats['unique_products']} products")

    # Check cache directory
    cache_dir = data_dir / 'cache' / 'images'

    if config.get('use_cached', False):
        # Count cached images
        cached_images = list(cache_dir.glob('*.jpg'))
        logger.info(f"Found {len(cached_images)} cached images")

        if len(cached_images) == 0:
            logger.error("No cached images found but use_cached=True")

            if download_if_missing:
                logger.info("Downloading all images...")
                from .url_dataset import download_dataset_images

                split_files = [splits_dir / f for f in required_splits]
                stats = download_dataset_images(
                    split_files,
                    cache_dir,
                    num_workers=8
                )

                if stats['failed'] > stats['downloaded'] * 0.1:
                    logger.warning(f"High failure rate: {stats['failed']} failures")
                    return False
            else:
                return False

    logger.info("Data preparation complete!")
    return True


def get_dataset_statistics(dataset: BaseDataset) -> Dict:
    """
    Get detailed statistics about a dataset

    Args:
        dataset: Dataset instance

    Returns:
        Dictionary with statistics
    """
    # Basic counts
    stats = {
        'total_samples': len(dataset),
        'num_classes': dataset.num_classes,
        'class_names': dataset.decades
    }

    # Class distribution
    labels = dataset.get_labels()
    class_counts = Counter(labels)

    stats['class_distribution'] = {
        dataset.idx_to_label[idx]: count
        for idx, count in class_counts.items()
    }

    # Class balance metrics
    counts = list(class_counts.values())
    stats['class_balance'] = {
        'min_samples': min(counts),
        'max_samples': max(counts),
        'imbalance_ratio': max(counts) / min(counts),
        'std_dev': np.std(counts)
    }

    # Product statistics
    product_counts = Counter(item['product_id'] for item in dataset.data)
    stats['product_stats'] = {
        'unique_products': len(product_counts),
        'avg_images_per_product': np.mean(list(product_counts.values())),
        'max_images_per_product': max(product_counts.values()),
        'min_images_per_product': min(product_counts.values())
    }

    # Other metadata
    classifications = Counter(item.get('classification', 'unknown') for item in dataset.data)
    countries = Counter(item.get('country', 'unknown') for item in dataset.data)

    stats['top_classifications'] = dict(classifications.most_common(10))
    stats['top_countries'] = dict(countries.most_common(10))

    return stats


if __name__ == "__main__":
    # Test data utilities
    from ..models.model_configs import TRAINING_CONFIGS

    # Create test config
    config = TRAINING_CONFIGS['efficientnet-b2'].copy()
    config.update({
        'model_name': 'efficientnet-b2',
        'data_dir': '../data',
        'use_cached': False,
        'use_class_weights': True,
        'num_workers': 4
    })

    # Check data preparation
    print("Checking data preparation...")
    if prepare_data_for_training(config, download_if_missing=False):
        print("Data is ready!")

        # Create loaders
        print("\nCreating data loaders...")
        train_loader, val_loader, class_weights, class_names = create_data_loaders(
            config,
            use_subset=True,
            subset_fraction=0.01
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Class names: {class_names}")

        if class_weights is not None:
            print(f"Class weights: {class_weights}")

        # Test loading a batch
        print("\nTesting batch loading...")
        for batch_idx, (images, labels, metadata) in enumerate(train_loader):
            print(f"Batch {batch_idx}: images shape={images.shape}, labels shape={labels.shape}")
            print(f"Sample metadata: {metadata[0]}")
            break
    else:
        print("Data preparation failed!")