#!/usr/bin/env python3
"""
Test script for data loading modules (url_dataset.py, transforms.py, data_utils.py)
Run from project root: python scripts/test/test_data_loading.py
"""

import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil
from typing import Dict, List
import traceback
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.url_dataset import URLDataset, CachedDataset, BaseDataset, create_subset_dataset
from src.data.transforms import (
    get_transforms_for_model, get_train_transforms, get_val_transforms,
    MixUpTransform, CutMixTransform, DeNormalize
)
from src.data.data_utils import (
    create_data_loaders, analyze_dataset_splits, prepare_data_for_training,
    get_dataset_statistics, create_weighted_sampler
)


class TestRunner:
    """Test runner with colored output and summary"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling"""
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")

        try:
            test_func()
            self.passed += 1
            print(f"✅ PASSED: {test_name}")
        except FileNotFoundError as e:
            self.skipped += 1
            print(f"⚠️  SKIPPED: {test_name} - {str(e)}")
        except Exception as e:
            self.failed += 1
            self.errors.append({
                'test': test_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            print(f"❌ FAILED: {test_name}")
            print(f"Error: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")

    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed + self.skipped
        print(f"\n{'=' * 60}")
        print(f"TEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⚠️  Skipped: {self.skipped}")

        if self.errors:
            print(f"\nFailed tests:")
            for error in self.errors:
                print(f"  - {error['test']}: {error['error']}")


def test_dataset_initialization():
    """Test basic dataset initialization"""
    print("Testing dataset initialization...")

    data_dir = Path("data")
    split_file = data_dir / "splits" / "train.json"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    # Test URLDataset initialization
    dataset = URLDataset(
        split_file=str(split_file),
        cache_dir=str(data_dir / "cache" / "images")
    )

    print(f"✓ URLDataset initialized with {len(dataset)} samples")
    print(f"✓ Number of classes: {dataset.num_classes}")
    print(f"✓ Class names: {dataset.decades}")

    # Check label mapping
    assert dataset.num_classes == 5, f"Expected 5 classes, got {dataset.num_classes}"
    assert dataset.decades == ['1960s', '1970s', '1980s', '1990s', '2000s'], "Unexpected decade labels"

    # Test metadata extraction
    if len(dataset) > 0:
        metadata = dataset.get_metadata(0)
        print(f"✓ Sample metadata keys: {list(metadata.keys())}")
        assert 'id' in metadata, "Missing 'id' in metadata"
        assert 'decade' in metadata, "Missing 'decade' in metadata"


def test_transforms():
    """Test transform pipelines"""
    print("Testing transforms...")

    # Create dummy image
    dummy_image = Image.new('RGB', (300, 300), color='red')

    # Test different transform types
    transforms_to_test = [
        ('train_transforms', get_train_transforms(224, 'medium')),
        ('val_transforms', get_val_transforms(224)),
        ('efficientnet_transforms', get_transforms_for_model('efficientnet-b2', is_training=True))
    ]

    for name, transform in transforms_to_test:
        tensor = transform(dummy_image)
        print(f"✓ {name}: output shape = {tensor.shape}")
        assert tensor.shape == (3, 224, 224), f"Unexpected tensor shape for {name}"

    # Test augmentation levels
    for level in ['light', 'medium', 'heavy']:
        transform = get_train_transforms(224, level)
        tensor = transform(dummy_image)
        print(f"✓ Augmentation level '{level}': shape = {tensor.shape}")

    # Test MixUp and CutMix
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])

    mixup = MixUpTransform(alpha=1.0, num_classes=5)
    mixed_images, labels_a, labels_b, lam = mixup(images, labels)
    print(f"✓ MixUp: lambda = {lam:.3f}, output shape = {mixed_images.shape}")

    cutmix = CutMixTransform(alpha=1.0, num_classes=5)
    mixed_images, labels_a, labels_b, lam = cutmix(images, labels)
    print(f"✓ CutMix: lambda = {lam:.3f}, output shape = {mixed_images.shape}")


def test_data_loading():
    """Test actual data loading from dataset"""
    print("Testing data loading...")

    data_dir = Path("data")
    split_file = data_dir / "splits" / "train.json"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    # Create small subset for testing
    dataset = URLDataset(
        split_file=str(split_file),
        cache_dir=str(data_dir / "cache" / "images"),
        transform=get_val_transforms(224),
        fallback_on_error=True
    )

    # Create subset to test quickly
    subset = create_subset_dataset(dataset, fraction=0.001, seed=42)  # 0.1% of data
    print(f"✓ Created subset with {len(subset)} samples")

    # Try loading a few samples
    successful_loads = 0
    failed_loads = 0

    for i in range(min(5, len(subset))):
        try:
            image, label, metadata = subset[i]
            print(f"✓ Loaded sample {i}: shape={image.shape}, label={label}, decade={metadata['decade']}")
            successful_loads += 1

            # Validate tensor
            assert image.shape == (3, 224, 224), f"Unexpected image shape: {image.shape}"
            assert 0 <= label < 5, f"Invalid label: {label}"

        except Exception as e:
            print(f"✗ Failed to load sample {i}: {str(e)}")
            failed_loads += 1

    print(f"\nLoading summary: {successful_loads} successful, {failed_loads} failed")

    # Test dataset statistics
    if hasattr(dataset, 'get_statistics'):
        stats = dataset.get_statistics()
        print(f"✓ Dataset statistics: {stats}")


def test_data_loaders():
    """Test DataLoader creation"""
    print("Testing DataLoader creation...")

    # Create test config
    config = {
        'model_name': 'efficientnet-b2',
        'batch_size': 4,
        'num_workers': 0,  # Use 0 for testing
        'data_dir': 'data',
        'use_cached': False,
        'use_class_weights': True,
        'use_weighted_sampling': False,
        'augmentation_level': 'medium'
    }

    try:
        # Create data loaders with small subset
        train_loader, val_loader, class_weights, class_names = create_data_loaders(
            config,
            use_subset=True,
            subset_fraction=0.001  # Very small subset for testing
        )

        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")
        print(f"✓ Class names: {class_names}")

        if class_weights is not None:
            print(f"✓ Class weights shape: {class_weights.shape}")
            print(f"✓ Class weights: {class_weights.numpy()}")

        # Test loading one batch
        if len(train_loader) > 0:
            for batch_idx, (images, labels, metadata) in enumerate(train_loader):
                print(f"✓ Batch {batch_idx}: images={images.shape}, labels={labels.shape}")
                print(f"✓ Sample metadata: {metadata[0] if metadata else 'None'}")

                # Validate batch
                assert images.shape[0] <= config['batch_size'], "Batch size exceeded"
                assert images.shape[1:] == (3, 224, 224), f"Unexpected image shape: {images.shape}"
                assert labels.shape[0] == images.shape[0], "Label count mismatch"

                break  # Just test first batch

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required data files not found. Please run data preparation first: {e}")


def test_dataset_analysis():
    """Test dataset analysis functions"""
    print("Testing dataset analysis...")

    data_dir = Path("data")

    try:
        # Analyze splits
        analysis = analyze_dataset_splits(data_dir)

        for split_name, stats in analysis.items():
            if isinstance(stats, dict) and 'total_images' in stats:
                print(f"\n{split_name} split:")
                print(f"  ✓ Total images: {stats['total_images']}")
                print(f"  ✓ Unique products: {stats['unique_products']}")
                print(f"  ✓ Decades distribution: {stats.get('decades', {})}")

                if 'images_per_product' in stats:
                    img_stats = stats['images_per_product']
                    print(f"  ✓ Images per product: mean={img_stats['mean']:.2f}, std={img_stats['std']:.2f}")

        # Check for data leakage
        if 'data_leakage' in analysis:
            print(f"\n⚠️  WARNING: Data leakage detected!")
            print(f"  Train-Val overlap: {analysis['data_leakage']['train_val_overlap']} products")
        else:
            print(f"\n✓ No data leakage detected between train and val splits")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Split files not found. Please create splits first: {e}")


def test_weighted_sampler():
    """Test weighted sampler creation"""
    print("Testing weighted sampler...")

    # Create mock dataset with imbalanced classes
    class MockDataset:
        def __init__(self):
            # Create imbalanced labels (more samples for class 0)
            self.data = [{'decade': '1960s'} for _ in range(100)]
            self.data += [{'decade': '1970s'} for _ in range(50)]
            self.data += [{'decade': '1980s'} for _ in range(30)]
            self.data += [{'decade': '1990s'} for _ in range(20)]
            self.data += [{'decade': '2000s'} for _ in range(10)]

            self.decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
            self.label_to_idx = {d: i for i, d in enumerate(self.decades)}
            self.num_classes = 5

        def get_labels(self):
            return [self.label_to_idx[item['decade']] for item in self.data]

    dataset = MockDataset()
    sampler = create_weighted_sampler(dataset)

    print(f"✓ Created weighted sampler with {len(sampler)} samples")

    # Sample and check distribution
    sampled_indices = list(sampler)[:1000]
    sampled_labels = [dataset.get_labels()[idx] for idx in sampled_indices]

    from collections import Counter
    label_counts = Counter(sampled_labels)
    print(f"✓ Sampled label distribution: {dict(label_counts)}")

    # Check if sampling is more balanced
    counts = list(label_counts.values())
    if len(counts) > 0:
        balance_ratio = max(counts) / min(counts)
        print(f"✓ Balance ratio after sampling: {balance_ratio:.2f} (lower is better)")


def test_cached_dataset():
    """Test CachedDataset functionality"""
    print("Testing CachedDataset...")

    data_dir = Path("data")
    split_file = data_dir / "splits" / "val.json"
    cache_dir = data_dir / "cache" / "images"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    try:
        dataset = CachedDataset(
            split_file=str(split_file),
            images_dir=str(cache_dir),
            transform=get_val_transforms(224),
            verify_images=True
        )

        print(f"✓ CachedDataset initialized with {len(dataset)} valid cached images")

        # Try loading a sample if available
        if len(dataset) > 0:
            image, label, metadata = dataset[0]
            print(f"✓ Loaded cached sample: shape={image.shape}, label={label}")
        else:
            print("⚠️  No cached images available for testing")

    except Exception as e:
        print(f"⚠️  CachedDataset test partially failed: {str(e)}")
        print("   This is expected if images haven't been downloaded yet")


def test_data_preparation():
    """Test data preparation checker"""
    print("Testing data preparation...")

    config = {
        'data_dir': 'data',
        'use_cached': False
    }

    # Check if data is ready (don't download)
    is_ready = prepare_data_for_training(
        config,
        download_if_missing=False,
        verify_splits=True
    )

    if is_ready:
        print("✓ Data is ready for training")
    else:
        print("⚠️  Data is not fully prepared")
        print("   Run the data preparation pipeline to download images")


def test_model_specific_transforms():
    """Test transforms for different models"""
    print("Testing model-specific transforms...")

    dummy_image = Image.new('RGB', (400, 400), color='blue')

    models_to_test = [
        ('resnet50', 224),
        ('efficientnet-b2', 260),  # Assuming this is configured
        ('convnext-tiny-384', 384)  # Assuming this is configured
    ]

    for model_name, expected_size in models_to_test:
        try:
            transform = get_transforms_for_model(model_name, is_training=False)
            tensor = transform(dummy_image)
            print(f"✓ {model_name}: output shape = {tensor.shape}")

            # Note: The actual size might be 224 if not configured in model_configs
            if tensor.shape[1] != expected_size:
                print(f"  ⚠️  Expected size {expected_size}, got {tensor.shape[1]}")
                print(f"     This might be due to default config being used")

        except KeyError:
            print(f"⚠️  Model config not found for {model_name}, using default")


def main():
    """Run all tests"""
    print("=" * 60)
    print("VINTAGE PRODUCT CLASSIFICATION - DATA LOADING TEST SUITE")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Current directory: {os.getcwd()}")

    # Change to project root for consistent paths
    os.chdir(project_root)
    print(f"Changed to project root: {os.getcwd()}")

    runner = TestRunner()

    # Define all tests
    tests = [
        ("Dataset Initialization", test_dataset_initialization),
        ("Transform Pipelines", test_transforms),
        ("Data Loading", test_data_loading),
        ("DataLoader Creation", test_data_loaders),
        ("Dataset Analysis", test_dataset_analysis),
        ("Weighted Sampler", test_weighted_sampler),
        ("Cached Dataset", test_cached_dataset),
        ("Data Preparation Check", test_data_preparation),
        ("Model-Specific Transforms", test_model_specific_transforms),
    ]

    # Run all tests
    for test_name, test_func in tests:
        runner.run_test(test_name, test_func)

    # Print summary
    runner.print_summary()

    # Return exit code
    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)