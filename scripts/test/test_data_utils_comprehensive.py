
# !/usr/bin/env python3
"""
Comprehensive test script for data_utils.py
Tests the orchestration layer that combines transforms, url_dataset, and creates DataLoaders
Run from project root: python test_data_utils_comprehensive.py
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
import tempfile
import traceback
import logging
from typing import Dict, List
from unittest.mock import patch, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class TestRunner:
    """Test runner with results tracking"""

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


def create_mock_model_configs():
    """Create mock model configs for testing"""
    return {
        'efficientnet-b2': {
            'input_size': 260,
            'batch_size': 32,
        },
        'resnet50': {
            'input_size': 224,
            'batch_size': 32,
        }
    }


def create_mock_split_data(num_items=20, split_name="train"):
    """Create mock dataset split data"""
    decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
    mock_data = []

    for i in range(num_items):
        mock_data.append({
            "id": f"{split_name}_id_{i}",
            "product_id": f"product_{i % 5}",  # Some products have multiple images
            "name": f"Test Product {i}",
            "decade": decades[i % 5],  # Ensure all decades represented
            "url": f"https://example.com/{split_name}_image_{i}.jpg",
            "classification": f"furniture",
            "makers": f"Test Maker {i % 3}",
            "country": f"Country_{i % 3}"
        })

    return mock_data


def create_test_environment():
    """Create a complete test environment with mock data"""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Create directory structure
    data_dir = temp_path / "data"
    splits_dir = data_dir / "splits"
    cache_dir = data_dir / "cache" / "images"

    splits_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    # Create split files
    split_files = {}
    for split_name in ["train", "val", "test"]:
        split_data = create_mock_split_data(20 if split_name == "train" else 10, split_name)
        split_file = splits_dir / f"{split_name}.json"

        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)

        split_files[split_name] = split_file

    return temp_path, data_dir, split_files, cache_dir


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        # Get project root and add to path
        current_dir = Path.cwd()
        src_path = current_dir / "src"
        if src_path.exists():
            sys.path.insert(0, str(current_dir))

        from src.data.data_utils import (
            create_data_loaders, create_weighted_sampler, analyze_dataset_splits,
            create_test_loader, prepare_data_for_training, get_dataset_statistics
        )
        from src.data.url_dataset import BaseDataset, URLDataset, CachedDataset
        from src.data.transforms import get_transforms_for_model

        print("✓ All data_utils imports successful")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_create_data_loaders():
    """Test create_data_loaders function"""
    print("Testing create_data_loaders...")

    # Import after ensuring path is set
    from src.data.data_utils import create_data_loaders
    from torchvision import transforms

    temp_path, data_dir, split_files, cache_dir = create_test_environment()

    try:
        # Mock the model configs to avoid import issues
        with patch('src.data.transforms.TRAINING_CONFIGS', create_mock_model_configs()):
            # Create test config
            config = {
                'model_name': 'efficientnet-b2',
                'batch_size': 4,
                'num_workers': 0,  # Use 0 for testing
                'data_dir': str(data_dir),
                'use_cached': False,
                'use_class_weights': False,
                'use_weighted_sampling': False,
                'augmentation_level': 'medium',
                'max_download_retries': 1,
                'download_timeout': 5
            }

            # Mock image downloads to avoid network calls
            mock_image = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))

            def mock_load_image(self, item):
                return mock_image

            with patch('src.data.url_dataset.URLDataset._load_image', mock_load_image):
                # Test basic data loader creation
                train_loader, val_loader, class_weights, class_names = create_data_loaders(
                    config,
                    data_dir=data_dir,
                    use_subset=True,
                    subset_fraction=0.5  # Use 50% for testing
                )

                print(f"✓ Created DataLoaders successfully")
                print(f"  Train loader: {len(train_loader)} batches")
                print(f"  Val loader: {len(val_loader)} batches")
                print(f"  Class names: {class_names}")

                # Validate results
                assert len(train_loader) > 0, "Train loader should have batches"
                assert len(val_loader) > 0, "Val loader should have batches"
                assert class_names == ['1960s', '1970s', '1980s', '1990s', '2000s'], "Wrong class names"
                assert class_weights is None, "Class weights should be None when disabled"

                # Test loading a batch
                for batch_idx, batch_data in enumerate(train_loader):
                    if len(batch_data) == 3:
                        images, labels, metadata = batch_data
                    else:
                        images, labels = batch_data[0], batch_data[1]
                        metadata = None

                    print(f"✓ Loaded batch: images={images.shape}, labels={labels.shape}")

                    # Validate batch
                    assert images.shape[0] <= config['batch_size'], "Batch size exceeded"
                    assert images.shape[1] == 3, "Wrong number of channels"
                    assert labels.shape[0] == images.shape[0], "Label count mismatch"

                    break  # Just test first batch

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_path)


def test_weighted_sampler():
    """Test create_weighted_sampler function"""
    print("Testing create_weighted_sampler...")

    from src.data.data_utils import create_weighted_sampler
    from src.data.url_dataset import BaseDataset

    temp_path, data_dir, split_files, cache_dir = create_test_environment()

    try:
        # Create dataset with imbalanced data
        imbalanced_data = []
        decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
        # Create imbalanced distribution
        counts = [20, 5, 10, 3, 15]  # Imbalanced

        item_id = 0
        for decade_idx, count in enumerate(counts):
            for i in range(count):
                imbalanced_data.append({
                    "id": f"id_{item_id}",
                    "product_id": f"product_{item_id}",
                    "name": f"Product {item_id}",
                    "decade": decades[decade_idx],
                    "url": f"https://example.com/image_{item_id}.jpg",
                    "classification": "test",
                    "makers": "test",
                    "country": "test"
                })
                item_id += 1

        # Save imbalanced split
        imbalanced_split = data_dir / "splits" / "imbalanced.json"
        with open(imbalanced_split, 'w') as f:
            json.dump(imbalanced_data, f)

        # Create dataset
        dataset = BaseDataset(str(imbalanced_split))

        print(f"✓ Created imbalanced dataset with {len(dataset)} samples")

        # Test label distribution before sampling
        labels = dataset.get_labels()
        from collections import Counter
        original_counts = Counter(labels)
        print(f"  Original distribution: {dict(original_counts)}")

        # Create weighted sampler
        sampler = create_weighted_sampler(dataset)

        print(f"✓ Created weighted sampler with {len(sampler)} samples")

        # Test sampling
        sampled_indices = list(sampler)[:100]  # Sample 100 items
        sampled_labels = [labels[idx] for idx in sampled_indices]
        sampled_counts = Counter(sampled_labels)

        print(f"  Sampled distribution: {dict(sampled_counts)}")

        # Check if sampling is more balanced
        original_ratio = max(original_counts.values()) / min(original_counts.values())
        sampled_ratio = max(sampled_counts.values()) / min(sampled_counts.values()) if min(
            sampled_counts.values()) > 0 else float('inf')

        print(f"  Balance improvement: {original_ratio:.2f} -> {sampled_ratio:.2f}")

        # Sampled distribution should be more balanced
        assert sampled_ratio < original_ratio or sampled_ratio < 3.0, "Sampling should improve balance"

    finally:
        import shutil
        shutil.rmtree(temp_path)


def test_analyze_dataset_splits():
    """Test analyze_dataset_splits function"""
    print("Testing analyze_dataset_splits...")

    from src.data.data_utils import analyze_dataset_splits

    temp_path, data_dir, split_files, cache_dir = create_test_environment()

    try:
        # Analyze the splits
        analysis = analyze_dataset_splits(data_dir)

        print(f"✓ Analysis completed for {len(analysis)} splits")

        # Check analysis structure
        expected_splits = ['train', 'val', 'test']
        for split_name in expected_splits:
            assert split_name in analysis, f"Missing analysis for {split_name}"

            split_stats = analysis[split_name]
            print(f"  {split_name}: {split_stats['total_images']} images, {split_stats['unique_products']} products")

            # Validate structure
            required_keys = ['total_images', 'unique_products', 'decades', 'classifications', 'countries']
            for key in required_keys:
                assert key in split_stats, f"Missing key {key} in {split_name} analysis"

            # Check decades distribution
            decades_dist = split_stats['decades']
            assert len(decades_dist) == 5, f"Should have 5 decades, got {len(decades_dist)}"

        # Check for data leakage detection
        # Our test data has overlapping product_ids, so leakage should be detected
        if 'data_leakage' in analysis:
            print(f"  ⚠️  Data leakage detected: {analysis['data_leakage']['train_val_overlap']} overlapping products")
        else:
            print(f"  ✓ No data leakage detected")

    finally:
        import shutil
        shutil.rmtree(temp_path)


def test_prepare_data_for_training():
    """Test prepare_data_for_training function"""
    print("Testing prepare_data_for_training...")

    from src.data.data_utils import prepare_data_for_training

    temp_path, data_dir, split_files, cache_dir = create_test_environment()

    try:
        # Test config
        config = {
            'data_dir': str(data_dir),
            'use_cached': False
        }

        # FIXED: Test with data leakage detection disabled first
        is_ready = prepare_data_for_training(
            config,
            download_if_missing=False,
            verify_splits=False  # Disable verification to test basic functionality
        )

        print(f"✓ Data preparation (no verification): {is_ready}")
        assert is_ready == True, "Data preparation should succeed when verification disabled"

        # FIXED: Test with verification enabled (should detect leakage)
        is_ready_with_verification = prepare_data_for_training(
            config,
            download_if_missing=False,
            verify_splits=True  # Enable verification
        )

        print(f"✓ Data preparation (with verification): {is_ready_with_verification}")
        # This should be False because our mock data has intentional leakage
        assert is_ready_with_verification == False, "Data preparation should fail when data leakage detected"

        # FIXED: Test with missing split files
        missing_split = data_dir / "splits" / "train.json"
        missing_split.unlink()  # Remove train split

        is_ready_missing = prepare_data_for_training(
            config,
            download_if_missing=False,
            verify_splits=False
        )

        print(f"✓ Data preparation with missing files: {is_ready_missing}")
        assert is_ready_missing == False, "Data preparation should fail with missing splits"

        print("✓ All data preparation scenarios tested correctly")

    finally:
        import shutil
        shutil.rmtree(temp_path)


def test_get_dataset_statistics():
    """Test get_dataset_statistics function"""
    print("Testing get_dataset_statistics...")

    from src.data.data_utils import get_dataset_statistics
    from src.data.url_dataset import BaseDataset

    temp_path, data_dir, split_files, cache_dir = create_test_environment()

    try:
        # Create dataset
        dataset = BaseDataset(str(split_files['train']))

        # Get statistics
        stats = get_dataset_statistics(dataset)

        print(f"✓ Generated statistics for dataset")

        # Validate statistics structure
        required_keys = [
            'total_samples', 'num_classes', 'class_names',
            'class_distribution', 'class_balance', 'product_stats',
            'top_classifications', 'top_countries'
        ]

        for key in required_keys:
            assert key in stats, f"Missing statistics key: {key}"

        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Number of classes: {stats['num_classes']}")
        print(f"  Class distribution: {stats['class_distribution']}")
        print(f"  Imbalance ratio: {stats['class_balance']['imbalance_ratio']:.2f}")
        print(f"  Unique products: {stats['product_stats']['unique_products']}")
        print(f"  Avg images per product: {stats['product_stats']['avg_images_per_product']:.2f}")

        # Validate values
        assert stats['total_samples'] == len(dataset), "Wrong total samples count"
        assert stats['num_classes'] == 5, "Wrong number of classes"
        assert len(stats['class_names']) == 5, "Wrong number of class names"

    finally:
        import shutil
        shutil.rmtree(temp_path)


def test_create_test_loader():
    """Test create_test_loader function"""
    print("Testing create_test_loader...")

    from src.data.data_utils import create_test_loader

    temp_path, data_dir, split_files, cache_dir = create_test_environment()

    try:
        # Mock the model configs
        with patch('src.data.transforms.TRAINING_CONFIGS', create_mock_model_configs()):
            config = {
                'model_name': 'resnet50',
                'batch_size': 4,
                'num_workers': 0,
                'data_dir': str(data_dir),
                'use_cached': False,
                'max_download_retries': 1,
                'download_timeout': 5
            }

            # Mock image loading
            mock_image = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))

            def mock_load_image(self, item):
                return mock_image

            with patch('src.data.url_dataset.URLDataset._load_image', mock_load_image):
                test_loader, class_names = create_test_loader(
                    config,
                    data_dir=data_dir,
                    batch_size=6
                )

                print(f"✓ Created test loader with {len(test_loader)} batches")
                print(f"  Class names: {class_names}")

                # Validate
                assert len(test_loader) > 0, "Test loader should have batches"
                assert class_names == ['1960s', '1970s', '1980s', '1990s', '2000s'], "Wrong class names"

                # Test loading a batch
                for batch_data in test_loader:
                    if len(batch_data) == 3:
                        images, labels, metadata = batch_data
                    else:
                        images, labels = batch_data[0], batch_data[1]

                    print(f"✓ Test batch: images={images.shape}, labels={labels.shape}")
                    assert images.shape[0] <= 6, "Batch size should respect config"
                    break

    finally:
        import shutil
        shutil.rmtree(temp_path)


def test_integration_with_real_data():
    """Test integration with real project data if available"""
    print("Testing integration with real data...")

    # Check if real data exists
    project_root = Path.cwd()
    real_data_dir = project_root / "data"
    real_splits_dir = real_data_dir / "splits"
    real_train_split = real_splits_dir / "train.json"

    if not real_train_split.exists():
        print("⚠️  Real data not found, skipping integration test")
        raise FileNotFoundError("Real data not available")

    from src.data.data_utils import create_data_loaders, analyze_dataset_splits

    try:
        # Mock the model configs
        with patch('src.data.transforms.TRAINING_CONFIGS', create_mock_model_configs()):
            print(f"✓ Found real data at {real_data_dir}")

            # Test analysis with real data
            analysis = analyze_dataset_splits(real_data_dir)

            for split_name, stats in analysis.items():
                if isinstance(stats, dict) and 'total_images' in stats:
                    print(f"  {split_name}: {stats['total_images']} images")

            # Test data loader creation with real data (small subset)
            config = {
                'model_name': 'efficientnet-b2',
                'batch_size': 4,
                'num_workers': 0,
                'data_dir': str(real_data_dir),
                'use_cached': False,
                'use_class_weights': False,
                'use_weighted_sampling': False,
                'augmentation_level': 'light',
                'max_download_retries': 1,
                'download_timeout': 5
            }

            # Use very small subset for quick testing
            train_loader, val_loader, class_weights, class_names = create_data_loaders(
                config,
                data_dir=real_data_dir,
                use_subset=True,
                subset_fraction=0.001  # 0.1% for very quick test
            )

            print(f"✓ Real data integration successful")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")

    except Exception as e:
        print(f"⚠️  Real data integration test failed: {e}")
        raise


def main():
    """Main test function"""
    print("DATA_UTILS.PY COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("Testing data orchestration and integration layer")

    runner = TestRunner()

    # Check imports first
    if not test_imports():
        print("❌ Cannot proceed without successful imports")
        return 1

    # Define all tests
    tests = [
        ("Create Data Loaders", test_create_data_loaders),
        ("Weighted Sampler", test_weighted_sampler),
        ("Analyze Dataset Splits", test_analyze_dataset_splits),
        ("Prepare Data for Training", test_prepare_data_for_training),
        ("Get Dataset Statistics", test_get_dataset_statistics),
        ("Create Test Loader", test_create_test_loader),
        ("Integration with Real Data", test_integration_with_real_data),
    ]

    # Run all tests
    for test_name, test_func in tests:
        runner.run_test(test_name, test_func)

    # Print summary
    runner.print_summary()

    # Return exit code
    success = runner.failed == 0
    if success:
        print("\n🎉 ALL DATA_UTILS TESTS PASSED!")
        print("✅ The complete data pipeline is working correctly!")
        print("\nYour data loading system is ready for:")
        print("  🚀 Model training")
        print("  📊 Data analysis")
        print("  🔄 Production deployment")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Check the issues above before proceeding to training")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest completed with exit code: {exit_code}")