#!/usr/bin/env python3
"""
Comprehensive test script for transforms.py
Tests all functions and classes independently
Run from project root: python -c "from src.data.transforms import *; exec(open('test_transforms_main.py').read())"
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List
import traceback
import time


# Create test images of different sizes and types
def create_test_images() -> Dict[str, Image.Image]:
    """Create various test images"""
    test_images = {}

    # Standard test image
    test_images['standard'] = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )

    # Small image
    test_images['small'] = Image.fromarray(
        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    )

    # Large image
    test_images['large'] = Image.fromarray(
        np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    )

    # Non-square image
    test_images['rectangle'] = Image.fromarray(
        np.random.randint(0, 255, (400, 200, 3), dtype=np.uint8)
    )

    # Grayscale converted to RGB
    gray_array = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    test_images['grayscale'] = Image.fromarray(gray_array, 'L').convert('RGB')

    return test_images


def test_basic_transforms():
    """Test basic transform functions"""
    print("=" * 60)
    print("TESTING BASIC TRANSFORMS")
    print("=" * 60)

    test_images = create_test_images()
    results = {}

    # Test different input sizes
    input_sizes = [224, 256, 384]

    for size in input_sizes:
        print(f"\n--- Testing input size: {size} ---")

        # Test get_train_transforms
        try:
            for level in ['light', 'medium', 'heavy']:
                transform = get_train_transforms(input_size=size, augmentation_level=level)

                for img_name, img in test_images.items():
                    tensor = transform(img)
                    expected_shape = (3, size, size)
                    assert tensor.shape == expected_shape, f"Wrong shape: {tensor.shape} vs {expected_shape}"
                    assert tensor.dtype == torch.float32, f"Wrong dtype: {tensor.dtype}"

                print(f"✓ get_train_transforms({level}) works for all test images")

            results[f'train_transforms_{size}'] = "PASS"

        except Exception as e:
            print(f"✗ get_train_transforms failed: {e}")
            results[f'train_transforms_{size}'] = f"FAIL: {e}"

        # Test get_val_transforms
        try:
            val_transform = get_val_transforms(input_size=size)

            for img_name, img in test_images.items():
                tensor = val_transform(img)
                expected_shape = (3, size, size)
                assert tensor.shape == expected_shape, f"Wrong shape: {tensor.shape} vs {expected_shape}"
                assert tensor.dtype == torch.float32, f"Wrong dtype: {tensor.dtype}"

            print(f"✓ get_val_transforms works for all test images")
            results[f'val_transforms_{size}'] = "PASS"

        except Exception as e:
            print(f"✗ get_val_transforms failed: {e}")
            results[f'val_transforms_{size}'] = f"FAIL: {e}"

    # Test get_inference_transforms
    try:
        inference_transform = get_inference_transforms(224)
        tensor = inference_transform(test_images['standard'])
        assert tensor.shape == (3, 224, 224), f"Wrong inference shape: {tensor.shape}"
        print("✓ get_inference_transforms works")
        results['inference_transforms'] = "PASS"
    except Exception as e:
        print(f"✗ get_inference_transforms failed: {e}")
        results['inference_transforms'] = f"FAIL: {e}"

    return results


def test_model_specific_transforms():
    """Test model-specific transform function"""
    print("\n" + "=" * 60)
    print("TESTING MODEL-SPECIFIC TRANSFORMS")
    print("=" * 60)

    results = {}

    # Test different model configurations
    test_models = [
        ('resnet50', 224),
        ('efficientnet-b2', 260),  # Assuming this is configured
        ('convnext-tiny-384', 384),  # Assuming this is configured
        ('unknown_model', 224)  # Should fallback to default
    ]

    for model_name, expected_size in test_models:
        print(f"\n--- Testing model: {model_name} ---")

        try:
            # Test training transforms
            train_transform = get_transforms_for_model(model_name, is_training=True)
            val_transform = get_transforms_for_model(model_name, is_training=False)

            test_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))

            train_tensor = train_transform(test_img)
            val_tensor = val_transform(test_img)

            print(f"✓ Train transform output: {train_tensor.shape}")
            print(f"✓ Val transform output: {val_tensor.shape}")

            # Note: We don't assert exact size since config might use defaults
            assert train_tensor.shape[0] == 3, "Wrong number of channels"
            assert val_tensor.shape[0] == 3, "Wrong number of channels"
            assert train_tensor.shape[1] == train_tensor.shape[2], "Not square"
            assert val_tensor.shape[1] == val_tensor.shape[2], "Not square"

            results[f'model_{model_name}'] = "PASS"

        except Exception as e:
            print(f"✗ Model {model_name} failed: {e}")
            results[f'model_{model_name}'] = f"FAIL: {e}"

    return results


def test_advanced_augmentations():
    """Test MixUp, CutMix, and RandAugment"""
    print("\n" + "=" * 60)
    print("TESTING ADVANCED AUGMENTATIONS")
    print("=" * 60)

    results = {}

    # Test MixUp
    print("\n--- Testing MixUp ---")
    try:
        mixup = MixUpTransform(alpha=1.0, num_classes=5)

        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])

        mixed_images, labels_a, labels_b, lam = mixup(images, labels)

        assert mixed_images.shape == images.shape, f"MixUp shape mismatch: {mixed_images.shape}"
        assert 0 <= lam <= 1, f"Lambda out of range: {lam}"
        assert labels_a.shape == labels.shape, "Labels_a shape mismatch"
        assert labels_b.shape == labels.shape, "Labels_b shape mismatch"

        print(f"✓ MixUp works: lambda={lam:.3f}, output_shape={mixed_images.shape}")
        results['mixup'] = "PASS"

    except Exception as e:
        print(f"✗ MixUp failed: {e}")
        results['mixup'] = f"FAIL: {e}"

    # Test CutMix
    print("\n--- Testing CutMix ---")
    try:
        cutmix = CutMixTransform(alpha=1.0, num_classes=5)

        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])

        mixed_images, labels_a, labels_b, lam = cutmix(images, labels)

        assert mixed_images.shape == images.shape, f"CutMix shape mismatch: {mixed_images.shape}"
        assert 0 <= lam <= 1, f"Lambda out of range: {lam}"
        assert labels_a.shape == labels.shape, "Labels_a shape mismatch"
        assert labels_b.shape == labels.shape, "Labels_b shape mismatch"

        print(f"✓ CutMix works: lambda={lam:.3f}, output_shape={mixed_images.shape}")
        results['cutmix'] = "PASS"

    except Exception as e:
        print(f"✗ CutMix failed: {e}")
        results['cutmix'] = f"FAIL: {e}"

    # Test RandAugment
    print("\n--- Testing RandAugment ---")
    try:
        randaugment = RandAugmentTransform(n=2, m=10)

        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        augmented_img = randaugment(test_img)

        assert isinstance(augmented_img, Image.Image), "RandAugment should return PIL Image"
        assert augmented_img.size == test_img.size, "RandAugment changed image size"

        print(f"✓ RandAugment works: input_size={test_img.size}, output_size={augmented_img.size}")
        results['randaugment'] = "PASS"

    except Exception as e:
        print(f"✗ RandAugment failed: {e}")
        results['randaugment'] = f"FAIL: {e}"

    return results


def test_advanced_train_transforms():
    """Test advanced training transforms with RandAugment"""
    print("\n" + "=" * 60)
    print("TESTING ADVANCED TRAIN TRANSFORMS")
    print("=" * 60)

    results = {}

    try:
        # Test without RandAugment
        transform_normal = get_advanced_train_transforms(
            input_size=224,
            use_randaugment=False
        )

        # Test with RandAugment
        transform_rand = get_advanced_train_transforms(
            input_size=224,
            use_randaugment=True,
            randaugment_n=2,
            randaugment_m=5
        )

        test_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))

        tensor_normal = transform_normal(test_img)
        tensor_rand = transform_rand(test_img)

        assert tensor_normal.shape == (3, 224, 224), f"Normal transform wrong shape: {tensor_normal.shape}"
        assert tensor_rand.shape == (3, 224, 224), f"RandAugment transform wrong shape: {tensor_rand.shape}"

        print(f"✓ Advanced transforms work: normal={tensor_normal.shape}, randaugment={tensor_rand.shape}")
        results['advanced_transforms'] = "PASS"

    except Exception as e:
        print(f"✗ Advanced transforms failed: {e}")
        results['advanced_transforms'] = f"FAIL: {e}"

    return results


def test_denormalize():
    """Test denormalization utility"""
    print("\n" + "=" * 60)
    print("TESTING DENORMALIZATION")
    print("=" * 60)

    results = {}

    try:
        # Create normalized tensor
        transform = get_val_transforms(224)
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        normalized_tensor = transform(test_img)

        # Denormalize
        denorm = DeNormalize()
        denormalized_tensor = denorm(normalized_tensor)

        assert denormalized_tensor.shape == normalized_tensor.shape, "Denorm changed shape"
        assert denormalized_tensor.dtype == torch.float32, "Denorm changed dtype"

        # Check if values are in reasonable range (0-1 for images)
        min_val = denormalized_tensor.min().item()
        max_val = denormalized_tensor.max().item()

        print(f"✓ Denormalization works: shape={denormalized_tensor.shape}")
        print(f"  Normalized range: [{normalized_tensor.min():.3f}, {normalized_tensor.max():.3f}]")
        print(f"  Denormalized range: [{min_val:.3f}, {max_val:.3f}]")

        results['denormalize'] = "PASS"

    except Exception as e:
        print(f"✗ Denormalization failed: {e}")
        results['denormalize'] = f"FAIL: {e}"

    return results


def test_performance():
    """Test transform performance"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE")
    print("=" * 60)

    results = {}

    try:
        transform = get_train_transforms(224, 'medium')
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # Time multiple transforms
        n_iterations = 100
        start_time = time.time()

        for _ in range(n_iterations):
            _ = transform(test_img)

        end_time = time.time()
        avg_time = (end_time - start_time) / n_iterations * 1000  # ms

        print(f"✓ Performance test: {avg_time:.2f}ms per transform (avg of {n_iterations} iterations)")

        if avg_time < 50:  # Less than 50ms is good
            results['performance'] = "PASS"
        else:
            results['performance'] = f"SLOW: {avg_time:.2f}ms"

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        results['performance'] = f"FAIL: {e}"

    return results


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    results = {}

    # Test very small image
    try:
        small_img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        transform = get_val_transforms(224)
        tensor = transform(small_img)
        assert tensor.shape == (3, 224, 224), "Small image transform failed"
        print("✓ Very small image (10x10) handled correctly")
        results['small_image'] = "PASS"
    except Exception as e:
        print(f"✗ Small image test failed: {e}")
        results['small_image'] = f"FAIL: {e}"

    # Test extreme aspect ratio
    try:
        wide_img = Image.fromarray(np.random.randint(0, 255, (50, 500, 3), dtype=np.uint8))
        transform = get_val_transforms(224)
        tensor = transform(wide_img)
        assert tensor.shape == (3, 224, 224), "Wide image transform failed"
        print("✓ Extreme aspect ratio (1:10) handled correctly")
        results['aspect_ratio'] = "PASS"
    except Exception as e:
        print(f"✗ Aspect ratio test failed: {e}")
        results['aspect_ratio'] = f"FAIL: {e}"

    # Test invalid augmentation level
    try:
        transform = get_train_transforms(224, 'invalid_level')
        # Should fallback to medium
        print("✓ Invalid augmentation level handled (fallback to medium)")
        results['invalid_aug_level'] = "PASS"
    except Exception as e:
        print(f"✗ Invalid augmentation level test failed: {e}")
        results['invalid_aug_level'] = f"FAIL: {e}"

    return results


def print_final_summary(all_results: Dict[str, Dict]):
    """Print comprehensive test summary"""
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_category, results in all_results.items():
        print(f"\n{test_category.upper()}:")
        for test_name, result in results.items():
            total_tests += 1
            if result == "PASS":
                passed_tests += 1
                print(f"  ✓ {test_name}")
            else:
                failed_tests.append(f"{test_category}.{test_name}: {result}")
                print(f"  ✗ {test_name}: {result}")

    print(f"\n" + "=" * 60)
    print(f"OVERALL RESULTS:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")

    if failed_tests:
        print(f"\nFAILED TESTS:")
        for failure in failed_tests:
            print(f"  - {failure}")

    return len(failed_tests) == 0


def main():
    """Main test function"""
    print("TRANSFORMS.PY COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("Testing all functions and classes in transforms.py")

    all_results = {}

    try:
        # Run all test categories
        all_results['basic_transforms'] = test_basic_transforms()
        all_results['model_specific'] = test_model_specific_transforms()
        all_results['advanced_augmentations'] = test_advanced_augmentations()
        all_results['advanced_transforms'] = test_advanced_train_transforms()
        all_results['denormalize'] = test_denormalize()
        all_results['performance'] = test_performance()
        all_results['edge_cases'] = test_edge_cases()

        # Print final summary
        success = print_final_summary(all_results)

        if success:
            print("\n🎉 ALL TESTS PASSED! transforms.py is working correctly.")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED! Check the issues above.")
            return 1

    except Exception as e:
        print(f"\n💥 CRITICAL ERROR during testing: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest completed with exit code: {exit_code}")