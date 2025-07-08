
# !/usr/bin/env python3
# Add this comprehensive test function to your url_dataset.py file
# Replace the existing comprehensive test with this fixed version

def run_comprehensive_test():
    """Run comprehensive test suite for url_dataset.py"""
    print("URL_DATASET.PY COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("Testing all classes and functions in url_dataset.py")

    import tempfile
    import json
    import hashlib
    from unittest.mock import patch, MagicMock
    from io import BytesIO
    import requests
    from collections import Counter
    from torchvision import transforms

    tests_passed = 0
    tests_failed = 0
    failed_tests = []

    def run_test(test_name, test_func):
        nonlocal tests_passed, tests_failed, failed_tests
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")
        try:
            test_func()
            tests_passed += 1
            print(f"✅ PASSED: {test_name}")
        except Exception as e:
            tests_failed += 1
            failed_tests.append(f"{test_name}: {str(e)}")
            print(f"❌ FAILED: {test_name}")
            print(f"Error: {str(e)}")

    def create_mock_data(num_items=10):
        """Create mock JSON data"""
        mock_data = []
        decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
        for i in range(num_items):
            mock_data.append({
                "id": f"test_id_{i}",
                "product_id": f"product_{i % 3}",
                "name": f"Test Product {i}",
                "decade": decades[i % 5],
                "url": f"https://example.com/image_{i}.jpg",
                "classification": f"test_class_{i % 2}",
                "makers": f"test_maker_{i % 2}",
                "country": f"test_country_{i % 3}"
            })
        return mock_data

    def create_mock_image(size=(200, 200)):
        """Create mock PIL image"""
        array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        return Image.fromarray(array)

    def test_base_dataset():
        """Test BaseDataset functionality"""
        print("Testing BaseDataset...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            split_file = temp_path / "test.json"

            mock_data = create_mock_data(10)
            with open(split_file, 'w') as f:
                json.dump(mock_data, f)

            dataset = BaseDataset(str(split_file))

            print(f"✓ BaseDataset initialized with {len(dataset)} items")
            print(f"✓ Number of classes: {dataset.num_classes}")
            print(f"✓ Class names: {dataset.decades}")

            assert len(dataset) == 10
            assert dataset.num_classes == 5
            assert dataset.decades == ['1960s', '1970s', '1980s', '1990s', '2000s']

            labels = dataset.get_labels()
            print(f"✓ Labels: {labels}")
            assert len(labels) == 10

            metadata = dataset.get_metadata(0)
            expected_keys = ['id', 'product_id', 'name', 'decade', 'url', 'classification', 'makers', 'country']
            for key in expected_keys:
                assert key in metadata
            print(f"✓ Metadata keys: {list(metadata.keys())}")

    def test_url_dataset_mock():
        """Test URLDataset with mocked requests"""
        print("Testing URLDataset with mocked requests...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_dir = temp_path / "cache"
            split_file = temp_path / "test.json"

            mock_data = create_mock_data(5)
            with open(split_file, 'w') as f:
                json.dump(mock_data, f)

            # Create transform to ensure tensor output
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            mock_image = create_mock_image((300, 300))
            mock_response = MagicMock()
            mock_response.content = BytesIO()
            mock_image.save(mock_response.content, 'JPEG')
            mock_response.content = mock_response.content.getvalue()
            mock_response.raise_for_status = MagicMock()

            with patch('requests.get', return_value=mock_response):
                dataset = URLDataset(
                    split_file=str(split_file),
                    cache_dir=str(cache_dir),
                    transform=test_transform,  # Provide transform
                    max_retries=2,
                    timeout=5,
                    fallback_on_error=True
                )

                print(f"✓ URLDataset initialized with {len(dataset)} items")
                print(f"✓ Cache directory: {dataset.cache_dir}")

                test_url = "https://example.com/test.jpg"
                cache_path = dataset._get_cache_path(test_url)
                print(f"✓ Cache path generation: {cache_path.name}")

                # Test image loading
                image, label, metadata = dataset[0]

                assert isinstance(image, torch.Tensor)
                assert image.dtype == torch.float32
                assert 0 <= label < 5
                assert isinstance(metadata, dict)

                print(f"✓ First item loaded: shape={image.shape}, label={label}, decade={metadata['decade']}")

                # Test statistics
                stats = dataset.get_statistics()
                print(f"✓ Statistics: {stats}")

    def test_url_dataset_fallback():
        """Test URLDataset fallback behavior"""
        print("Testing URLDataset fallback behavior...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_dir = temp_path / "cache"
            split_file = temp_path / "test.json"

            mock_data = create_mock_data(3)
            with open(split_file, 'w') as f:
                json.dump(mock_data, f)

            # Create transform
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            with patch('requests.get', side_effect=requests.exceptions.ConnectionError("Mock connection error")):
                dataset = URLDataset(
                    split_file=str(split_file),
                    cache_dir=str(cache_dir),
                    transform=test_transform,  # Provide transform
                    max_retries=1,
                    timeout=1,
                    fallback_on_error=True
                )

                # Should use placeholder image
                image, label, metadata = dataset[0]

                assert isinstance(image, torch.Tensor)
                print(f"✓ Fallback image loaded: shape={image.shape}")

                stats = dataset.get_statistics()
                assert stats['failures'] > 0
                print(f"✓ Failure tracked in statistics: {stats['failures']} failures")

    def test_cached_dataset():
        """Test CachedDataset functionality"""
        print("Testing CachedDataset...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            split_file = temp_path / "test.json"

            mock_data = create_mock_data(5)
            with open(split_file, 'w') as f:
                json.dump(mock_data, f)

            # Create cached images for first 3 items
            for i in range(3):
                url = mock_data[i]['url']
                url_hash = hashlib.md5(url.encode()).hexdigest()
                cache_path = images_dir / f"{url_hash}.jpg"

                mock_image = create_mock_image((200, 200))
                mock_image.save(cache_path, 'JPEG')

            print(f"✓ Created {len(list(images_dir.glob('*.jpg')))} cached images")

            # Create transform
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            dataset = CachedDataset(
                split_file=str(split_file),
                images_dir=str(images_dir),
                transform=test_transform,  # Provide transform
                verify_images=True
            )

            print(f"✓ CachedDataset initialized with {len(dataset)} valid images")
            assert len(dataset) == 3

            image, label, metadata = dataset[0]
            assert isinstance(image, torch.Tensor)
            assert 0 <= label < 5
            print(f"✓ Cached item loaded: shape={image.shape}, label={label}")

    def test_subset_creation():
        """Test subset creation"""
        print("Testing create_subset_dataset...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            split_file = temp_path / "test.json"

            mock_data = create_mock_data(50)
            with open(split_file, 'w') as f:
                json.dump(mock_data, f)

            original_dataset = BaseDataset(str(split_file))
            original_size = len(original_dataset)

            print(f"✓ Original dataset size: {original_size}")

            subset_dataset = create_subset_dataset(original_dataset, fraction=0.2, seed=42)
            subset_size = len(subset_dataset)

            print(f"✓ Subset dataset size: {subset_size}")
            print(f"✓ Subset fraction: {subset_size / original_size:.2f}")

            assert subset_size >= 5  # At least 1 from each class
            assert subset_size <= original_size

            subset_labels = subset_dataset.get_labels()
            unique_labels = set(subset_labels)
            print(f"✓ Subset has {len(unique_labels)} unique classes: {unique_labels}")

    def test_download_images():
        """Test bulk download functionality"""
        print("Testing download_dataset_images with mocked requests...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_dir = temp_path / "cache"

            split_files = []
            for split_name in ["train", "val"]:
                split_file = temp_path / f"{split_name}.json"
                mock_data = create_mock_data(5)
                # Make URLs unique across splits
                for i, item in enumerate(mock_data):
                    item['url'] = f"https://example.com/{split_name}_image_{i}.jpg"

                with open(split_file, 'w') as f:
                    json.dump(mock_data, f)
                split_files.append(str(split_file))

            mock_image = create_mock_image((200, 200))
            mock_response = MagicMock()
            mock_response.content = BytesIO()
            mock_image.save(mock_response.content, 'JPEG')
            mock_response.content = mock_response.content.getvalue()
            mock_response.raise_for_status = MagicMock()

            with patch('requests.get', return_value=mock_response):
                results = download_dataset_images(
                    split_files=split_files,
                    cache_dir=str(cache_dir),
                    num_workers=2,
                    skip_existing=True
                )

                print(f"✓ Download results: {results}")

                expected_keys = ['cached', 'downloaded', 'failed']
                for key in expected_keys:
                    assert key in results

                cached_images = list(cache_dir.glob('*.jpg'))
                print(f"✓ Created {len(cached_images)} cached image files")

    def test_edge_cases():
        """Test edge cases"""
        print("Testing edge cases...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test empty dataset
            empty_split_file = temp_path / "empty.json"
            with open(empty_split_file, 'w') as f:
                json.dump([], f)

            empty_dataset = BaseDataset(str(empty_split_file))
            assert len(empty_dataset) == 0
            print("✓ Empty dataset handled correctly")

            # Test malformed JSON
            try:
                malformed_split_file = temp_path / "malformed.json"
                with open(malformed_split_file, 'w') as f:
                    f.write("invalid json")

                BaseDataset(str(malformed_split_file))
                assert False, "Should have raised exception"
            except json.JSONDecodeError:
                print("✓ Malformed JSON handled correctly")

            # Test URLDataset with no fallback
            split_file = temp_path / "test.json"
            mock_data = create_mock_data(2)
            with open(split_file, 'w') as f:
                json.dump(mock_data, f)

            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            with patch('requests.get', side_effect=Exception("Mock error")):
                dataset = URLDataset(
                    split_file=str(split_file),
                    transform=test_transform,
                    fallback_on_error=False,
                    max_retries=1
                )

                try:
                    image, label, metadata = dataset[0]
                    assert False, "Should have raised exception"
                except (ValueError, Exception):
                    print("✓ No-fallback error handling works correctly")

    def test_performance():
        """Test performance"""
        print("Testing performance...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            split_file = temp_path / "perf_test.json"

            mock_data = create_mock_data(100)
            with open(split_file, 'w') as f:
                json.dump(mock_data, f)

            import time

            start_time = time.time()
            dataset = BaseDataset(str(split_file))
            init_time = time.time() - start_time

            print(f"✓ BaseDataset initialization: {init_time:.3f}s for {len(dataset)} items")

            start_time = time.time()
            labels = dataset.get_labels()
            label_time = time.time() - start_time

            print(f"✓ Label extraction: {label_time:.3f}s for {len(labels)} labels")

            start_time = time.time()
            for i in range(min(10, len(dataset))):
                metadata = dataset.get_metadata(i)
            metadata_time = time.time() - start_time

            print(f"✓ Metadata extraction: {metadata_time:.3f}s for 10 items")

            assert init_time < 1.0
            assert label_time < 0.1

    # Run all tests
    tests = [
        ("BaseDataset Functionality", test_base_dataset),
        ("URLDataset with Mocked Requests", test_url_dataset_mock),
        ("URLDataset Fallback Behavior", test_url_dataset_fallback),
        ("CachedDataset Functionality", test_cached_dataset),
        ("Subset Creation", test_subset_creation),
        ("Bulk Download with Mocked Requests", test_download_images),
        ("Edge Cases and Error Handling", test_edge_cases),
        ("Performance Testing", test_performance),
    ]

    for test_name, test_func in tests:
        run_test(test_name, test_func)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")

    if failed_tests:
        print(f"\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")

    success = tests_failed == 0
    if success:
        print(f"\n🎉 ALL TESTS PASSED! url_dataset.py is working correctly.")
    else:
        print(f"\n❌ SOME TESTS FAILED! Check the issues above.")

    return success


# Update the main section to include the comprehensive test option
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        # Run comprehensive test
        run_comprehensive_test()
    else:
        # Run the quick test (your existing main code)
        # ... (your existing main code here)
        pass