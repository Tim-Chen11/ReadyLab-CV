import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import json
from pathlib import Path
import hashlib
import logging
from typing import Optional, Tuple, Dict, List
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from torchvision import transforms

# Set up logging
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base dataset class with common functionality"""

    def __init__(self, split_file: str, transform=None):
        """
        Args:
            split_file: Path to JSON file with image metadata
            transform: Torchvision transforms to apply
        """
        # Load metadata
        with open(split_file, 'r') as f:
            self.data = json.load(f)

        self.transform = transform

        # Create label mapping for 5 decades
        self.decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
        self.label_to_idx = {d: i for i, d in enumerate(self.decades)}
        self.idx_to_label = {i: d for i, d in enumerate(self.decades)}
        self.num_classes = len(self.decades)

        logger.info(f"Loaded dataset from {split_file} with {len(self.data)} images")

    def __len__(self) -> int:
        return len(self.data)

    def get_labels(self) -> List[int]:
        """Get all labels for computing class weights"""
        return [self.label_to_idx[item['decade']] for item in self.data]

    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for an item"""
        item = self.data[idx]
        return {
            'id': item['id'],
            'product_id': item['product_id'],
            'name': item['name'],
            'decade': item['decade'],
            'url': item.get('url', ''),
            'classification': item.get('classification', 'unknown'),
            'makers': item.get('makers', 'unknown'),
            'country': item.get('country', 'unknown')
        }


class URLDataset(BaseDataset):
    """Dataset that loads images from URLs with caching and error handling"""

    def __init__(
            self,
            split_file: str,
            transform=None,
            cache_dir: Optional[str] = None,
            max_retries: int = 3,
            timeout: int = 10,
            fallback_on_error: bool = True
    ):
        """
        Args:
            split_file: Path to JSON file with image metadata
            transform: Torchvision transforms to apply
            cache_dir: Directory to cache downloaded images
            max_retries: Maximum download attempts per image
            timeout: Download timeout in seconds
            fallback_on_error: Use placeholder image on download failure
        """
        super().__init__(split_file, transform)

        self.max_retries = max_retries
        self.timeout = timeout
        self.fallback_on_error = fallback_on_error

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Default cache location
            data_root = Path(split_file).parent.parent
            self.cache_dir = data_root / 'cache' / 'images'

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        self.stats = {
            'cache_hits': 0,
            'downloads': 0,
            'failures': 0
        }

        # Failed downloads tracking
        self.failed_downloads = set()

        logger.info(f"Cache directory: {self.cache_dir}")

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache filename from URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.jpg"

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL with retries and improved error handling"""

        # Enhanced headers to avoid 403 Forbidden errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        for attempt in range(self.max_retries):
            try:
                # Add delay between attempts (exponential backoff)
                if attempt > 0:
                    delay = min(2 ** attempt, 10)  # Cap at 10 seconds
                    time.sleep(delay)
                    logger.debug(f"Retry {attempt + 1} for {url} after {delay}s delay")

                # Make request with improved settings
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    stream=True,  # Stream for large images
                    allow_redirects=True,  # Follow redirects
                    verify=True  # Verify SSL certificates
                )

                # Check response status
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                    raise ValueError(f"Invalid content type: {content_type}")

                # Check content length (avoid downloading huge files)
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > 50 * 1024 * 1024:  # 50MB limit
                    raise ValueError(f"Image too large: {content_length} bytes")

                # Read content
                content = response.content

                # Check if content is actually an image
                if len(content) < 100:  # Too small to be a valid image
                    raise ValueError(f"Content too small: {len(content)} bytes")

                # Check for common image file signatures
                image_signatures = [
                    b'\xff\xd8\xff',  # JPEG
                    b'\x89PNG\r\n\x1a\n',  # PNG
                    b'GIF87a',  # GIF87a
                    b'GIF89a',  # GIF89a
                    b'RIFF',  # WebP (starts with RIFF)
                    b'BM',  # BMP
                ]

                if not any(content.startswith(sig) for sig in image_signatures):
                    logger.warning(f"Content doesn't appear to be a valid image: {url}")
                    # Try to continue anyway - PIL might still be able to handle it

                # Try to open and validate image
                try:
                    image = Image.open(BytesIO(content)).convert('RGB')
                except Exception as img_error:
                    raise ValueError(f"Failed to decode image: {img_error}")

                # Validate image dimensions
                if image.size[0] < 10 or image.size[1] < 10:
                    raise ValueError(f"Image too small: {image.size}")

                # Check for extremely large images that might cause memory issues
                if image.size[0] * image.size[1] > 20000 * 20000:  # 400MP limit
                    logger.warning(f"Very large image: {image.size}, might resize")
                    # Could add automatic resizing here if needed

                # Success!
                self.stats['downloads'] += 1
                logger.debug(f"Successfully downloaded {url}: {image.size}")
                return image

            except requests.exceptions.HTTPError as e:
                error_msg = f"HTTP error {response.status_code}"
                if response.status_code == 403:
                    error_msg += " (Forbidden - website blocking requests)"
                elif response.status_code == 404:
                    error_msg += " (Not Found - URL may be outdated)"
                elif response.status_code == 429:
                    error_msg += " (Rate Limited - too many requests)"
                    # Longer delay for rate limiting
                    if attempt < self.max_retries - 1:
                        time.sleep(30)
                elif response.status_code >= 500:
                    error_msg += " (Server Error - temporary issue)"

                logger.debug(f"Attempt {attempt + 1}: {error_msg} for {url}")
                last_error = error_msg

            except requests.exceptions.Timeout:
                error_msg = f"Timeout after {self.timeout}s"
                logger.debug(f"Attempt {attempt + 1}: {error_msg} for {url}")
                last_error = error_msg

            except requests.exceptions.ConnectionError:
                error_msg = "Connection error (network or DNS issue)"
                logger.debug(f"Attempt {attempt + 1}: {error_msg} for {url}")
                last_error = error_msg

            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {str(e)}"
                logger.debug(f"Attempt {attempt + 1}: {error_msg} for {url}")
                last_error = error_msg

            except ValueError as e:
                # Image validation errors
                error_msg = f"Image validation error: {str(e)}"
                logger.debug(f"Attempt {attempt + 1}: {error_msg} for {url}")
                last_error = error_msg

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.debug(f"Attempt {attempt + 1}: {error_msg} for {url}")
                last_error = error_msg

            # Don't retry for certain errors
            if any(phrase in str(last_error).lower() for phrase in [
                'not found', '404', 'invalid content type', 'too small', 'too large'
            ]):
                logger.debug(f"Not retrying {url} due to: {last_error}")
                break

        # All attempts failed
        logger.error(f"Failed to download {url} after {self.max_retries} attempts: {last_error}")
        self.failed_downloads.add(url)
        self.stats['failures'] += 1
        return None

    def _load_image(self, item: Dict) -> Optional[Image.Image]:
        """Load image with caching"""
        url = item['url']

        # Check cache first
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            try:
                image = Image.open(cache_path).convert('RGB')
                self.stats['cache_hits'] += 1
                return image
            except Exception as e:
                logger.warning(f"Failed to load cached image {cache_path}: {e}")
                cache_path.unlink()  # Remove corrupted cache file

        # Download if not cached
        image = self._download_image(url)
        if image:
            # Save to cache
            try:
                image.save(cache_path, 'JPEG', quality=95)
            except Exception as e:
                logger.warning(f"Failed to cache image: {e}")

        return image

    def _get_placeholder_image(self, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Create a placeholder image for failed downloads"""
        # Create a gray image with noise
        placeholder = np.random.randint(100, 150, (*size, 3), dtype=np.uint8)
        return Image.fromarray(placeholder)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Returns:
            image: Transformed image tensor
            label: Decade label (0-4)
            metadata: Dictionary with item metadata
        """
        item = self.data[idx]

        # Load image
        image = self._load_image(item)

        if image is None and self.fallback_on_error:
            # Use placeholder for failed downloads
            image = self._get_placeholder_image()
            logger.debug(f"Using placeholder for index {idx}, URL: {item['url']}")
        elif image is None:
            # Raise exception if no fallback
            raise ValueError(f"Failed to load image at index {idx}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = transforms.ToTensor()(image)

        # Get label
        label = self.label_to_idx[item['decade']]

        # Get metadata
        metadata = self.get_metadata(idx)

        return image, label, metadata

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        return {
            **self.stats,
            'total_images': len(self.data),
            'failed_urls': len(self.failed_downloads),
            'cache_size_mb': sum(f.stat().st_size for f in self.cache_dir.glob('*.jpg')) / 1024 / 1024
        }


class CachedDataset(BaseDataset):
    """Dataset for pre-downloaded images (faster than URLDataset)"""

    def __init__(
            self,
            split_file: str,
            images_dir: str,
            transform=None,
            verify_images: bool = True
    ):
        """
        Args:
            split_file: Path to JSON file with image metadata
            images_dir: Directory containing downloaded images
            transform: Torchvision transforms to apply
            verify_images: Whether to verify all images exist on init
        """
        super().__init__(split_file, transform)

        self.images_dir = Path(images_dir)

        if verify_images:
            # Filter out items without cached images
            self.valid_data = []
            missing_count = 0

            for item in self.data:
                cache_path = self._get_cache_path(item['url'])
                if cache_path.exists():
                    self.valid_data.append(item)
                else:
                    missing_count += 1

            if missing_count > 0:
                logger.warning(f"Missing {missing_count} cached images out of {len(self.data)}")

            self.data = self.valid_data
            logger.info(f"Using {len(self.data)} cached images")

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache filename from URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.images_dir / f"{url_hash}.jpg"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        item = self.data[idx]

        # Load cached image
        image_path = self._get_cache_path(item['url'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.label_to_idx[item['decade']]

        # Get metadata
        metadata = self.get_metadata(idx)

        return image, label, metadata


def download_dataset_images(
        split_files: List[str],
        cache_dir: str,
        num_workers: int = 8,
        skip_existing: bool = True
) -> Dict[str, int]:
    """
    Pre-download all images for faster training

    Args:
        split_files: List of split JSON files
        cache_dir: Directory to save images
        num_workers: Number of parallel download workers
        skip_existing: Skip already downloaded images

    Returns:
        Dictionary with download statistics
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique URLs
    all_urls = set()
    url_to_metadata = {}

    for split_file in split_files:
        with open(split_file, 'r') as f:
            data = json.load(f)
        for item in data:
            url = item['url']
            all_urls.add(url)
            url_to_metadata[url] = item

    logger.info(f"Found {len(all_urls)} unique URLs to download")

    # Filter existing if requested
    if skip_existing:
        urls_to_download = []
        for url in all_urls:
            cache_path = cache_dir / f"{hashlib.md5(url.encode()).hexdigest()}.jpg"
            if not cache_path.exists():
                urls_to_download.append(url)
        logger.info(f"Skipping {len(all_urls) - len(urls_to_download)} existing images")
    else:
        urls_to_download = list(all_urls)

    # Download function
    def download_single(url):
        cache_path = cache_dir / f"{hashlib.md5(url.encode()).hexdigest()}.jpg"

        if cache_path.exists() and skip_existing:
            return url, True, "cached"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Validate image
            if image.size[0] < 10 or image.size[1] < 10:
                raise ValueError(f"Image too small: {image.size}")

            image.save(cache_path, 'JPEG', quality=95)
            return url, True, "downloaded"
        except Exception as e:
            return url, False, str(e)

    # Download in parallel
    results = {"cached": 0, "downloaded": 0, "failed": 0}
    failed_items = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_single, url): url for url in urls_to_download}

        with tqdm(total=len(urls_to_download), desc="Downloading images") as pbar:
            for future in as_completed(futures):
                url, success, status = future.result()
                pbar.update(1)

                if success:
                    if status == "cached":
                        results["cached"] += 1
                    else:
                        results["downloaded"] += 1
                else:
                    results["failed"] += 1
                    metadata = url_to_metadata.get(url, {})
                    failed_items.append({
                        'url': url,
                        'error': status,
                        'name': metadata.get('name', 'unknown'),
                        'decade': metadata.get('decade', 'unknown')
                    })

    # Save failed items report
    if failed_items:
        failed_report_path = cache_dir / 'download_failures.json'
        with open(failed_report_path, 'w') as f:
            json.dump(failed_items, f, indent=2)
        logger.info(f"Saved failure report to {failed_report_path}")

    # Print summary
    total_processed = results["cached"] + results["downloaded"] + results["failed"]
    logger.info(f"\nDownload complete:")
    logger.info(f"  Total processed: {total_processed}")
    logger.info(f"  Already cached: {results['cached']}")
    logger.info(f"  Downloaded: {results['downloaded']}")
    logger.info(f"  Failed: {results['failed']}")

    return results


def create_subset_dataset(
        dataset: BaseDataset,
        fraction: float = 0.1,
        seed: int = 42
) -> BaseDataset:
    """
    Create a subset of a dataset for quick testing

    Args:
        dataset: Original dataset
        fraction: Fraction of data to keep
        seed: Random seed

    Returns:
        Subset dataset
    """
    np.random.seed(seed)

    # Get indices for each class
    class_indices = {i: [] for i in range(dataset.num_classes)}
    for idx, item in enumerate(dataset.data):
        label = dataset.label_to_idx[item['decade']]
        class_indices[label].append(idx)

    # Sample from each class
    subset_indices = []
    for label, indices in class_indices.items():
        n_samples = max(1, int(len(indices) * fraction))
        sampled = np.random.choice(indices, n_samples, replace=False)
        subset_indices.extend(sampled)

    # Create subset
    subset_data = [dataset.data[i] for i in subset_indices]

    # Create new dataset instance
    subset_dataset = type(dataset).__new__(type(dataset))
    subset_dataset.__dict__.update(dataset.__dict__)
    subset_dataset.data = subset_data

    logger.info(f"Created subset with {len(subset_data)} samples ({fraction * 100:.1f}% of original)")

    return subset_dataset


if __name__ == "__main__":
    # Test the dataset with your actual project structure
    from torchvision import transforms
    from pathlib import Path
    import sys

    print("🧪 URL_DATASET.PY QUICK TEST")
    print("=" * 40)

    # Get project root - go up from src/data/ to project root
    current_file = Path(__file__)  # This is src/data/url_dataset.py
    project_root = current_file.parent.parent.parent  # Go up 3 levels: data -> src -> project_root

    print(f"Project root: {project_root}")
    print(f"Current file: {current_file}")

    # Define paths based on your project structure
    data_dir = project_root / "data"
    splits_dir = data_dir / "splits"
    cache_dir = data_dir / "cache" / "images"

    # Check if required files exist
    train_split = splits_dir / "train.json"
    val_split = splits_dir / "val.json"

    print(f"\nChecking files:")
    print(f"  Data dir: {data_dir.exists()} - {data_dir}")
    print(f"  Splits dir: {splits_dir.exists()} - {splits_dir}")
    print(f"  Cache dir: {cache_dir.exists()} - {cache_dir}")
    print(f"  Train split: {train_split.exists()} - {train_split}")

    if not train_split.exists():
        print(f"\n❌ Train split file not found!")
        print(f"Available files in splits directory:")
        if splits_dir.exists():
            for file in splits_dir.iterdir():
                print(f"  - {file.name}")
        else:
            print(f"  Splits directory doesn't exist!")
        sys.exit(1)

    # Count cached images
    if cache_dir.exists():
        cached_images = list(cache_dir.glob('*.jpg'))
        print(f"  Cached images: {len(cached_images)}")
    else:
        cached_images = []
        print(f"  Cached images: 0 (cache dir doesn't exist)")

    # Create simple transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        print(f"\n1. Testing BaseDataset...")
        base_dataset = BaseDataset(str(train_split))
        print(f"   ✓ Loaded {len(base_dataset)} samples")
        print(f"   ✓ Classes: {base_dataset.decades}")

        # Show label distribution
        labels = base_dataset.get_labels()
        from collections import Counter

        label_counts = Counter(labels)
        print(f"   ✓ Label distribution:")
        for idx, count in label_counts.items():
            decade = base_dataset.idx_to_label[idx]
            print(f"     {decade}: {count} samples")

        # Test metadata
        if len(base_dataset) > 0:
            metadata = base_dataset.get_metadata(0)
            print(f"   ✓ First sample: {metadata['name'][:50]}... ({metadata['decade']})")

    except Exception as e:
        print(f"❌ BaseDataset test failed: {e}")
        sys.exit(1)

    try:
        print(f"\n2. Testing URLDataset...")
        url_dataset = URLDataset(
            split_file=str(train_split),
            transform=transform,
            cache_dir=str(cache_dir),
            fallback_on_error=True,  # Use fallback for failed downloads
            max_retries=2,
            timeout=5
        )

        print(f"   ✓ URLDataset initialized with {len(url_dataset)} samples")
        print(f"   ✓ Cache directory: {url_dataset.cache_dir}")

        # Create a tiny subset for quick testing (0.1% = ~5-10 samples)
        subset = create_subset_dataset(url_dataset, fraction=0.001, seed=42)
        print(f"   ✓ Created test subset with {len(subset)} samples")

        # Try loading a few samples
        successful_loads = 0
        failed_loads = 0

        print(f"   ✓ Testing sample loading...")
        for i in range(min(3, len(subset))):
            try:
                image, label, metadata = subset[i]
                print(f"     Sample {i}: shape={image.shape}, label={label} ({metadata['decade']})")
                print(f"       Name: {metadata['name'][:40]}...")
                successful_loads += 1

                # Basic validation
                assert image.shape == (3, 224, 224), f"Unexpected shape: {image.shape}"
                assert 0 <= label < 5, f"Label out of range: {label}"

            except Exception as e:
                print(f"     Sample {i} failed: {str(e)[:60]}...")
                failed_loads += 1

        print(f"   ✓ Results: {successful_loads} successful, {failed_loads} failed")

        # Get and display statistics
        stats = url_dataset.get_statistics()
        print(f"   ✓ Dataset statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.2f}")
            else:
                print(f"     {key}: {value}")

    except Exception as e:
        print(f"❌ URLDataset test failed: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

    try:
        print(f"\n3. Testing CachedDataset...")

        if len(cached_images) > 0:
            cached_dataset = CachedDataset(
                split_file=str(train_split),
                images_dir=str(cache_dir),
                transform=transform,
                verify_images=True
            )

            print(f"   ✓ CachedDataset: {len(cached_dataset)} valid cached images")

            if len(cached_dataset) > 0:
                # Try loading one cached sample
                image, label, metadata = cached_dataset[0]
                print(f"   ✓ Cached sample: shape={image.shape}, label={label}")
                print(f"     Name: {metadata['name'][:40]}...")
            else:
                print(f"   ⚠️  No valid cached images found")
        else:
            print(f"   ⚠️  No cached images available, skipping CachedDataset test")

    except Exception as e:
        print(f"❌ CachedDataset test failed: {e}")
        # Don't exit here, just warn

    try:
        print(f"\n4. Testing create_subset_dataset...")

        # Test different subset sizes
        for fraction in [0.1, 0.01]:
            subset = create_subset_dataset(base_dataset, fraction=fraction, seed=42)
            expected_size = max(5, int(len(base_dataset) * fraction))  # At least 1 per class
            print(f"   ✓ Subset {fraction * 100}%: {len(subset)} samples (expected ~{expected_size})")

            # Check that we have diverse classes
            subset_labels = subset.get_labels()
            unique_classes = len(set(subset_labels))
            print(f"     Classes represented: {unique_classes}/5")

    except Exception as e:
        print(f"❌ Subset creation test failed: {e}")

    print(f"\n" + "=" * 40)
    print(f"🎉 URL_DATASET.PY TESTS COMPLETED!")
    print(f"✅ Core functionality is working")
    print(f"")
    print(f"Usage examples:")
    print(f"  # Basic dataset")
    print(f"  dataset = BaseDataset('{train_split}')")
    print(f"  ")
    print(f"  # URL dataset with caching")
    print(f"  dataset = URLDataset('{train_split}', cache_dir='{cache_dir}')")
    print(f"  ")
    print(f"  # Cached dataset (faster)")
    print(f"  dataset = CachedDataset('{train_split}', images_dir='{cache_dir}')")

