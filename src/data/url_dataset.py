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
        """Download image from URL with retries"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')

                # Validate image
                if image.size[0] < 10 or image.size[1] < 10:
                    raise ValueError(f"Image too small: {image.size}")

                self.stats['downloads'] += 1
                return image

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download {url} after {self.max_retries} attempts: {e}")
                    self.failed_downloads.add(url)
                    self.stats['failures'] += 1
                    return None
                time.sleep(1)  # Wait before retry
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
    # Test the dataset
    from torchvision import transforms

    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test URL dataset
    print("Testing URLDataset...")
    dataset = URLDataset(
        split_file='../data/splits/train.json',
        transform=transform,
        cache_dir='../data/cache/images'
    )

    # Create a small subset for testing
    subset = create_subset_dataset(dataset, fraction=0.01)

    # Try loading first few images
    for i in range(min(5, len(subset))):
        try:
            image, label, metadata = subset[i]
            print(f"Image {i}: shape={image.shape}, label={label} ({metadata['decade']}), name={metadata['name']}")
        except Exception as e:
            print(f"Error loading image {i}: {e}")

    # Print statistics
    print("\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")