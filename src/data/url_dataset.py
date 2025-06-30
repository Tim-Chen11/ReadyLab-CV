import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import json
from pathlib import Path
import hashlib
import logging
from typing import Optional, Tuple, Dict
import time
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URLDataset(Dataset):
    """Dataset that loads images from URLs with caching and error handling"""

    def __init__(
            self,
            split_file: str,
            transform=None,
            cache_dir: Optional[str] = None,
            max_retries: int = 3,
            timeout: int = 10
    ):
        """
        Args:
            split_file: Path to JSON file with image metadata
            transform: Torchvision transforms to apply
            cache_dir: Directory to cache downloaded images
            max_retries: Maximum download attempts per image
            timeout: Download timeout in seconds
        """
        # Load metadata
        with open(split_file, 'r') as f:
            self.data = json.load(f)

        self.transform = transform
        self.max_retries = max_retries
        self.timeout = timeout

        # Set up cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path('../data/cache/images')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create label mapping
        self.decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
        self.label_to_idx = {d: i for i, d in enumerate(self.decades)}
        self.idx_to_label = {i: d for i, d in enumerate(self.decades)}

        # Track failed downloads
        self.failed_downloads = set()

        logger.info(f"Loaded dataset with {len(self.data)} images")
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
                return image
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download {url} after {self.max_retries} attempts: {e}")
                    self.failed_downloads.add(url)
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
                return Image.open(cache_path).convert('RGB')
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

    def _get_placeholder_image(self) -> torch.Tensor:
        """Create a placeholder image for failed downloads"""
        if self.transform:
            # Create a gray placeholder image
            placeholder = Image.new('RGB', (224, 224), (128, 128, 128))
            return self.transform(placeholder)
        else:
            return torch.ones(3, 224, 224) * 0.5

    def __len__(self) -> int:
        return len(self.data)

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

        if image is None:
            # Use placeholder for failed downloads
            image_tensor = self._get_placeholder_image()
            logger.warning(f"Using placeholder for index {idx}, URL: {item['url']}")
        else:
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Default transform if none provided
                image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Get label
        label = self.label_to_idx[item['decade']]

        # Return image, label, and metadata
        return image_tensor, label, {
            'id': item['id'],
            'product_id': item['product_id'],
            'name': item['name'],
            'decade': item['decade'],
            'url': item['url']
        }

    def get_failed_downloads(self) -> set:
        """Return set of URLs that failed to download"""
        return self.failed_downloads


class CachedDataset(Dataset):
    """Dataset for pre-downloaded images (faster than URLDataset)"""

    def __init__(self, split_file: str, images_dir: str, transform=None):
        """
        Args:
            split_file: Path to JSON file with image metadata
            images_dir: Directory containing downloaded images
            transform: Torchvision transforms to apply
        """
        with open(split_file, 'r') as f:
            self.data = json.load(f)

        self.images_dir = Path(images_dir)
        self.transform = transform

        # Create label mapping
        self.decades = ['1960s', '1970s', '1980s', '1990s', '2000s']
        self.label_to_idx = {d: i for i, d in enumerate(self.decades)}

        # Filter out items without cached images
        self.valid_data = []
        for item in self.data:
            cache_path = self._get_cache_path(item['url'])
            if cache_path.exists():
                self.valid_data.append(item)

        logger.info(f"Found {len(self.valid_data)}/{len(self.data)} cached images")
        self.data = self.valid_data

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache filename from URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.images_dir / f"{url_hash}.jpg"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        item = self.data[idx]

        # Load cached image
        image_path = self._get_cache_path(item['url'])
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.label_to_idx[item['decade']]

        return image, label, {
            'id': item['id'],
            'name': item['name'],
            'decade': item['decade']
        }


# Utility function for pre-downloading all images
def download_all_images(split_files: list, cache_dir: str, num_workers: int = 4):
    """Pre-download all images for faster training"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique URLs
    all_urls = set()
    for split_file in split_files:
        with open(split_file, 'r') as f:
            data = json.load(f)
        for item in data:
            all_urls.add(item['url'])

    print(f"Found {len(all_urls)} unique URLs to download")

    def download_single(url):
        cache_path = cache_dir / f"{hashlib.md5(url.encode()).hexdigest()}.jpg"
        if cache_path.exists():
            return url, True, "cached"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image.save(cache_path, 'JPEG', quality=95)
            return url, True, "downloaded"
        except Exception as e:
            return url, False, str(e)

    # Download in parallel
    results = {"cached": 0, "downloaded": 0, "failed": 0}
    failed_urls = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_single, url): url for url in all_urls}

        for future in tqdm(as_completed(futures), total=len(all_urls), desc="Downloading images"):
            url, success, status = future.result()
            if success:
                if status == "cached":
                    results["cached"] += 1
                else:
                    results["downloaded"] += 1
            else:
                results["failed"] += 1
                failed_urls.append((url, status))

    print(f"\nDownload complete:")
    print(f"  Cached: {results['cached']}")
    print(f"  Downloaded: {results['downloaded']}")
    print(f"  Failed: {results['failed']}")

    if failed_urls:
        print(f"\nFailed URLs:")
        for url, error in failed_urls[:10]:  # Show first 10
            print(f"  {url}: {error}")
        if len(failed_urls) > 10:
            print(f"  ... and {len(failed_urls) - 10} more")


if __name__ == "__main__":
    # Test the dataset
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test loading
    dataset = URLDataset(
        split_file='../data/splits/train.json',
        transform=transform,
        cache_dir='../data/cache/images'
    )

    # Try loading first few images
    for i in range(min(5, len(dataset))):
        image, label, metadata = dataset[i]
        print(f"Image {i}: shape={image.shape}, label={label} ({metadata['decade']}), name={metadata['name']}")