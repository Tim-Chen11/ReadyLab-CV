import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
import json
from pathlib import Path
import hashlib
import logging
from typing import Optional, Tuple, Dict
import time
import random
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
            timeout: int = 30
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

        # Set up session with browser-like headers and retry strategy
        self._setup_session()

        logger.info(f"Loaded dataset with {len(self.data)} images")
        logger.info(f"Cache directory: {self.cache_dir}")

    def _setup_session(self):
        """Set up requests session with browser-like configuration"""
        self.session = requests.Session()

        # Set comprehensive browser headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        })

        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,  # Wait 2, 4, 8 seconds between retries
            status_forcelist=[403, 429, 500, 502, 503, 504],
            raise_on_status=False  # Don't raise on retry-able status codes
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache filename from URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.jpg"

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL with retries and proper headers"""
        for attempt in range(self.max_retries):
            try:
                # Add random delay to avoid being detected as a bot
                if attempt > 0:
                    time.sleep(random.uniform(1, 3))
                else:
                    time.sleep(random.uniform(0.1, 0.5))

                # Prepare headers for image requests
                image_headers = {
                    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                    'Sec-Fetch-Dest': 'image',
                    'Sec-Fetch-Mode': 'no-cors',
                    'Sec-Fetch-Site': 'cross-site',
                }

                # Add site-specific headers
                if 'moma.org' in url:
                    image_headers['Referer'] = 'https://www.moma.org/'
                    image_headers['Origin'] = 'https://www.moma.org'
                elif 'cooperhewitt.org' in url:
                    image_headers['Referer'] = 'https://collection.cooperhewitt.org/'
                elif '1stdibs.com' in url:
                    image_headers['Referer'] = 'https://www.1stdibs.com/'

                response = self.session.get(
                    url,
                    headers=image_headers,
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=True
                )
                response.raise_for_status()

                # Load image from response
                image_data = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    image_data.write(chunk)
                image_data.seek(0)

                image = Image.open(image_data).convert('RGB')
                logger.debug(f"Successfully downloaded: {url}")
                return image

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    logger.warning(f"Access forbidden for {url} (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        # Try with different headers on 403
                        continue
                logger.error(f"HTTP error downloading {url}: {e}")
                if attempt == self.max_retries - 1:
                    self.failed_downloads.add(url)
                    return None
            except Exception as e:
                logger.error(f"Error downloading {url} (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    self.failed_downloads.add(url)
                    return None

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
                logger.debug(f"Cached image: {cache_path}")
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


# Utility function for pre-downloading all images with improved headers
def download_all_images(split_files: list, cache_dir: str, num_workers: int = 4):
    """Pre-download all images for faster training with proper headers"""
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

    def create_session():
        """Create a session with browser-like headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        return session

    def download_single(url):
        cache_path = cache_dir / f"{hashlib.md5(url.encode()).hexdigest()}.jpg"
        if cache_path.exists():
            return url, True, "cached"

        session = create_session()

        try:
            # Add site-specific headers
            headers = {}
            if 'moma.org' in url:
                headers['Referer'] = 'https://www.moma.org/'
            elif 'cooperhewitt.org' in url:
                headers['Referer'] = 'https://collection.cooperhewitt.org/'

            # Add small delay
            time.sleep(random.uniform(0.1, 0.5))

            response = session.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            # Read image data
            image_data = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                image_data.write(chunk)
            image_data.seek(0)

            image = Image.open(image_data).convert('RGB')
            image.save(cache_path, 'JPEG', quality=95)
            return url, True, "downloaded"
        except Exception as e:
            return url, False, str(e)
        finally:
            session.close()

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
    from transforms import get_transforms_for_model  # Use your custom transforms

    # Test with EfficientNet-B2 transforms (260x260 input size)
    transform = get_transforms_for_model('efficientnet-b2', is_training=False)

    # Test loading
    dataset = URLDataset(
        split_file='../data/splits/train.json',
        transform=transform,
        cache_dir='../data/cache/images'
    )

    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Decades: {dataset.decades}")

    # Try loading first few images
    for i in range(min(5, len(dataset))):
        image, label, metadata = dataset[i]
        print(f"Image {i}: shape={image.shape}, label={label} ({metadata['decade']}), name={metadata['name']}")

    # Show failed downloads if any
    failed = dataset.get_failed_downloads()
    if failed:
        print(f"\nFailed downloads: {len(failed)}")
        for url in list(failed)[:3]:  # Show first 3
            print(f"  {url}")
        if len(failed) > 3:
            print(f"  ... and {len(failed) - 3} more")