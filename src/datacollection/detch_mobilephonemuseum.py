#!/usr/bin/env python3
"""
Advanced Mobile Phone Museum Scraper
Handles JavaScript-heavy SPA website with dynamic content loading
"""

import json
import logging
import re
import time
from typing import List, Set
from urllib.parse import urljoin
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MobilePhoneMuseumScraper:
    def __init__(self, headless: bool = False, delay: float = 2.0):
        """Initialize scraper with browser settings."""
        self.base_url = "https://www.mobilephonemuseum.com"
        self.catalogue_url = f"{self.base_url}/catalogue"
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.phone_links: Set[str] = set()
        self.driver = self._setup_driver(headless)

    def _setup_driver(self, headless: bool) -> webdriver.Chrome:
        """Set up Selenium WebDriver with Chrome."""
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument(f"--user-agent={self.headers['User-Agent']}")
        options.add_experimental_option("prefs", {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2
        })

        try:
            driver = webdriver.Chrome(options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("Chrome WebDriver initialized successfully")
            return driver
        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            raise

    def _wait_for_element(self, by: str, value: str, timeout: float = 10):
        """Wait for an element to be present."""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
        except TimeoutException:
            return None

    def _scroll_and_load_content(self):
        """Scroll page to load dynamic content."""
        print("🔄 Scrolling to load all content...")
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        max_scrolls = 50
        scroll_attempts = 0

        while scroll_attempts < max_scrolls:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            # Check for load more buttons
            buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Load') or contains(text(), 'More') or contains(text(), 'Show')]")
            for button in buttons:
                try:
                    if button.is_displayed() and button.is_enabled():
                        self.driver.execute_script("arguments[0].click();", button)
                        print(f"   ✅ Clicked load more button")
                        time.sleep(0.5)
                        break
                except:
                    continue

            # Count current links
            current_links = len(self.driver.find_elements(By.XPATH, "//a[contains(@href, 'phone') or contains(@href, 'detail')]"))
            print(f"   📱 Found {current_links} phone links so far...")

            # Check for pagination
            pagination = self.driver.find_elements(By.XPATH, "//button[@class='pagination'] | //a[@class='next'] | //button[contains(@class, 'load')] | //div[contains(@class, 'load-more')]")
            clicked = False
            for element in pagination:
                try:
                    if element.is_displayed() and element.is_enabled():
                        self.driver.execute_script("arguments[0].click();", element)
                        print(f"   ✅ Clicked pagination element")
                        time.sleep(0.5)
                        clicked = True
                        break
                except:
                    continue

            if new_height == last_height and not clicked:
                print(f"   ⏹️ No more content to load (attempt {scroll_attempts + 1})")
                break

            last_height = new_height
            scroll_attempts += 1
            time.sleep(0.5 + (scroll_attempts * 0.1))

        print(f"✅ Finished scrolling after {scroll_attempts} attempts")

    def _extract_phone_links(self) -> Set[str]:
        """Extract phone links from the current page."""
        print("🔍 Extracting all phone links from page...")
        link_selectors = [
            "//a[contains(@href, 'phone-detail')]",
            "//a[contains(@href, '/detail/')]",
            "//a[contains(@href, '/phone/')]",
            "//div[contains(@class, 'catalogue')]//a"
        ]
        links = set()

        # Extract from HTML
        for selector in link_selectors:
            try:
                elements = self.driver.find_elements(By.XPATH, selector)
                for element in elements:
                    href = element.get_attribute('href')
                    if href and ('phone' in href.lower() or 'detail' in href.lower()):
                        full_url = urljoin(self.base_url, href)
                        if full_url not in links:
                            links.add(full_url)
                            print(f"📱 Found: {full_url}")
            except Exception as e:
                logger.debug(f"Error with selector {selector}: {e}")

        # Extract from JavaScript
        try:
            scripts = self.driver.find_elements(By.TAG_NAME, "script")
            for script in scripts:
                content = script.get_attribute('innerHTML')
                if content:
                    urls = re.findall(r'["\']([^"\']*(?:phone-detail|/detail/)[^"\']*)["\']', content)
                    for url in urls:
                        if not url.startswith('http'):
                            full_url = urljoin(self.base_url, url)
                            if full_url not in links:
                                links.add(full_url)
                                print(f"📱 Found in JS: {full_url}")
        except Exception as e:
            logger.debug(f"Error extracting from scripts: {e}")

        return links

    def comprehensive_scrape(self) -> List[str]:
        """Execute comprehensive scraping strategy."""
        print("🚀 Starting comprehensive Mobile Phone Museum scraping...")
        print("=" * 60)

        # Strategy 2: Catalogue page scraping
        print("\n📋 Strategy 2: Loading main catalogue page...")
        try:
            self.driver.get(self.catalogue_url)
            time.sleep(0.5)
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            print(f"   📄 Page loaded, content length: {len(body_text)} characters")

            initial_links = self._extract_phone_links()
            print(f"   📱 Initial links found: {len(initial_links)}")

            self._scroll_and_load_content()
            self.phone_links.update(self._extract_phone_links())

        except Exception as e:
            logger.error(f"Error in catalogue scraping: {e}")

        print(f"\n✅ Total unique phone links found: {len(self.phone_links)}")
        return sorted(list(self.phone_links))

    def save_links(self, filename: str = "mobile_phone_museum_links.txt"):
        """Save phone links to a text file in ../../data/raw/"""
        # Relative path from the script location
        output_path = Path(__file__).parent / '..' / '..' / 'data' / 'raw' / filename
        output_path = output_path.resolve()  # Optional: get absolute path

        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

        with output_path.open('w', encoding='utf-8') as f:
            for link in sorted(self.phone_links):
                f.write(f"{link}\n")
        print(f"💾 Saved {len(self.phone_links)} links to {output_path}")

    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()

def main():
    """Main execution function."""
    print("🚀 Advanced Mobile Phone Museum Scraper")
    print("=" * 50)
    print("⚠️ Note: This website uses heavy JavaScript. Ensure Chrome is installed.")
    print("⚠️ The scraper will open a browser window to load content.\n")

    headless_choice = input("Run in headless mode? (y/n - 'n' for debugging): ").lower()
    headless = headless_choice == 'y'

    scraper = MobilePhoneMuseumScraper(headless=headless)
    try:
        phone_links = scraper.comprehensive_scrape()
        if phone_links:
            scraper.save_links()
        else:
            print("❌ No phone links found. Website structure may have changed.")
            print("💡 Try running with headless=False to debug.")
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        logger.error(f"Scraping error: {e}", exc_info=True)
    finally:
        scraper.close()
        print("\n🏁 Scraping completed!")

##################################################Parts for check the links are correct##########################################################################################################
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_link(link):
    try:
        response = requests.head(link, allow_redirects=True, timeout=10)
        if response.status_code == 404:
            return (link, "❌ 404 Not Found")
        return (link, f"✅ {response.status_code}")
    except requests.RequestException as e:
        return (link, f"⚠️ Error: {e}")


def check_links_concurrently(filename="mobile_phone_museum_links.txt", max_workers=20):
    input_path = Path(__file__).parent / '..' / '..' / 'data' / 'raw' / filename
    input_path = input_path.resolve()

    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return

    with input_path.open("r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    print(f"🔍 Checking {len(links)} links with {max_workers} threads...\n")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {executor.submit(check_link, link): link for link in links}
        for future in as_completed(future_to_link):
            link, result = future.result()
            print(f"{result}: {link}")
            results.append((link, result))

    return results





import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
import time
import json
from urllib.parse import urljoin
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from src.datacollection.design_object_model import DesignObject


class MobilePhoneMuseumScraper:
    def __init__(self):
        self.base_url = "https://www.mobilephonemuseum.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def scrape_phone_details(self, url: str) -> Dict[str, Optional[str]]:
        """
        Scrape phone details from a Mobile Phone Museum detail page.

        Args:
            url: The URL of the phone detail page

        Returns:
            Dictionary containing phone details
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Initialize result dictionary
            phone_data = {
                'url': url,
                'title': None,
                'brand': None,
                'model': None,
                'announced': None,
                'weight': None,
                'images': []
            }

            # Extract title from the page title tag
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.text.strip()
                # Remove " | Mobile Phone Museum" if present
                if ' | Mobile Phone Museum' in title_text:
                    title_text = title_text.replace(' | Mobile Phone Museum', '')
                phone_data['title'] = title_text

                # Split brand and model from title
                if ' - ' in title_text:
                    parts = title_text.split(' - ', 1)
                    phone_data['brand'] = parts[0].strip()
                    phone_data['model'] = parts[1].strip()

            # Extract brand and model from h1 tags with specific class structure
            h1_tag = soup.find('h1', class_='bb')
            if h1_tag:
                brand_span = h1_tag.find('span', class_='brandName')
                model_span = h1_tag.find('span', class_='phoneName')

                if brand_span:
                    phone_data['brand'] = brand_span.text.strip()
                if model_span:
                    phone_data['model'] = model_span.text.strip()

            # Extract specifications from the structured blocks
            blocks = soup.find_all('div', class_='block')

            for block in blocks:
                # Look for paragraphs with label spans
                paragraphs = block.find_all('p', class_='mR')

                for p in paragraphs:
                    label_span = p.find('span', class_='label')
                    if label_span:
                        label = label_span.text.strip().lower()

                        # Get the text after the label (after the <br/> tag)
                        label_span.extract()  # Remove the label span
                        value = p.text.strip()

                        if label == 'announced':
                            phone_data['announced'] = value
                        elif label == 'weight':
                            phone_data['weight'] = value

            # Alternative method: look for the specific pattern in the HTML
            # Sometimes the date is split across multiple text nodes
            if not phone_data['announced']:
                for p in soup.find_all('p', class_='mR'):
                    if 'Announced' in p.text:
                        # Extract all text content after "Announced"
                        text_parts = []
                        found_announced = False
                        for elem in p.descendants:
                            if elem.name is None:  # Text node
                                if 'Announced' in str(elem):
                                    found_announced = True
                                elif found_announced:
                                    text_parts.append(str(elem).strip())

                        if text_parts:
                            # Join the parts and clean up
                            announced_text = ' '.join(text_parts).strip()
                            phone_data['announced'] = announced_text

            # Find the related models section to know where to stop
            related_section = soup.find('div', class_='related')

            # Extract images only from the main phone section (before related models)
            seen_urls = set()

            # Method 1: Get images from the swiper container
            swiper_container = soup.find('div', class_='swiper-wrapper')
            if swiper_container:
                for slide in swiper_container.find_all('div', class_='swiper-slide'):
                    imgs = slide.find_all('img')
                    for img in imgs:
                        src = img.get('src', '')
                        srcset = img.get('srcSet', '')

                        # Extract URL from srcSet if available (usually higher quality)
                        if srcset:
                            urls = re.findall(r'(https://[^\s]+)', srcset)
                            if urls:
                                src = urls[0]  # Take the first URL

                        if src and src not in seen_urls and 'placeholder' not in src.lower():
                            phone_data['images'].append(src)
                            seen_urls.add(src)

            # Method 2: Get images before the related section
            if related_section:
                # Get all elements before the related section
                for elem in soup.find_all():
                    # Stop when we reach the related section
                    if elem == related_section:
                        break

                    # Look for images in imageWrapper divs
                    if elem.name == 'div' and 'imageWrapper' in elem.get('class', []):
                        img = elem.find('img')
                        if img:
                            src = img.get('src', '')
                            srcset = img.get('srcSet', '')

                            if srcset:
                                urls = re.findall(r'(https://[^\s]+)', srcset)
                                if urls:
                                    src = urls[0]

                            # Exclude thumbnails and placeholders
                            if (src and src not in seen_urls and
                                    'placeholder' not in src.lower() and
                                    'thumbnail' not in src.lower()):
                                phone_data['images'].append(src)
                                seen_urls.add(src)

            return phone_data

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def extract_year_from_announced(self, announced: str) -> str:
        """Extract year from announced date string."""
        if not announced:
            return "Unknown"

        # Look for 4-digit year pattern
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', announced)
        if year_match:
            return year_match.group(1)

        return "Unknown"

    def scrape_multiple_phones(self, urls: List[str], delay: float = 1.0) -> List[Dict]:
        """
        Scrape multiple phone detail pages with delay between requests.

        Args:
            urls: List of URLs to scrape
            delay: Delay in seconds between requests

        Returns:
            List of phone data dictionaries
        """
        results = []
        total = len(urls)

        for i, url in enumerate(urls, 1):
            print(f"Scraping {i}/{total}: {url}")
            data = self.scrape_phone_details(url)

            if data:
                results.append(data)
                print(f"  ✓ Found: {data['title']}")
                print(f"    Brand: {data['brand']}")
                print(f"    Model: {data['model']}")
                print(f"    Announced: {data['announced']}")
                print(f"    Weight: {data['weight']}")
                print(f"    Images: {len(data['images'])}")
            else:
                print(f"  ✗ Failed to scrape")


        return results

    def convert_to_design_objects(self, scraped_data: List[Dict]) -> List[DesignObject]:
        """Convert scraped data to DesignObject format."""
        design_objects = []

        for data in scraped_data:
            if data and data.get('images'):  # Only process if images exist
                year = self.extract_year_from_announced(data['announced'])

                # Check if year is valid and <= 2010
                try:
                    year_int = int(year)
                    if year_int > 2010:
                        continue  # Skip phones after 2010
                except ValueError:
                    # Skip if year is not a valid number (e.g., "Unknown")
                    continue

                design_obj = DesignObject(
                    name=data['model'] or "Unknown Model",
                    year=year,
                    classification="Phone",
                    dimension=data['weight'] or "Unknown",
                    makers=[data['brand']] if data['brand'] else ["Unknown"],
                    image_urls=data['images'],
                    country="USA",
                    price=None,
                    popularity=None,
                    source="https://www.mobilephonemuseum.com/"
                )
                design_objects.append(design_obj)

        return design_objects

    def save_to_xlsx(self, design_objects: List[DesignObject], output_path: Path):
        """Save DesignObject list to XLSX file with ||| separated lists."""
        if not design_objects:
            print("No data to save")
            return

        # Convert to dictionaries for DataFrame
        rows = []
        for obj in design_objects:
            row = {
                'name': obj.name,
                'year': obj.year,
                'classification': obj.classification,
                'dimension': obj.dimension,
                'makers': '|||'.join(obj.makers),
                'image_urls': '|||'.join(obj.image_urls),
                'country': obj.country,
                'price': obj.price or '',
                'popularity': obj.popularity or '',
                'source': obj.source
            }
            rows.append(row)

        # Create DataFrame and save to Excel
        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\nData saved to {output_path}")

    def save_to_json(self, data: List[Dict], filename: str = 'phone_data.json'):
        """Save scraped data to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nData saved to {filename}")

    def save_to_csv(self, data: List[Dict], filename: str = 'phone_data.csv'):
        """Save scraped data to CSV file."""
        import csv

        if not data:
            return

        fieldnames = ['url', 'title', 'brand', 'model', 'announced', 'weight', 'image_count', 'images']

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in data:
                csv_row = {
                    'url': row['url'],
                    'title': row['title'] or '',
                    'brand': row['brand'] or '',
                    'model': row['model'] or '',
                    'announced': row['announced'] or '',
                    'weight': row['weight'] or '',
                    'image_count': len(row.get('images', [])),
                    'images': ';'.join(row.get('images', []))
                }
                writer.writerow(csv_row)

        print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    main()  # Get all the phone detail page links
    check_links_concurrently()  # Check the links are correct

    # Get all the phone detail page links and store to xlsx
    scraper = MobilePhoneMuseumScraper()

    # Read URLs from file
    script_dir = Path(__file__).parent / '..' / '..' / 'data' / 'raw'
    input_file = script_dir / 'mobile_phone_museum_links.txt'

    output_path = script_dir / 'mobile_phone_museum_data.xlsx'

    # Load URLs from file
    urls = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
        print(f"Loaded {len(urls)} URLs from {input_file}")
    except Exception as e:
        print(f"Error loading URLs: {e}")
        exit(1)

    # Scrape all phones
    print("Starting to scrape phones...")
    scraped_data = scraper.scrape_multiple_phones(urls, delay=1.0)

    # Convert to DesignObject format
    design_objects = scraper.convert_to_design_objects(scraped_data)
    print(f"\nConverted {len(design_objects)} phones to DesignObject format")

    # Save to Excel
    scraper.save_to_xlsx(design_objects, output_path)