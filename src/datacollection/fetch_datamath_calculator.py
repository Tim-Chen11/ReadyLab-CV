import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin, urlparse
import time
import json
import os
from typing import List, Optional
from dataclasses import dataclass, asdict
import openpyxl
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

@dataclass
class DesignObject:
    name: str
    year: str
    classification: str
    dimension: str
    makers: List[str]
    image_urls: List[str]
    country: str
    price: Optional[str] = None
    popularity: Optional[str] = None
    source: Optional[str] = None

class DatamathCompleteScraper:
    def __init__(self):
        self.base_url = "http://www.datamath.org/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.calculator_links = []
        
    def collect_links_from_page(self, url, category_name):
        """Step 1: Collect ALL .htm links from a page"""
        print(f"\nProcessing: {category_name}")
        print(f"URL: {url}")
        print("-" * 50)
        
        links_data = []
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Get ALL links
                all_links = soup.find_all('a', href=True)
                
                for link in all_links:
                    href = link['href']
                    link_text = link.get_text().strip()
                    
                    # Skip anchors, mailto, and external links
                    if href.startswith('#') or href.startswith('mailto:'):
                        continue
                    if href.startswith('http') and 'datamath.org' not in href:
                        continue
                    
                    # Get all .htm and .html files
                    if href.endswith('.htm') or href.endswith('.html'):
                        # Build full URL
                        if href.startswith('http'):
                            full_url = href
                        else:
                            full_url = urljoin(url, href)
                        
                        # Extract filename
                        filename = os.path.basename(urlparse(full_url).path)
                        
                        # Skip index/main pages
                        skip_files = ['index.htm', 'main.htm', 'start.htm', 'album_']
                        if any(skip in filename.lower() for skip in skip_files):
                            continue
                        
                        links_data.append({
                            'category': category_name,
                            'name': link_text if link_text else filename.replace('.htm', ''),
                            'filename': filename,
                            'url': full_url
                        })
                        print(f"  Found: {link_text[:50] if link_text else filename}")
                
                print(f"  Total links found: {len(links_data)}")
                
            else:
                print(f"  Error: Status code {response.status_code}")
                
        except Exception as e:
            print(f"  Error fetching page: {e}")
        
        return links_data
    
    def collect_all_links(self):
        """Step 1: Collect all calculator links from the 10 Album pages"""
        print("=" * 70)
        print(" STEP 1: COLLECTING ALL CALCULATOR LINKS")
        print("=" * 70)
        
        # The 10 main category pages
        categories = [
            {'filename': 'Album_Basic.htm', 'name': 'Basic Calculators'},
            {'filename': 'Album_Desktop.htm', 'name': 'Desktop Calculators'},
            {'filename': 'Album_Sci.htm', 'name': 'Scientific Calculators'},
            {'filename': 'Album_Graph.htm', 'name': 'Graphing Calculators'},
            {'filename': 'Album_Edu.htm', 'name': 'Educational Products'},
            {'filename': 'Album_Personal.htm', 'name': 'Personal Calculators'},
            {'filename': 'Album_Speech.htm', 'name': 'Speech Products'},
            {'filename': 'Album_TISTUFF.htm', 'name': 'TI Stuff'},
            {'filename': 'Album_Others.htm', 'name': 'Other Brands'},
            {'filename': 'Album_Related.htm', 'name': 'Related Products'},
        ]
        
        all_links = []
        
        # Process each category
        for category in categories:
            url = urljoin(self.base_url, category['filename'])
            links = self.collect_links_from_page(url, category['name'])
            all_links.extend(links)
            time.sleep(0.5)  # Be polite
        
        # Remove duplicates based on URL
        unique_links = []
        seen_urls = set()
        for link in all_links:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])
        
        print("\n" + "=" * 70)
        print(f" LINKS COLLECTION SUMMARY")
        print("=" * 70)
        print(f"Total links collected: {len(all_links)}")
        print(f"Unique links: {len(unique_links)}")
        
        self.calculator_links = unique_links
        return unique_links
    
    def scrape_calculator_page(self, url):
        """Step 2: Scrape a single calculator page for DesignObject data"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            all_text = soup.get_text()
            
            # Initialize DesignObject fields
            name = ""
            year = ""
            dimension = ""
            country = ""
            image_urls = []
            
            # 1. Extract NAME from title
            title = soup.find('title')
            if title:
                title_text = title.get_text().strip()
                # Remove "Texas Instruments" or "DATAMATH" prefix
                name = title_text.replace('Texas Instruments', '').replace('DATAMATH', '').strip()
                # Clean up common patterns
                name = re.sub(r'^\s*-\s*', '', name)  # Remove leading dash
                name = name.strip()
            
            # If no name from title, try to get from URL
            if not name:
                filename = os.path.basename(urlparse(url).path)
                name = filename.replace('.htm', '').replace('_', ' ').replace('-', ' ')
            
            # 2. Extract YEAR (full Date of manufacture text)
            # Look for "Date of manufacture" first
            manufacture_pattern = r'Date of manufacture:\s*([^\n|]+)'
            match = re.search(manufacture_pattern, all_text, re.IGNORECASE)
            if match:
                year = match.group(1).strip()
                # Clean up
                year = re.sub(r'\s+', ' ', year)  # Remove extra spaces
            
            # If no manufacture date, try introduction date as fallback
            if not year:
                intro_pattern = r'Date of introduction:\s*([^\n|]+)'
                match = re.search(intro_pattern, all_text, re.IGNORECASE)
                if match:
                    year = match.group(1).strip()
                    year = re.sub(r'\s+', ' ', year)
            
            # 3. Extract DIMENSION (Physical Size, not Display size)
            # Make sure we get "Size:" not "Display size:"
            # Use negative lookbehind to exclude "Display size:"
            size_pattern = r'(?<!Display\s)Size:\s*([^\n|]+)'
            match = re.search(size_pattern, all_text, re.IGNORECASE)
            if match:
                dimension = match.group(1).strip()
                # Clean up dimension text
                dimension = re.sub(r'\s+', ' ', dimension)  # Remove extra spaces
                # Keep the full dimension text (inches and mm)
            
            # 4. Extract COUNTRY from Origin of manufacture
            origin_pattern = r'Origin of manufacture:\s*([^\n|]+)'
            match = re.search(origin_pattern, all_text, re.IGNORECASE)
            if match:
                country = match.group(1).strip()
                # Clean up country text
                country = re.sub(r'\s+', ' ', country)  # Remove extra spaces
                # Remove any parenthetical info
                country = re.sub(r'\([^)]*\)', '', country).strip()
                # Common mappings
                country_map = {
                    'USA': 'United States',
                    'US': 'United States',
                    'Taiwan (C)': 'Taiwan',
                    'Taiwan (I)': 'Taiwan',
                    'Taiwan': 'Taiwan',
                    'Japan': 'Japan',
                    'Thailand': 'Thailand',
                    'China': 'China',
                    'Italy': 'Italy',
                    'Brazil': 'Brazil',
                    'Malaysia': 'Malaysia',
                    'Philippines': 'Philippines',
                }
                for key, value in country_map.items():
                    if key.lower() in country.lower():
                        country = value
                        break
            
            # 5. Extract IMAGE URLs (front calculator image only)
            # Look for the main calculator image, not banners or icons
            images = soup.find_all('img')
            for img in images:
                src = img.get('src', '')
                if src:
                    src_lower = src.lower()
                    
                    # Skip navigation/logo/banner images
                    skip_patterns = ['logo', 'button', 'arrow', 'home', 'mail', 'icon', 
                                   'banner', 'pdf.gif', 'new.gif', 'hot.gif']
                    if any(skip in src_lower for skip in skip_patterns):
                        continue
                    
                    # Look for calculator images
                    # Prefer images in IMAGES folder or containing the model name
                    if any(ext in src_lower for ext in ['.jpg', '.jpeg', '.gif', '.png']):
                        # Check if it's in IMAGES folder (main calculator images are usually here)
                        if 'images/' in src_lower or 'IMAGES/' in src:
                            # Build full URL
                            if src.startswith('http'):
                                full_url = src
                            else:
                                full_url = urljoin(url, src)
                            
                            image_urls = [full_url]
                            break  # Found the right image
                        
                        # If no IMAGES folder image yet, check if filename contains calculator model
                        elif not image_urls:
                            # Extract model name/number from the page name
                            page_name = os.path.basename(urlparse(url).path).replace('.htm', '')
                            if page_name.lower() in src_lower:
                                if src.startswith('http'):
                                    full_url = src
                                else:
                                    full_url = urljoin(url, src)
                                
                                image_urls = [full_url]
                                # Don't break, keep looking for better match in IMAGES folder
            
            # Create DesignObject
            design_object = DesignObject(
                name=name,
                year=year,  # Now contains full date text like "mth 08 year 1972"
                classification="calculator",
                dimension=dimension,
                makers=[],  # Empty list as requested
                image_urls=image_urls,
                country=country,
                price=None,
                popularity=None,
                source="http://www.datamath.org/"
            )
            
            return design_object
            
        except Exception as e:
            print(f"    Error scraping {url}: {e}")
            return None
    
    def scrape_all_calculators(self, limit=None, max_workers=10):
        """Step 2: Scrape all calculator pages for design data using parallel processing"""
        print("\n" + "=" * 70)
        print(" STEP 2: SCRAPING CALCULATOR DATA (PARALLEL)")
        print("=" * 70)
        
        if not self.calculator_links:
            print("No links to scrape!")
            return []
        
        # Apply limit if specified
        links_to_scrape = self.calculator_links[:limit] if limit else self.calculator_links
        
        if limit:
            print(f"Limiting to first {limit} links for testing")
        
        print(f"\nStarting to scrape {len(links_to_scrape)} calculator pages using {max_workers} threads...")
        print("-" * 70)
        
        design_objects = []
        successful = 0
        failed = 0
        completed = 0
        
        # Thread-safe counter lock
        counter_lock = threading.Lock()
        
        def scrape_single_calculator(link_info):
            """Helper function to scrape a single calculator page"""
            nonlocal successful, failed, completed
            
            url = link_info['url']
            name = link_info['name']
            category = link_info['category']
            
            try:
                design_obj = self.scrape_calculator_page(url)
                
                with counter_lock:
                    completed += 1
                    progress_pct = (completed / len(links_to_scrape)) * 100
                    
                    if design_obj and design_obj.name:
                        successful += 1
                        print(f"[{completed:4d}/{len(links_to_scrape)}] ({progress_pct:5.1f}%) OK {name[:40]:<40} ({category})")
                        return design_obj
                    else:
                        failed += 1
                        print(f"[{completed:4d}/{len(links_to_scrape)}] ({progress_pct:5.1f}%) FAIL {name[:40]}")
                        return None
            except Exception as e:
                with counter_lock:
                    completed += 1
                    progress_pct = (completed / len(links_to_scrape)) * 100
                    failed += 1
                    print(f"[{completed:4d}/{len(links_to_scrape)}] ({progress_pct:5.1f}%) ERR {name[:40]} - {str(e)[:20]}")
                    return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_link = {
                executor.submit(scrape_single_calculator, link_info): link_info 
                for link_info in links_to_scrape
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_link):
                result = future.result()
                if result:
                    design_objects.append(result)
        
        print("\n" + "=" * 70)
        print(" SCRAPING COMPLETE")
        print("=" * 70)
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total processed: {completed}")
        
        return design_objects
    
    def save_to_xlsx(self, design_objects, filename="datamath_calculators.xlsx"):
        """Save DesignObject list to XLSX file with ||| separated lists."""
        if not design_objects:
            print("No data to save")
            return None
        
        # Create output path same as mobile phone museum: ../../data/metadata/
        script_dir = Path(__file__).parent / '..' / '..' / 'data'
        output_path = script_dir / 'metadata' / filename
        
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to rows for DataFrame
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
        print(f"\nSaved to {output_path}")
        
        # Display sample
        print("\n" + "=" * 70)
        print(" SAMPLE DATA:")
        print("=" * 70)
        sample_df = df[['name', 'year', 'country', 'dimension']].head(20)
        print(sample_df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "=" * 70)
        print(" SUMMARY:")
        print("=" * 70)
        print(f"Total calculators: {len(df)}")
        print(f"With year data: {df['year'].notna().sum()}")
        print(f"With country data: {df['country'].notna().sum()}")
        print(f"With dimension data: {df['dimension'].notna().sum()}")
        print(f"With images: {sum(1 for row in rows if row['image_urls'])}")
        
        return df

def main():
    print("=" * 70)
    print(" DATAMATH CALCULATOR MUSEUM - COMPLETE SCRAPER")
    print("=" * 70)
    print("\nThis will:")
    print("1. Collect all calculator links from 10 Album pages")
    print("2. Scrape each calculator page for design data")
    print("3. Save results to JSON")
    print("-" * 70)
    
    scraper = DatamathCompleteScraper()
    
    # Options
    print("\nOptions:")
    print("1. Test with first 10 calculators")
    print("2. Test with first 50 calculators")
    print("3. Scrape ALL calculators")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    # Ask for number of threads for parallel processing
    print("\nParallel processing settings:")
    print("- More threads = faster scraping")
    print("- Too many threads may overload the server")
    print("- Recommended: 5-15 threads")
    
    while True:
        try:
            max_workers = int(input("\nEnter number of threads (default 10): ") or "10")
            if max_workers > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Step 1: Collect all links
    links = scraper.collect_all_links()
    
    if not links:
        print("\nNo links found! Exiting.")
        return
    
    # Step 2: Scrape calculator data based on choice
    if choice == '1':
        print(f"\nThis will scrape the first 10 calculators using {max_workers} threads.")
        design_objects = scraper.scrape_all_calculators(limit=10, max_workers=max_workers)
    elif choice == '2':
        print(f"\nThis will scrape the first 50 calculators using {max_workers} threads.")
        design_objects = scraper.scrape_all_calculators(limit=50, max_workers=max_workers)
    else:
        print(f"\nThis will scrape {len(links)} calculators using {max_workers} threads.")
        print("This may take some time but will be much faster than sequential processing!")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            design_objects = scraper.scrape_all_calculators(max_workers=max_workers)
        else:
            print("Cancelled.")
            return
    
    # Step 3: Save results to Excel
    if design_objects:
        scraper.save_to_xlsx(design_objects)
        print("\n" + "=" * 70)
        print(" ALL DONE!")
        print("=" * 70)
        print("\nOutput file:")
        print("- ../../data/metadata/datamath_calculators.xlsx (all calculator data)")

if __name__ == "__main__":
    main()