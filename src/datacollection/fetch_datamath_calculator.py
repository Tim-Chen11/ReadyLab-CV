import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from urllib.parse import urljoin, urlparse
import time

class DatamathCalculatorScraper:
    def __init__(self):
        self.base_url = "http://www.datamath.org/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_basic_calculators_urls(self):
        """Get URLs of all basic calculators from the main BASIC section"""
        basic_urls = []
        
        # The BASIC calculators are organized in different categories
        # Based on the site structure, we'll target specific sections
        basic_sections = [
            'BASIC/DATAMATH/',  # Original Datamath calculators
            'BASIC/LED_Classic/',  # Classic LED calculators
            'BASIC/LCD_Classic/',  # Classic LCD calculators
            'BASIC/LED_Modern/',  # Modern LED calculators
            'BASIC/LCD_Modern/',  # Modern LCD calculators
            'BASIC/Exactra/',  # Exactra line
        ]
        
        calculator_data = []
        
        for section in basic_sections:
            section_url = urljoin(self.base_url, section)
            print(f"Checking section: {section_url}")
            
            try:
                # Try to get the index page for each section
                response = self.session.get(section_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find all links to calculator pages (.htm files)
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if href.endswith('.htm') and not href.startswith('http'):
                            full_url = urljoin(section_url, href)
                            basic_urls.append(full_url)
                            
            except Exception as e:
                print(f"Error accessing {section_url}: {e}")
                
        # Remove duplicates
        basic_urls = list(set(basic_urls))
        print(f"Found {len(basic_urls)} calculator URLs")
        return basic_urls
    
    def scrape_calculator_page(self, url):
        """Scrape individual calculator page for name, image, and manufacture date"""
        try:
            response = self.session.get(url)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            calculator_info = {
                'url': url,
                'name': None,
                'image_url': None,
                'manufacture_date': None,
                'description': None
            }
            
            # Extract calculator name from title or h1/h2 tags
            title = soup.find('title')
            if title:
                calculator_info['name'] = title.text.strip()
            else:
                # Try to find in h1 or h2 tags
                for heading in soup.find_all(['h1', 'h2']):
                    if heading.text:
                        calculator_info['name'] = heading.text.strip()
                        break
            
            # If still no name, extract from URL
            if not calculator_info['name']:
                path = urlparse(url).path
                filename = os.path.basename(path).replace('.htm', '')
                calculator_info['name'] = filename.replace('_', ' ')
            
            # Find calculator image
            # Look for images with common patterns in their names
            for img in soup.find_all('img'):
                img_src = img.get('src', '')
                # Check if it's likely a calculator image (not a logo or button)
                if any(pattern in img_src.lower() for pattern in ['.jpg', '.png', '.gif']):
                    if not any(skip in img_src.lower() for skip in ['logo', 'button', 'icon', 'arrow']):
                        calculator_info['image_url'] = urljoin(url, img_src)
                        break
            
            # Extract manufacture date
            # Look for patterns like "1972", "September 1973", etc.
            text = soup.get_text()
            
            # Common date patterns
            date_patterns = [
                r'introduced\s+(?:in\s+)?(\d{4})',
                r'manufactured\s+(?:in\s+)?(\d{4})',
                r'(\w+\s+\d{4})',  # Month Year
                r'(\d{4})',  # Just year
                r'Date\s+introduced[:\s]+([^,\n]+)',
                r'Manufacturing\s+Date[:\s]+([^,\n]+)'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    calculator_info['manufacture_date'] = match.group(1).strip()
                    break
            
            # Extract description (first paragraph with substantial text)
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 50:  # Substantial paragraph
                    calculator_info['description'] = text[:200] + '...' if len(text) > 200 else text
                    break
            
            return calculator_info
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def scrape_basic_calculators_direct(self):
        """Direct scraping approach for known calculator pages"""
        # Sample of known basic calculator URLs from the Datamath museum
        sample_calculators = [
            'BASIC/DATAMATH/Datamath.htm',
            'BASIC/DATAMATH/TI-2500.htm',
            'BASIC/DATAMATH/TI-2500-II.htm',
            'BASIC/LED_Classic/TI-1200.htm',
            'BASIC/LED_Classic/TI-1250.htm',
            'BASIC/LCD_Classic/TI-1030.htm',
            'BASIC/LCD_Classic/TI-1750.htm',
            'BASIC/Exactra/Exactra20.htm',
            'BASIC/Exactra/Exactra21.htm',
            'BASIC/LCD_Modern/TI-108.htm',
            'BASIC/LCD_Modern/TI-1726.htm',
            'BASIC/LCD_Modern/TI-1784.htm',
        ]
        
        calculator_data = []
        
        for calc_path in sample_calculators:
            url = urljoin(self.base_url, calc_path)
            print(f"Scraping: {url}")
            
            info = self.scrape_calculator_page(url)
            if info:
                calculator_data.append(info)
                
            # Be polite to the server
            time.sleep(0.5)
        
        return calculator_data
    
    def save_results(self, data, filename='datamath_calculators.csv'):
        """Save scraped data to CSV file"""
        if data:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            return df
        else:
            print("No data to save")
            return None
    
    def download_images(self, data, folder='calculator_images'):
        """Download calculator images to local folder"""
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        for item in data:
            if item.get('image_url'):
                try:
                    img_response = self.session.get(item['image_url'])
                    if img_response.status_code == 200:
                        # Extract filename from URL
                        img_name = os.path.basename(item['image_url'])
                        if not img_name:
                            img_name = f"{item['name'].replace(' ', '_')}.jpg"
                        
                        img_path = os.path.join(folder, img_name)
                        with open(img_path, 'wb') as f:
                            f.write(img_response.content)
                        print(f"Downloaded: {img_name}")
                        
                        # Add local image path to data
                        item['local_image_path'] = img_path
                        
                except Exception as e:
                    print(f"Error downloading image from {item['image_url']}: {e}")
                    
                time.sleep(0.3)  # Be polite

def main():
    """Main function to run the scraper"""
    scraper = DatamathCalculatorScraper()
    
    print("Starting Datamath Calculator Museum scraper...")
    print("=" * 50)
    
    # Method 1: Try to discover URLs automatically
    print("\nMethod 1: Attempting to discover calculator URLs...")
    urls = scraper.get_basic_calculators_urls()
    
    if urls:
        calculator_data = []
        for url in urls[:10]:  # Limit to first 10 for demo
            print(f"Scraping: {url}")
            info = scraper.scrape_calculator_page(url)
            if info:
                calculator_data.append(info)
            time.sleep(0.5)  # Be polite
    else:
        # Method 2: Use known calculator URLs
        print("\nMethod 2: Using known calculator URLs...")
        calculator_data = scraper.scrape_basic_calculators_direct()
    
    # Save results
    if calculator_data:
        print(f"\n{'-' * 50}")
        print(f"Successfully scraped {len(calculator_data)} calculators")
        
        # Save to CSV
        df = scraper.save_results(calculator_data)
        
        # Display sample results
        if df is not None:
            print("\nSample of scraped data:")
            print(df[['name', 'manufacture_date']].head(10))
        
        # Optional: Download images
        download_images = input("\nDo you want to download calculator images? (yes/no): ").lower()
        if download_images == 'yes':
            scraper.download_images(calculator_data)
    else:
        print("No data was scraped. The website structure might have changed.")

if __name__ == "__main__":
    main()