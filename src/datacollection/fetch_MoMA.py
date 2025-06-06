from src.datacollection.design_object_model import DesignObject
from typing import List
import csv
from pathlib import Path
import re

###################################Part for fetching testing - ignore#######################################################
file_path = Path("data/raw/Artworks.csv")

def normalize_nationality(raw: str) -> List[str]:
    # extract all text inside parentheses
    entries = re.findall(r"\((.*?)\)", raw)
    # strip and remove empty entries
    entries = [e.strip() for e in entries if e.strip()]
    return entries

def extract_nationalities(csv_path: str):
    nationalities = set()

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_nat = row.get("Nationality", "").strip()
            if not raw_nat:
                continue

            extracted = normalize_nationality(raw_nat)
            nationalities.update(extracted)

    return sorted(nationalities)

def fetch_nationalities():
    nationalities = extract_nationalities(file_path)

    print(f"Found {len(nationalities)} unique nationalities:\n")
    for nat in nationalities:
        print(nat)

from collections import defaultdict
def count_classifications(csv_path: str):
    classification_counts = defaultdict(int)
    total_items = 0

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_items += 1
            classification = row.get("Classification", "").strip()
            if classification:
                classification_counts[classification] += 1

    return total_items, classification_counts


def fetch_classifications():
    file_path = Path("data/raw/Artworks.csv")  # ✅ adjust path if needed

    total_items, classification_counts = count_classifications(file_path)

    print(f"Total items: {total_items}")
    print(f"Found {len(classification_counts)} unique classifications:\n")

    # Sort by count descending
    sorted_counts = sorted(classification_counts.items(), key=lambda x: -x[1])

    for classification, count in sorted_counts:
        print(f"{classification}: {count} items")


def extract_departments(csv_path: str):
    departments = defaultdict(int)
    total_items = 0

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_items += 1
            department = row.get("Department", "").strip()
            if department:
                departments[department] += 1

    return total_items, departments


def fetch_departments():
    file_path = Path("data/raw/Artworks.csv")  # ✅ your real path

    total_items, departments = extract_departments(file_path)

    print(f"Total items: {total_items}")
    print(f"Found {len(departments)} unique departments:\n")

    # Sort by count descending
    sorted_counts = sorted(departments.items(), key=lambda x: -x[1])

    for department, count in sorted_counts:
        print(f"{department}: {count} items")

def extract_mediums(csv_path: str):
    mediums = defaultdict(int)
    total_items = 0

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_items += 1
            medium = row.get("Medium", "").strip()
            if medium:
                mediums[medium] += 1

    return total_items, mediums

def fetch_mediums():
    file_path = Path("data/raw/Artworks.csv")  # ✅ your actual path here

    total_items, mediums = extract_mediums(file_path)

    print(f"Total items: {total_items}")
    print(f"Found {len(mediums)} unique mediums:\n")

    # Sort by count descending
    sorted_counts = sorted(mediums.items(), key=lambda x: -x[1])

    for medium, count in sorted_counts:
        if count > 40:  # ✅ Only output mediums with more than 40 items
            print(f"{medium}: {count} items")

#########################################################################################################

from curl_cffi import requests
from curl_cffi.requests import BrowserType
import json, ssl, time, math, warnings, os, re
from urllib.parse import unquote, urljoin
from pyquery import PyQuery as pq

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")


class DzSpider(object):
    def __init__(self):
        self.data = {}
        self.folder = fr'{os.getcwd()}'
        self.start_page = 1
        self.end_page = 5
        self.spider_num = 1
        self.page_size = 40
        self.has_finish = False
        self.reset_end_page = True
        self.session = None
        self.headers = {
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'cache-control': 'no-cache',
            'content-type': 'text/plain',
            'pragma': 'no-cache',
            'referer': 'https://www.moma.org/collection/',
            'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
            'x-requested-with': 'XMLHttpRequest'
        }

    def init_session(self):
        """Initialize session with proper configuration"""
        # Try different browser types in order of preference
        browser_types = [
            'chrome120',
            'chrome110',
            'chrome101',
            'chrome99',
            'safari15_5',
            'edge99'
        ]

        for browser_type in browser_types:
            try:
                if hasattr(BrowserType, browser_type):
                    self.session = requests.Session(
                        impersonate=getattr(BrowserType, browser_type),
                        timeout=30
                    )
                    print(f"Session initialized successfully with {browser_type}")
                    return True
            except Exception as e:
                print(f"Failed to initialize with {browser_type}: {e}")
                continue

        # Fallback: try without impersonate
        try:
            self.session = requests.Session(timeout=30)
            print("Session initialized successfully (fallback mode)")
            return True
        except Exception as e:
            print(f"Failed to initialize session completely: {e}")
            return False

    def run_task(self):
        if not self.init_session():
            print("Failed to initialize session, exiting...")
            return

        page_index = self.start_page
        self.reset_end_page = True
        self.has_finish = False

        print(f"Starting scraper from page {self.start_page} to {self.end_page}")

        while page_index <= self.end_page and not self.has_finish:
            try:
                print(f"\n{'=' * 50}")
                print(f"Processing page {page_index}/{self.end_page}")
                print(f"{'=' * 50}")

                success = self.get_one_page(page_index)
                if not success:
                    print(f"Failed to process page {page_index}, stopping...")
                    break

                page_index += 1
                time.sleep(1)

            except KeyboardInterrupt:
                print("\nScraping interrupted by user")
                break
            except Exception as e:
                print(f"Error processing page {page_index}: {e}")
                page_index += 1
                continue

        print(f"\nScraping completed! Total items processed: {self.spider_num - 1}")


    def get_one_page(self, page_index: int) -> List[DesignObject]:
        req_url = "https://www.moma.org/collection/"
        params = {
            "classifications": "37",
            "date_begin": "1960",
            "date_end": "2010",
            "include_uncataloged_works": "false",
            "on_view": "false",
            "recent_acquisitions": "false",
            "with_images": "true",
            "page": page_index,
        }

        print(f"Fetching page {page_index} …")
        html = self.session.get(req_url, headers=self.headers, params=params).text
        doc = pq(html)
        items = doc("li a")
        objects: list[DesignObject] = []

        for i, a in enumerate(items.items(), 1):
            href = urljoin(req_url, a.attr("href") or "")
            makers = a("div>div:eq(1) p:eq(0)").text()
            title = a("div>div:eq(1) p:eq(1)").text()
            year = a("div>div:eq(1) p:eq(2)").text()

            detail = self.get_detail(href)

            obj = DesignObject(
                name=title,
                year=year,  # raw string
                classification=detail["classification"],
                dimension=detail["dimension"],
                makers=[makers],  # raw, unsplit
                image_urls=detail["image_urls"],
                country="USA",
                source="https://www.moma.org/",
            )
            objects.append(obj)
            print(f"✔ {i}/{len(items)}  {obj}")

        return objects

    def get_detail(self, href: str) -> dict:
        detail = {
            "image_urls": [],
            "dimension": "N/A",
            "classification": "N/A",  # department → classification
        }

        try:
            resp = self.session.get(href, headers=self.headers, timeout=30)
            resp.raise_for_status()
            doc = pq(resp.text)

            # -------- images before <h1> -----------
            main = doc("#main") or doc
            title = main.find("h1").eq(0)

            seen = set()
            for node in main.find("*"):
                if node is title[0]:
                    break
                for img in pq(node).find("img").items():
                    src = img.attr("src")
                    if src and src not in seen:
                        seen.add(src)
                        detail["image_urls"].append(urljoin(href, src))

            # -------- dimensions --------
            dim_dd = (
                    doc('#caption dt:contains("Dimensions")').next("dd")
                    or doc('dt:contains("Dimensions")').next("dd")
            )
            if dim_dd:
                detail["dimension"] = dim_dd.text().strip()

            # -------- department classification -------------
            dept_dd = (
                    doc('#caption dt:contains("Department")').next("dd")
                    or doc('dt:contains("Department")').next("dd")
            )
            if dept_dd:
                detail["classification"] = dept_dd.text().strip()

        except Exception as exc:
            print(f"Error scraping {href}: {exc}")

        return detail


if __name__ == '__main__':
    print("Starting MoMA Collection Scraper")

    spider = DzSpider()
    spider.run_task()

    print("\nScraper finished!")