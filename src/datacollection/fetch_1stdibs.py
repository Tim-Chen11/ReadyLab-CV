#!/usr/bin/env python3
import json, time, random, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.1stdibs.com/furniture/lighting/origin/american/?per=1960s"
HEADERS = {"User-Agent": "Mozilla/5.0 (scraping-demo)"}

def scrape_page(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    ld = json.loads(soup.find("script", type="application/ld+json").string)[0]
    for item in ld["mainEntity"]["offers"]["itemOffered"]:
        print(f"{item['name']} | {item['image']} | {item['url']}")

    next_link = soup.find("link", rel="next")
    return urljoin(url, next_link["href"]) if next_link else None

url = BASE
while url:
    url = scrape_page(url)
    time.sleep(random.uniform(1, 2))        # small delay to be polite
