#!/usr/bin/env python3
"""
harvest_1stdibs.py
------------------
Scrape an entire 1stDibs *listing* (with pagination) and return
List[DesignObject] dataclass instances populated with raw strings.

USAGE EXAMPLE
-------------
from harvest_1stdibs import harvest_listing, DesignObject

objects = harvest_listing(
    "https://www.1stdibs.com/furniture/lighting/origin/american/?per=1960s"
)
print(objects[0])
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import json, time, random, requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# ------------------------------------------------------------------#
# 1.  Your dataclass (year is str)                                  #
# ------------------------------------------------------------------#

from src.datacollection.design_object_model import DesignObject

# ------------------------------------------------------------------#
# 2.  HTTP helpers                                                  #
# ------------------------------------------------------------------#
UA      = "Mozilla/5.0 (compatible; 1stdibs-design-objects/1.1)"
HEADERS = {"User-Agent": UA}

LISTING_DELAY = 1.0          # between paginated listing pages
DETAIL_DELAY  = (1.0, 2.0)   # random delay between detail pages

def _soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def _jsonld(soup: BeautifulSoup):
    tag = soup.find("script", type="application/ld+json")
    return json.loads(tag.string) if tag else None


# ------------------------------------------------------------------#
# 3.  Detail-page → DesignObject                                    #
# ------------------------------------------------------------------#
def _spec_text(soup: BeautifulSoup, tn: str) -> str:
    blk = soup.find("div", attrs={"data-tn": tn})
    return " ".join(blk.stripped_strings) if blk else ""

def _normalize_country(raw: str) -> str:
    raw_l = raw.lower()
    if "united states" in raw_l or "usa" in raw_l:
        return "USA"
    if "canada" in raw_l:
        return "Canada"
    return raw.strip().title() if raw else ""

def _classification(detail_url: str, product: dict) -> str:
    if cat := product.get("category"):
        return cat
    segs = detail_url.split("/")
    if "furniture" in segs:
        i = segs.index("furniture")
        if i + 1 < len(segs):
            return segs[i + 1]
    return "unknown"

def _price_string(product: dict) -> Optional[str]:
    offers = product.get("offers", [])
    cad = next((o for o in offers if o.get("priceCurrency") == "CAD"), None)
    offer = cad or (offers[0] if offers else {})
    if "price" in offer:
        cur = offer.get("priceCurrency", "")
        return f"{cur}${float(offer['price']):,.2f} per item"
    return None

# --- NEW helper (replace the old _extract_makers) -----------------
def _extract_makers(soup: BeautifulSoup) -> List[str]:
    """
    Return a single-element list whose item is the raw text that appears
    under “Creator:” or “Attributed to:”. If both rows exist, concatenate
    them with a comma, exactly as shown on the page.
    """
    raw_creator   = _spec_text(soup, "pdp-spec-creator")
    raw_attrib_to = _spec_text(soup, "pdp-spec-attributed-to")
    raw_designer  = _spec_text(soup, "pdp-spec-designer")

    combined = ", ".join(filter(None, (raw_creator, raw_attrib_to, raw_designer))).strip()
    return [combined] if combined else []


def detail_to_design_object(detail_url: str) -> DesignObject:
    soup = _soup(detail_url)
    ld   = _jsonld(soup) or {}
    prod = ld[0] if isinstance(ld, list) else ld

    name        = prod.get("name", "").strip()
    year_raw = (_spec_text(soup, "pdp-spec-date-of-manufacture") or prod.get("productionDate", ""))
    dimension   = _spec_text(soup, "pdp-spec-dimensions")
    country_raw = _spec_text(soup, "pdp-spec-place-of-origin")
    country     = _normalize_country(country_raw)

    image_blobs = prod.get("image", [])
    image_urls  = [b["contentUrl"] if isinstance(b, dict) else b for b in image_blobs]

    return DesignObject(
        name            = name,
        year            = year_raw,               # raw string
        classification   = _classification(detail_url, prod),
        dimension       = dimension,
        makers          = _extract_makers(soup),
        image_urls      = image_urls,
        country         = country,
        price           = _price_string(prod),
        source          = "https://www.1stdibs.com/",
    )


# ------------------------------------------------------------------#
# 4.  Listing-page traversal                                        #
# ------------------------------------------------------------------#
def _detail_urls(listing_url: str):
    """Yield every unique detail URL across paginated listing."""
    seen, url = set(), listing_url
    while url:
        soup = _soup(url)
        data = _jsonld(soup) or {}
        wrapper = data[0] if isinstance(data, list) else data
        items = (
            wrapper.get("mainEntity", {})
                   .get("offers", {})
                   .get("itemOffered", [])
        )
        for it in items:
            durl = it.get("url")
            if durl and durl not in seen:
                seen.add(durl)
                yield durl
        nxt = soup.find("link", rel="next")
        url = urljoin(url, nxt["href"]) if nxt else None
        time.sleep(LISTING_DELAY)

# ------------------------------------------------------------------#
# 5.  Public API                                                    #
# ------------------------------------------------------------------#
def harvest_listing(listing_url: str) -> List[DesignObject]:
    results: List[DesignObject] = []
    for durl in _detail_urls(listing_url):
        try:
            obj = detail_to_design_object(durl)
            print(obj)
            results.append(obj)
        except Exception as exc:
            print(f"[warn] failed {durl}: {exc}")
        time.sleep(random.uniform(*DETAIL_DELAY))
    return results


# ------------------------------------------------------------------#
# 6.  Quick CLI demo                                                #
# ------------------------------------------------------------------#
if __name__ == "__main__":
    demo = "https://www.1stdibs.com/fashion/handbags-purses-bags/?origin=american,canadian&per=1960s,1970s,1980s,1990s,21st-century-and-contemporary&sort=newest"
    print("Harvesting demo listing …")
    data = harvest_listing(demo)
    print(f"Scraped {len(data)} objects — first one:")
    print(data[0])