import os
import json
import requests
from dotenv import load_dotenv

# Load token from .env
load_dotenv()
EBAY_OAUTH_TOKEN = os.getenv("EBAY_CLIENT_SECRET", None)

if not EBAY_OAUTH_TOKEN:
    raise RuntimeError("❌ Missing EBAY_CLIENT_SECRET in your .env file")

def get_item_specifics(keyword):
    # Step 1: Search by keyword
    search_url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {EBAY_OAUTH_TOKEN}",
        "Content-Type": "application/json",
    }
    params = {
        "q": keyword,
        "limit": 5
    }

    search_resp = requests.get(search_url, headers=headers, params=params)
    if search_resp.status_code != 200:
        print("❌ Search failed:", search_resp.status_code, search_resp.text)
        return

    items = search_resp.json().get("itemSummaries", [])
    if not items:
        print("⚠️ No items found.")
        return

    item_id = items[0]["itemId"]
    print(f"✅ Found itemId: {item_id}")

    # Step 2: Lookup full item details
    item_url = f"https://api.ebay.com/buy/browse/v1/item/{item_id}"
    item_resp = requests.get(item_url, headers=headers)
    if item_resp.status_code != 200:
        print("❌ Item details failed:", item_resp.status_code, item_resp.text)
        return

    item_data = item_resp.json()

    # Step 3: Print item specifics
    print("\n🔍 Item Specifics:")
    specifics = item_data.get("itemSpecifics", [])
    if specifics:
        for spec in specifics:
            print(f"- {spec['name']}: {', '.join(spec['values'])}")
    else:
        print("⚠️ No item specifics available.")

# Example usage
get_item_specifics("chair")
