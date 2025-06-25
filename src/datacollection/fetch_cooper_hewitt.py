from typing import Optional, List
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from src.datacollection.design_object_model import DesignObject
import os
import inspect
import pandas as pd


# --- parameterized query ---
# Query should depened on the the following parameters:
# - type: the type of the object to search for
# - year: the year of this object
#
# The result should include:
# - year of manufacture
# - type of the object
# - dimensions
# - manufacturer
# - designer (same as manufacturer)
# - Price (not available in API, stub)
# - Popularity (not available in API, stub)
# - source (cooperhewitt)
# --- basic setting ---
def create_client() -> Client:
    """
    Create and return a preconfigured GraphQL client.
    """
    transport = RequestsHTTPTransport(
        url="https://api.cooperhewitt.org/",
        headers={
            "User-Agent": "Mozilla/5.0"  # helps avoid bot filtering
        },
        retries=3,
        verify=True,
    )
    client = Client(transport=transport, fetch_schema_from_transport=True)
    return client


def normalize_country(raw_country: str) -> str:
    raw_country_lower = raw_country.lower()
    if "canada" in raw_country_lower:
        return "Canada"
    if "usa" in raw_country_lower or "u.s.a." in raw_country_lower or "united states" in raw_country_lower:
        return "USA"
    if "probably usa" in raw_country_lower or "possibly usa" in raw_country_lower:
        return "USA"
    if "usa or" in raw_country_lower:
        return "USA"
    return None


def fetch_design_objects(
        client: Client,
        department: str,
        year: int,
        country: str,
        size: int = 10,
        page: int = 0
) -> List[DesignObject]:
    variables = {
        "department": department,
        "year": year,
        "country": country,
        "size": size,
        "page": page
    }

    QUERY = gql("""
    query GetObjects(
      $department: String!,
      $year: Int!,
      $country: String!,
      $size: Int!,
      $page: Int!
    ) {
      object(
        department: $department,
        year: $year,
        country: $country,
        hasImages: true,
        size: $size,
        page: $page
      ) {
        summary
        date
        classification
        measurements
        maker { summary }
        multimedia
        geography
      }
    }
    """)
    resp = client.execute(QUERY, variable_values=variables)

    size_order = ("large", "original", "preview", "zoom")
    results: List[DesignObject] = []
    for raw in resp.get("object", []):

        geography = raw.get("geography") or {}
        country_obj = geography.get("country") or {}
        raw_country = country_obj.get("value", "")
        norm_country = normalize_country(raw_country)
        if norm_country is None:
            continue

        urls: List[str] = []
        for media in raw.get("multimedia") or []:
            if media.get("type") == "image":
                for key in size_order:
                    if key in media and media[key] and media[key].get("url"):
                        urls.append(media[key]["url"])
                        break

        classifications = raw.get("classification") or [{}]
        classification_summary = classifications[0].get("summary") or {}
        classification = classification_summary.get("title", "")

        measurements = raw.get("measurements") or {}
        dimensions = measurements.get("dimensions") or [{}]
        dimension = dimensions[0].get("value", "")

        makers_list = raw.get("maker") or []
        makers = [
            m.get("summary", {}).get("title", "")
            for m in makers_list
        ]

        obj = DesignObject(
            name=raw.get("summary", {}).get("title", ""),
            year=year,
            classification=classification,
            dimension=dimension,
            makers=makers,
            image_urls=urls,
            country="USA" if "USA" in country else "Canada",
            source="https://apidocs.cooperhewitt.org/"
        )
        # print(obj)
        results.append(obj)

    return results



def save_design_objects_to_xlsx(objects: List[DesignObject], delimiter: str = "|||"):
    # Define the relative path to the target folder
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    os.makedirs(data_dir, exist_ok=True)

    # Use current Python script name to create output file
    current_script = os.path.basename(inspect.stack()[-1].filename)
    base_filename = os.path.splitext(current_script)[0]
    output_path = os.path.join(data_dir, f"{base_filename}.xlsx")

    # Prepare rows
    rows = []
    for obj in objects:
        rows.append({
            "name": obj.name,
            "year": obj.year,
            "classification": obj.classification,
            "dimension": obj.dimension,
            "makers": delimiter.join(obj.makers),
            "image_urls": delimiter.join(obj.image_urls),
            "country": obj.country,
            "price": obj.price or "",
            "popularity": obj.popularity or "",
            "source": obj.source or "",
        })

    # Write to Excel
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    print(f"✅ Saved {len(rows)} design objects to {output_path}")

if __name__ == '__main__':
        # Countries to consider for Cooper Hewitt
        AMERICA_CANADA_COUNTRIES = [
            "USA",
            "U.S.A.",
            "USA (silver)",
            "USA or England",
            "USA or Europe",
            "United States",
            "Puerto Rico",
            "possibly USA",
            "probably USA",
            "Canada",
        ]
        department = "Product Design and Decorative Arts"
        yearRange = range(1960, 2010)
        size = 100
        page = 0

        client = create_client()
        all_objects = []

        total_count = 0

        for year in yearRange:
            year_count = 0

            for country in AMERICA_CANADA_COUNTRIES:
                results = fetch_design_objects(client, department, year, country, size, page)
                count = len(results)
                year_count += count
                total_count += count
                all_objects.extend(results)

            print(f"Year: {year}, Found: {year_count}")

        print(f"\nTotal objects found: {total_count}")
        save_design_objects_to_xlsx(all_objects)