from typing import Optional, List
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from src.datacollection.design_object_model import DesignObject

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
        results.append(obj)

    return results
