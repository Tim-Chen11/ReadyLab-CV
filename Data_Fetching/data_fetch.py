from typing import Optional, List
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from .design_object_model import DesignObject

BASE_IMG_URL = "https://images.cooperhewitt.org/"
transport = RequestsHTTPTransport(url="https://api.cooperhewitt.org/")
client = Client(transport=transport, fetch_schema_from_transport=True)

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
# QUERY = gql("""
# query GetObjects(
#   $type: String!,
#   $from: Int!,
#   $to:   Int!,
#   $size: Int! = 10,
#   $page: Int! = 0
# ) {
#   object(
#     name: $type,
#     yearRange: { from: $from, to: $to },
#     hasImages: true,
#     size: $size,
#     page: $page
#   ) {
#     summary
#     date
#     classification
#     measurements
#     maker          { summary }       
#     legal
#     multimedia
#   }
# }
# """)


# def fetch_design_objects(
#     object_type: str,
#     year_from:    int,
#     year_to:      int,
#     size:         int = 10,
#     page:         int = 0
# ) -> List[DesignObject]:
#     variables = {
#         "type": object_type,
#         "from": year_from,
#         "to":   year_to,
#         "size": size,
#         "page": page
#     }
#     resp  = client.execute(QUERY, variable_values=variables)
#     items = resp["object"]
#     results: List[DesignObject] = []

#     for itm in items:
#         # extract core fields
#         title        = itm["summary"]["title"]
#         year         = int(itm["date"][0]["value"]) if itm.get("date") else None
#         product_type = itm["classification"][0]["summary"]["title"] if itm.get("classification") else None
#         dimensions   = itm["measurements"]["dimensions"][0]["value"] if itm.get("measurements") else None
#         manufacturer = itm["maker"][0]["summary"]["title"] if itm.get("maker") else None
#         source       = itm.get("legal", {}).get("credit")

#         # build full image URL
#         img_url = None
#         if itm.get("multimedia"):
#             loc = itm["multimedia"][0]["preview"].get("location")
#             if loc:
#                 img_url = BASE_IMG_URL + loc

#         results.append(DesignObject(
#             title        = title,
#             year         = year,
#             product_type = product_type,
#             dimensions   = dimensions,
#             manufacturer = manufacturer,
#             designer     = None,
#             price        = None,
#             popularity   = None,
#             source       = source,
#             image_url    = img_url
#         ))

#     return results

# # Fetch one chair from 1960
# one_chair = fetch_design_objects("chair", 1960, 1960, size=1, page=0)
# print(one_chair)

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport



# ✅ Query using yearRange correctly
QUERY = gql("""
query GetObjects(
  $type: String!,
  $yearRange: YearRangeInput!,
  $size: Int!,
  $page: Int!
) {
  object(
    name: $type,
    yearRange: $yearRange,
    hasImages: true,
    size: $size,
    page: $page
  ) {
    summary
    date
    classification
    measurements
    maker { summary }
    legal
    multimedia
  }
}
""")

# ✅ Pass yearRange as a variable object
variables = {
    "type": "chair",
    "yearRange": {"from": 1960, "to": 1960},
    "size": 1,
    "page": 0
}

# Execute
result = client.execute(QUERY, variable_values=variables)
print(result)

