from dataclasses import dataclass
from typing import Optional

@dataclass
class DesignObject:
    title: str
    year: Optional[int] = None
    product_type: Optional[str] = None
    dimensions: Optional[str] = None
    manufacturer: Optional[str] = None
    designer: Optional[str] = None
    price: Optional[str] = None           # Not available in API
    popularity: Optional[str] = None      # Not available in API
    source: Optional[str] = None
    image_url: Optional[str] = None
