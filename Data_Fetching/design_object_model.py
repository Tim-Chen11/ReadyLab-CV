from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DesignObject:
    name: str
    year: int
    classification: str
    dimension: str
    makers: List[str]
    image_urls: List[str]
    country: str
    price: Optional[str] = None           # Not available in API
    popularity: Optional[str] = None      # Not available in API
    source: Optional[str] = None