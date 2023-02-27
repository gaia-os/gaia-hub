from pydantic import BaseModel
from typing import Dict, Any
from geojson_pydantic import Polygon


class Lot(BaseModel):
    name: str
    bounds: Polygon
    strategy: str
    geo_params: Dict[str, Any]
