from pydantic import BaseModel
from typing import List, Dict, Any


class Objective(BaseModel):
    target_variable: str
    aggregator: str
    constraints: List[Dict[str, Any]]
