from pydantic import BaseModel
from typing import Dict, List
from .Objective import Objective


class Strategy(BaseModel):
    name: str
    species: List[str]
    interventions: Dict[str, str]
    objective: Objective
    policy: List[List[str]]
