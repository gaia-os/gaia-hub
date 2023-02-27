from pydantic import BaseModel
from typing import List
from datetime import date
from .Strategy import Strategy
from .Lot import Lot


class Project(BaseModel):
    name: str
    start_date: date
    duration_in_years: int
    lots: List[Lot]
    strategies: List[Strategy]
