from jax.numpy import DeviceArray
from geojson_pydantic import Point
from typing import Any, Union, Optional
from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import List
from uuid import UUID, uuid4
from .Observation import Observation


class Report(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    datetime: Union[datetime, date]
    location: Union[Point, List[Point]]
    project_name: Optional[str]   # Optional, to save the Ent from having to match locations
    reporter: str = ""
    provenance: str = ""
    observations: List[Observation] = []
    evidence: List[Any] = []

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        json_encoders = {
            DeviceArray: lambda v: v.tolist()
        }
