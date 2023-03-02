from jax.numpy import DeviceArray
from typing import Any, Union, List
from pydantic import BaseModel


class Observation(BaseModel):
    name: str
    lot_name: str
    value: Union[Any, List[Any]]

    class Config:
        json_encoders = {
            DeviceArray: lambda v: v.tolist()
        }
