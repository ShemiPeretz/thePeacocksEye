from pydantic import BaseModel
from typing import Optional , Dict
from enum import Enum
from typing import Union

class WeatherSummary(BaseModel):
    station: int


