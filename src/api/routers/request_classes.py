from pydantic import BaseModel
from typing import Optional , Dict
from enum import Enum
from typing import Union

class WeatherSummary(BaseModel):
    station: int


class TimeInterval(BaseModel):
    fromYear: int
    fromMonth: int
    fromDay: int
    toYear: int
    toMonth: int
    toDay: int
    fromYear: int

class GraphMeta(BaseModel):
    graphType: str
    graphSizeX: int
    graphSizeY: int
    region: int
    station: int
    isTime: bool
    channelX: int
    channelNamex: str
    channelsY: list[int]
    channelNamesY: list[str]
    timeInterval: TimeInterval


