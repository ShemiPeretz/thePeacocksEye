from pydantic import BaseModel
from typing import Optional, Dict, List
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

class GraphMeta(BaseModel):
    graphType: str
    graphSizeX: int
    graphSizeY: int
    station: int
    isTime: bool
    channelX: str
    channelNameX: str
    channelsY: List[str]
    channelNamesY: List[str]
    timeInterval: TimeInterval
    hourly: bool
    daily: bool
    cumulative: bool
