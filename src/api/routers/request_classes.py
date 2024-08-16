from pydantic import BaseModel
from typing import Optional, Dict, List, Union
from enum import Enum
from datetime import datetime


class WeatherSummary(BaseModel):
    station: int


class Hours(str, Enum):
    six_am = "06:00"
    six_pm = "18:00"
    twenty_am = "00:00"
    twelve_pm = "12:00"


class Range(BaseModel):
    starting_from: Union[int, Hours]
    ending_at: Union[int, Hours]


class TimeInterval(BaseModel):
    startTime: datetime = None
    endTime: datetime = None
    interval: str = "day"
    # year: Optional[Range] = None
    # month: Optional[Range] = None
    # day: Optional[Range] = None
    # hour: Optional[Range] = None


class Dataset(str, Enum):
    daily_rain = "daily_rain"
    monty_rain = "monthly_rain"
    yearly_rain = "yearly_rain"
    hourly = "hourly"
    daily = "daily"
    radiation = "radiation"


class GraphMeta(BaseModel):
    graphType: str
    graphSizeX: int
    graphSizeY: int
    station: List[int]
    isTime: bool
    channelX: str
    channelNameX: str
    channelsY: List[str]
    channelNamesY: List[str]
    dataset: Dataset
    timeInterval: TimeInterval
    cumulative: bool
