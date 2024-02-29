import logging
import os
import traceback
from .request_classes import WeatherSummary
from src.api.IMS_getters.raw_data import RawDataGetter
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

def get_channels_per_station(station_id, channels):
    getter = RawDataGetter()
    station_metadata = getter.get_stations(station_id=station_id)
    monitors = station_metadata["monitors"]
    station_channel_map = {}
    for monitor in monitors:
        channel_name = monitor["name"]
        channel_id = monitor["channelId"]
        if channel_name in channels:
            station_channel_map[channel_name] = channel_id
    return station_channel_map
@router.post("/weather_summery/")
async def get_evaluation_status(request: WeatherSummary):
    station_id = request.dict()["station"]
    request = {"request": "time_period",
            "data": {"time_period": "latest"}
               }
    getter = RawDataGetter()
    channels_names = ["TD", "WS", "WD", "RH","Grad"]
    station_channel_map = get_channels_per_station(station_id=station_id, channels=channels_names)
    data = {}
    for channel_name, channel_id in station_channel_map.items():
        raw_data = getter.get_channels(stations_id=station_id,channel_id=channel_id,request=request)
        filtered_data = raw_data["data"][0]["channels"][0]["value"]
        data[channel_name] = filtered_data
    return JSONResponse(content=data)




