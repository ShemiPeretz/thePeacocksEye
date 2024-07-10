import logging
from .request_classes import WeatherSummary
from src.api.IMS_getters.raw_data import RawDataGetter
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()
logger = logging.getLogger(__name__)
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
async def get_weather_summery(request: WeatherSummary):
    station_id = request.dict()["station"]
    request = {"request": "time_period",
            "data": {"time_period": "latest"}
               }
    getter = RawDataGetter()
    channels_names = {"IMS": ["TD", "WS", "WD", "RH", "Grad", "TDmax", "TDmin", "Rain", "BP"],
                      "DB": ["temperature_dry", "wind_speed", "wind_direction",
                             "relative_humidity", "radiation_global", "temperature_max", "temperature_min", "rain", "pressure"]
                      }

    try:
        raw_data = getter.get_channels_from_db(station_id=station_id,channel_name=channels_names["DB"])
        return JSONResponse(content={channel_name: max(value, 0)
                                     for channel_name, value in zip(channels_names["DB"], raw_data)})
    except Exception as e:
        logger.warning(f"Got exception while getting station id {station_id} from DB. Exception: {e}")
        pass

    raw_data = getter.get_channels(stations_id=station_id, channel_id="", request=request)
    if raw_data is None:
        return None
    raw_data = raw_data["data"][0]
    channels = raw_data["channels"]
    data = {}
    for channel in channels:
        channel_name = channel["name"]
        if channel_name in channels_names["IMS"]:
            index = channels_names["IMS"].index(channel_name)
            db_mapped_name = channels_names["DB"][index]
            data[db_mapped_name] = channel["value"]
    return JSONResponse(content=data)