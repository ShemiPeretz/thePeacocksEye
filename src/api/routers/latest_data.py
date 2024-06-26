
from src.api.IMS_getters.raw_data import RawDataGetter
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

@router.get("/latetst_data/")
async def get_latest_data():
    getter = RawDataGetter()

    request = {"request": "time_period",
               "data": {"time_period": "latest"}
               }
    stations_meta = get_stations()

    latest_data = {
        (
        station_meta["stationId"], station_meta["name"]):
        getter.get_channels(stations_id=station_meta["stationId"], channel_id="", request=request
                            )
        for station_meta in stations_meta
    }

    return JSONResponse(content=latest_data)
def get_stations():
    getter = RawDataGetter()
    stations = getter.get_stations()
    stations_meta = [getter.pars_stations_meta(station) for station in stations]
    return stations_meta