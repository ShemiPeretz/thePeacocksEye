from src.api.IMS_getters.raw_data import RawDataGetter
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

@router.get("/active-cities/")
async def get_active_cities():
    getter = RawDataGetter()
    cities_metadata = getter.get_stations()
    active_cities = {meta["stationId"]: meta["name"] for meta in cities_metadata
                     if
                     meta["active"]
                     and
                     any(m["active"] and m["name"] == "TD" for m in meta["monitors"])}
    return JSONResponse(content=active_cities)


@router.get("/get-active-cities-weather/")
async def get_active_cities_weather():
    getter = RawDataGetter()

    request = {"request": "time_period",
               "data": {"time_period": "latest"}
               }
    stations = getter.get_stations_with_retry()
    allowed_stations = [
      {"stationId": 411,"name": "BEER SHEVA BGU"},
      {"stationId": 178, "name": "TEL AVIV"},
      {"stationId": 23, "name": "JERUSALEM"},
      {"stationId": 42, "name": "HAIFA"},
      {"stationId": 124, "name": "ASHDOD"},
      {"stationId": 208, "name": "ASHKELON"},
      {"stationId": 10, "name": "MAROM GOLAN"},
      {"stationId": 54, "name": "BEIT DAGAN"},
      {"stationId": 64, "name": "EILAT"}
    ]
    allowed_stations_ids = [411,
                            178,
                            23,
                            42,
                            124,
                            208,
                            10,
                            54,
                            64]
    stations = [station for station in stations if station["stationId"] in allowed_stations_ids]
    stations_meta = [getter.pars_stations_meta(station) for station in stations]

    latest_data = {
        station_meta["stationId"]: {
            "name": station_meta["name"],
            "location": station_meta["location"],
            "data": getter.get_channels_with_retry(
                stations_id=station_meta["stationId"],
                channel_id="",
                request=request
            )
        }
        for station_meta in stations_meta
    }

    return JSONResponse(content={"data": latest_data})





