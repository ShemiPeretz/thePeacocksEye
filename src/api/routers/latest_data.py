import logging
from src.api.IMS_getters.raw_data import RawDataGetter
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
load_dotenv()
router = APIRouter()

defult_station_ids_ims = [2, 6, 8, 10, 11, 13, 16, 17, 18, 20,
                          21, 22, 23, 24, 25, 26, 28, 29, 30, 32,
                          33, 35, 36, 41, 42, 43, 44, 45, 46, 54,
                          58, 59, 60, 62, 64, 65, 67, 69, 73, 74,
                          75, 77, 78, 79, 82, 85, 90, 98, 99, 106,
                          107, 112, 115, 121, 123, 124, 178, 186, 188, 202,
                          205, 206, 207, 208, 210, 211, 212, 218, 224, 227,
                          228, 232, 233, 236, 238, 239, 240, 241, 242, 243,
                          244, 245, 246, 247, 248, 249, 250, 251, 252, 257,
                          259, 263, 264, 265, 269, 270, 271, 274, 275, 276,
                          277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
                          287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
                          297, 298, 299, 300, 301, 302, 303, 304, 305, 306,
                          307, 309, 310, 311, 312, 313, 314, 315, 316, 317,
                          318, 319, 320, 322, 323, 324, 325, 327, 328, 329,
                          330, 332, 333, 335, 336, 338, 343, 344, 345, 346,
                          348, 349, 350, 351, 352, 353, 354, 355, 366, 367,
                          370, 373, 379, 380, 381, 411, 412, 443, 480, 498, 499]


@router.get("/latest_data/")
async def get_latest_data():
    getter = RawDataGetter()

    request = {"request": "time_period",
               "data": {"time_period": "latest"}
               }
    stations_meta = get_stations()
    latest_data = {}
    for station_meta in stations_meta:
        channels_data = getter.get_channels(stations_id=station_meta["stationId"],
                                            channel_id="",
                                            request=request)
        if not channels_data:
            continue

        channels_data["station_name"] = station_meta["name"]
        latest_data[station_meta["stationId"]] = channels_data

    logger.info(f"Latest data on {len(latest_data.keys())} stations")
    return JSONResponse(content=latest_data)


def get_stations():
    getter = RawDataGetter()
    stations = getter.get_stations()
    if not stations:
        return [{"stationId": station_id} for station_id in defult_station_ids_ims]
    stations_meta = [getter.pars_stations_meta(station) for station in stations]
    return stations_meta
