import logging
import os
import traceback
from src.api.IMS_getters.raw_data import RawDataGetter
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

@router.get("/active-cities/")
async def get_evaluation_status():
    getter = RawDataGetter()
    cities_metadata = getter.get_stations()
    active_cities = {meta["stationId"]: meta["name"] for meta in cities_metadata
                     if
                     meta["active"]
                     and
                     any(m["active"] and m["name"] == "TD" for m in meta["monitors"])}
    return JSONResponse(content=active_cities)




