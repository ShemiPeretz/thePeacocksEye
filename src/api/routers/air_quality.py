from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import requests
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

app = FastAPI()

API_URL = "https://Air-api.sviva.gov.il/v1/envista/stations/6/data/latest"
API_TOKEN = "86764FD9-58CE-4C43-9B47-70E60C71203A"  # Replace with your actual API token


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.RequestException, ValueError))
)
def fetch_air_quality_data() -> Dict:
    headers = {
        "Authorization": 'ApiToken 86764FD9-58CE-4C43-9B47-70E60C71203A',
        "envi-data-source": "MANA"
    }
    params = {
        'ApiToken': API_TOKEN,
        "envi-data-source": "MANA"
    }

    response = requests.get(API_URL, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError("Empty response received")

    return data


def filter_channels(station_channels: List[Dict]) -> Dict:
    required_monitors = ["O3", "NO2", "PM10", "PM2.5"]
    filtered_monitors = {}
    for channel in station_channels:
        name = channel.get("name")
        if name in required_monitors:
            value = channel.get("value", None)
            filtered_monitors[name] = value
    return filtered_monitors


@app.get("/air_quality")
def get_air_quality():
    try:
        raw_data = fetch_air_quality_data()
        station_data = raw_data.get("data")[0]

        filtered_channels = filter_channels(station_data.get("channels"))
        filtered_data = {
            "station": raw_data.get("stationId"),
            "channels": filtered_channels
        }
        return JSONResponse(content=filtered_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    print(get_air_quality())
