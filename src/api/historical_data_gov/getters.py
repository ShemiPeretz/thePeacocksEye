import json
import requests
import logging
import traceback
import pandas as pd
import datetime
from typing import List

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_YEAR = [datetime.datetime.now().year]
DEFAULT_MONTH = [datetime.datetime.now().month - 2]


def generate_api_url(resource_id,
                     year: List[int],
                     month: List[int],
                     day: List[int],
                     station_number: List[int]) -> str:
    url = (('https://data.gov.il/api/action/datastore_search?resource_id={resource_id}&'
            'filters={{"year":{year},"month":{month},"day":{day},"stn_num":{station_number}}}&limit=10000000000').
           format(resource_id=resource_id, year=year, month=month, day=day, station_number=station_number))
    return url


def get_historical_data(resource_id: str,
                        station_number: List[int],
                        year: List[int],
                        month: List[int],
                        day: List[int]
                        ) -> pd.DataFrame:
    """
    Get historical data aggregated per hour
    :return: dict with historical data
    """
    try:
        # TODO fix month defult for case the year changed and prev month is 12
        url = generate_api_url(resource_id=resource_id,
                               year=year,
                               month=month,
                               day=day,
                               station_number=station_number)
        response = requests.get(url)
        data = json.loads(response.text)["result"]["records"]
        df = pd.DataFrame(data)
        return df
    except Exception as err:
        logging.error(f"Unexpected error occurred while getting channels: %s", err)
        logging.error(traceback.format_exc())


def get_historical_data_daily(station_number: List[int],
                              channels: List[str],
                              year: List[int],
                              month: List[int],
                              day: List[int]) -> pd.DataFrame:
    """
    Get historical data aggregated per day
    :return: dict with historical data
    """
    resource_id = "cee3ad4a-4e77-4015-8245-52505417d7ea"
    data = get_historical_data(resource_id=resource_id,
                               year=year,
                               month=month,
                               day=day,
                               station_number=station_number)
    if data.shape[0] == 0:
        return
    if len(channels) == 0:
        channels = list(data.columns)
    return data[channels]

def get_historical_data_hourly(station_number: List[int],
                               channels: List[str],
                               year: List[int],
                               month: List[int],
                               day: List[int]):
    resource_id = "c02e5c7d-0adb-4e04-a941-06e281180294"
    data = get_historical_data(resource_id=resource_id,
                               year=year,
                               month=month,
                               day=day,
                               station_number=station_number)
    if data.shape[0] == 0:
        return
    if len(channels) == 0:
        channels = list(data.columns)
    return data[channels]


def get_historical_data_radiation(station_number: List[int],
                                  year: List[int],
                                  month: List[int],
                                  day: List[int]) -> pd.DataFrame:
    """
    Get historical data for daily radiation
    :return: dict with historical data
    """
    resource_id = "219bc8fd-43a7-4c6e-9882-bdc4cdd8c8b0"
    data = get_historical_data(resource_id=resource_id,
                               year=year,
                               month=month,
                               day=day,
                               station_number=station_number)
    return data

def get_historical_data_rain(station_number: List[int],
                             year: List[int],
                             month: List[int],
                             day: List[int]) -> pd.DataFrame:
    resource_id = "e80b470f-fcbc-4987-a685-d4fbefbd75d1"
    data = get_historical_data(resource_id=resource_id,
                               year=year,
                               month=month,
                               day=day,
                               station_number=station_number)
    return data

def get_station_metadata() -> pd.DataFrame:
    try:
        url = 'https://data.gov.il/api/3/action/datastore_search?resource_id=83841660-b9c4-4ecc-a403-d435b3e8c92f'
        response = requests.get(url)
        data = json.loads(response.text)["result"]["records"]
        with open("station_data.json", "w") as outfile:
            json.dump(data, outfile)
        df = pd.DataFrame(data)
        df = df[df['date_close'].isna() & ~df['stn_num_env'].isna()] # don't want close stations.
        return df[["stn_num", "stn_name", "stn_name_heb","stn_num_env", "stn_type"]]
    except Exception as err:
        logging.error(f"Unexpected error occurred while getting channels: %s", err)
        logging.error(traceback.format_exc())

def map_station_id(station_id):
    station_meta = get_station_metadata()
    return list(station_meta[station_meta["stn_num_env"].isin(station_id)]["stn_num"].unique())

df = get_historical_data_hourly(year=[2024], month=[3,4,5], day=[29,28],station_number=map_station_id([60, 411]),channels=[])
print(df)
