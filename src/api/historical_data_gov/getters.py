import json
import requests
import logging
import traceback
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_historical_data(resource_id: str,
                        filters: dict
                        ) -> pd.DataFrame:
    """
    Get historical data aggregated per hour
    :return: dict with historical data
    """
    try:
        url = generate_url(resource_id=resource_id,
                           filters=filters)
        response = requests.get(url)
        data = json.loads(response.text)["result"]["records"]
        df = pd.DataFrame(data)
        return df
    except Exception as err:
        logging.error(f"Unexpected error occurred while getting channels: %s", err)
        logging.error(traceback.format_exc())
        return pd.DataFrame()


def get_station_metadata() -> pd.DataFrame:
    try:
        url = 'https://data.gov.il/api/3/action/datastore_search?resource_id=83841660-b9c4-4ecc-a403-d435b3e8c92f'
        response = requests.get(url)
        data = json.loads(response.text)["result"]["records"]
        # with open("station_data.json", "w") as outfile:
        #     json.dump(data, outfile)
        df = pd.DataFrame(data)
        # df = df[df['date_close'].isna()]  # don't want close stations.
        # return df[["stn_num", "stn_name", "stn_name_heb", "stn_num_env", "stn_type"]]
        return df
    except Exception as err:
        logging.error(f"Unexpected error occurred while getting channels: %s", err)
        logging.error(traceback.format_exc())


def map_station_id(station_id):
    station_meta = get_station_metadata()
    return list(station_meta[station_meta["stn_num_env"].isin(station_id)]["stn_num"].unique())


def generate_url(resource_id: str, filters: dict) -> str:
    base_url = 'https://data.gov.il/api/action/datastore_search?resource_id={resource_id}&'

    # Constructing the filters part of the URL
    filters_str = ','.join([f'"{key}":{value}' for key, value in filters.items()])
    filters_url_part = f'filters={{{filters_str}}}&limit=10000000000'

    # Combining everything into the final URL
    final_url = base_url.format(resource_id=resource_id) + filters_url_part
    return final_url


station_metadata = get_station_metadata()
rain_stations = station_metadata[station_metadata["stn_type"] == 2][
    ["stn_num", "stn_name", "stn_name_heb", "date_web_frst", "date_web_last"]].rename(
    columns={"stn_num": "station_number",
             "stn_name": "station_name_english",
             "station_name_heb": "station_name_hebrew",
             "date_web_frst": "first_data_date",
             "date_web_last": "last_data_date"})
radiation_stations = station_metadata[station_metadata["stn_type"] == 4].rename(
    columns={"stn_num": "station_number",
             "stn_name": "station_name_english",
             "station_name_heb": "station_name_hebrew",
             "date_web_frst": "first_data_date",
             "date_web_last": "last_data_date"})
weather_stations = station_metadata[station_metadata["stn_type"] == 1].rename(
    columns={"stn_num": "station_number",
             "stn_name": "station_name_english",
             "station_name_heb": "station_name_hebrew",
             "date_web_frst": "first_data_date",
             "date_web_last": "last_data_date"})

rain_stations.to_csv("rain_stations.csv", index=False)
weather_stations.to_csv("weather_stations.csv", index=False)
radiation_stations.to_csv("radiation_stations.csv", index=False)