import logging

import mysql.connector
import schedule
import time
import dotenv
import os
import datetime
from api.IMS_getters.raw_data import RawDataGetter

dotenv.load_dotenv()
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_DB = os.getenv("MYSQL_DB")

# Function to fetch data from API
def get_data():
    # Replace 'YOUR_API_ENDPOINT' with the actual API endpoint
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

    insert_data = []
    for (station_id,station_name), data in latest_data.items():
        if data is not None:
            channels = data["data"][0]["channels"]
            channels = transform_channels(channels)
            insert_data.append({"columns": [c["name"] for c in channels], "values": [c["value"] for c in channels]})
            insert_data[-1]["columns"].insert(0, "station_id")
            insert_data[-1]["columns"].insert(2, "station_name")
            insert_data[-1]["values"].insert(0, station_id)
            insert_data[-1]["values"].insert(2, station_name)
    return insert_data

def transform_channels(channels):
    channel_map = {
        'Rain': "rain",
        'WSmax': "wind_speed_max",
        'WDmax': "wind_direction_top",
        'WS': "wind_speed",
        'WD': "wind_direction",
        'STDwd': "std_wind_direction",
        'TD': "temperature_dry",
        'RH': "relative_humidity",
        'TDmax': "temperature_dry_max",
        'TDmin': "temperature_dry_min",
        'TG': "temperature_ground",
        'WS1mm': "wind_speed_max_1_min",
        'Ws10mm': "wind_speed_max_10_min",
        'Time': "measurement_date",
        'TW': "temperature_wind",
        'Grad': "radiation_global",
        'DiffR': "radiation_spread",
        'NIP': "radiation_direct",
        'BP': "pressure_at_station",
        'Rain_1_min': "rain_per_min"}
    channels_trans = []
    for channel in channels:
        if channel["name"] in channel_map.keys():
            channel["name"] = channel_map[channel["name"]]
            channels_trans.append(channel)
    return channels_trans


def get_stations():
    getter = RawDataGetter()
    stations = getter.get_stations()
    stations_meta = [getter.pars_stations_meta(station) for station in stations]
    return stations_meta
def get_connection():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB
    )
    return conn

# Function to insert data into MySQL database
def insert_data(data):
    """
    Inserts the data into the table weather_data
    :param data: list of dict with the following structure [{columns:list[str],  values: list[mix_types]}]
    :return:
    """
    try:
        # Connect to MySQL database
        conn = get_connection()
        cursor = conn.cursor()
        current_timestamp = datetime.datetime.now()
        for row in data:
            # Construct the SQL query dynamically based on available columns in the row
            columns = row["columns"]
            columns.insert(1, "timestamp")
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(columns))
            sql = f"INSERT INTO weather_data ({columns_str}) VALUES ({placeholders})"

            # Extract values from the row dictionary and execute the query
            row["values"].insert(1, current_timestamp)
            values = tuple(row["values"])
            cursor.execute(sql, values)

        # Commit changes and close connection
        conn.commit()
        conn.close()
        print("Data inserted successfully")
    except Exception as e:
        print("Error inserting data into MySQL:", e)

# Function to run the process
def run_process():
    try:
        logging.info("Fetching data from API...")
        data = get_data()
    except Exception as e:
        logging.error("Error getting data from IMS:", e)
    try:
        if data:
            insert_data(data)
        else:
            logging.warning("No data fetched from the API")
    except Exception as e:
        logging.error("Error inserting data into MySQL:", e)

if __name__ == '__main__':
    # Schedule the process to run every 10 minutes
    schedule.every(0.5).minutes.do(run_process)

    # Infinite loop to run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(1)






