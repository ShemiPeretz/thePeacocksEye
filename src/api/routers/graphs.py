import logging
import os
import pprint
import traceback
from src.api.IMS_getters.raw_data import RawDataGetter
from .request_classes import GraphMeta, TimeInterval
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from datetime import datetime, timedelta
from IPython.display import HTML
from .channels_config import hourly_channels, daily_channels, non_cumulative_channels
from src.api.historical_data_gov.getters import (get_historical_data_rain, get_historical_data_daily,
                                                 get_historical_data_hourly, get_historical_data_radiation,
                                                 get_station_metadata, map_station_id)

load_dotenv()
router = APIRouter()

from fastapi import FastAPI, HTTPException
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_html

app = FastAPI()


async def generate_graph(x_data: str, t1: str, t2: str):
    data = pd.DataFrame({
        'X': [1, 2, 3, 4, 5],
        'Y': [10, 14, 18, 24, 30]
    })

    # Filter data based on date range
    data = data[(data['X'] >= int(t1)) & (data['X'] <= int(t2))]

    # Create an interactive line chart using Plotly
    fig = px.line(data, x='X', y='Y', title='Interactive Line Chart')

    # Convert the figure to JSON for easy transmission
    graph_json = fig.to_json()

    return {"graph_json": graph_json}


def create_graph(data, x_name: str, y_names: list, graph_type, x_size, y_size):
    # Create an interactive line chart using Plotly
    if graph_type == "line":
        fig = px.line(data, x=x_name, y=y_names, width=x_size, height=y_size, title='Interactive Line Chart')
    elif graph_type == "bar":
        fig = px.bar(data, x=x_name, y=y_names, width=x_size, height=y_size, title='Interactive Bar Chart')
    elif graph_type == "scatter":
        fig = px.scatter(data, x=x_name, y=y_names,width=x_size, height=y_size, title='Interactive Scatter Chart')
    graph_json = fig.to_html()

    return graph_json


def extract_data(chanel_json):
    values = [data_item['channels'][0]['value'] for data_item in chanel_json['data']]
    return values


def get_time_graph(start_time, data, graph_type, x_size, y_size):
    time_interval = timedelta(minutes=10)
    num_points = len(list(data.values())[0])
    times = [start_time + i * time_interval for i in range(num_points)]
    print(times)
    y_names = list(data.keys())
    data["Time"] = times

    fig = px.scatter(data, x='Time', y=y_names, width=x_size, height=y_size, title='Data Over time')

    # Set X-axis tick positions and labels
    hourly_ticks = pd.date_range(start=start_time, end=start_time + num_points * time_interval, freq='1D')
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=hourly_ticks, ticktext=hourly_ticks.strftime('%Y-%m-%d %H:%M:%S')))
    graph = fig.to_html(include_plotlyjs='cdn', full_html=False)

    return graph


def validate_channels(channel_list):
    return any([all([channel in hourly_channels for channel in channel_list]),
                all([channel in daily_channels for channel in channel_list])])
def get_range(start, end):
    return [i for i in range(start, end + 1)]

def map_channels_names(channel_list):
    pass
@router.post("/graphs/")
async def get_graph(request: GraphMeta):
    request = request.dict()
    interval_dict = request["timeInterval"]
    from_year, from_month, from_day = interval_dict["fromYear"], interval_dict["fromMonth"], interval_dict["fromDay"]
    to_year, to_month, to_day = interval_dict["toYear"], interval_dict["toMonth"], interval_dict["toDay"]
    year = get_range(from_year, to_year)
    month = get_range(from_month, to_month)
    day = get_range(from_day, to_day)
    stations_id = request["station"]
    y_channels = request["channelsY"]
    x_channels = request["channelX"]
    if x_channels in y_channels:
        return JSONResponse({"error": "Channels X and Y can't have overlap"},
                            status_code=400)
    channels = y_channels + [x_channels]
    is_valid = validate_channels(channels)
    if not is_valid: return JSONResponse({"error": "Channels Not exists"}, status_code=400)

    if request["hourly"]:
        df = get_historical_data_hourly(station_number=stations_id,
                                        channels=channels,
                                        year=year,
                                        month=month,
                                        day=day)
    elif request["daily"]:
        df = get_historical_data_daily(station_number=stations_id,
                                       channels=channels,
                                       year=year,
                                       month=month,
                                       day=day)
    else:
        df = None
    if datetime.today().year in year and (datetime.today().month -1 in month or datetime.today().month -2 in month):
        ims_channels_names = map_channels_names(channels)
        data = RawDataGetter.get_channels_from_db_in_range(channel_name=ims_channels_names,
                                                           station_id=stations_id,
                                                           start_date="-".join([str(from_year),
                                                                                str(from_month),
                                                                                str(from_day)]),
                                                           end_date="-".join([str(to_year),
                                                                              str(to_month),
                                                                              str(to_day)]))
        ims_df = pd.DataFrame(data, columns=channels)
        df = pd.concat([df, ims_df])


    if df is None:
        return JSONResponse({"error": "No data"})
    y_names = request["channelNamesY"]
    x_name = request["channelNameX"]
    if request["cumulative"]:
        cumulative_columns = [col for col in df.columns.tolist() if col not in non_cumulative_channels]
        df[cumulative_columns] = df[cumulative_columns].cumsum()
    graph = create_graph(data=df,
                         x_name=x_name,
                         y_names=y_names,
                         graph_type=request["graphType"],
                         x_size=request["graphSizeX"],
                         y_size=request["graphSizeY"])
    return {"graph_json": graph}
request = GraphMeta(graphType="line",
                    graphSizeX=1000,
                    graphSizeY=1000,
                    station=9111,
                    isTime=False,
                    channelX="time_obs",
                    channelNameX="time_obs",
                    channelsY=["tmp_air_wet"],
                    channelNamesY=["tmp_air_wet"],
                    timeInterval=TimeInterval(fromYear=2023,
                                              fromMonth=3,
                                              fromDay=1,
                                              toYear=2023,
                                              toMonth=3,
                                              toDay=20),
                    hourly=True,
                    daily=False,
                    cumulative=True)
# import asyncio
# import webview
# def display_plot(html_content):
#     webview.create_window('Plot', html=html_content)
#     webview.start()
#
# graph = asyncio.run(get_graph(request))
#


