import datetime
import logging
import uuid
import os
from .request_classes import GraphMeta, TimeInterval, Range, Hours, Dataset
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from .channels_config import (hourly_channels, daily_channels, rain_channels_daily, rain_channels_yearly,
                             radiation_channels,rain_channels_monthly, non_cumulative_channels, dataset_resource_id_map)
from src.api.historical_data_gov.getters import get_historical_data

load_dotenv()
router = APIRouter()

from fastapi import FastAPI, HTTPException
import pandas as pd
import plotly.express as px

app = FastAPI()

HOURLY_RESOURCE_ID = os.getenv('HOURLY_RESOURCE_ID')
DAILY_RESOURCE_ID = os.getenv('DAILY_RESOURCE_ID')
RADIATION_RESOURCE_ID = os.getenv("RADIATION_RESOURCE_ID")
RAIN_HOURLY_RESOURCE_ID = os.getenv('RAIN_HOURLY_RESOURCE_ID')
RAIN_MONTHLY_RESOURCE_ID = os.getenv('RAIN_MONTHLY_RESOURCE_ID')
RAIN_YEARLY_RESOURCE_ID = os.getenv("RAIN_YEARLY_RESOURCE_ID")


def create_graph(data, x_name: str, y_names: list, x_channel: str, y_channels: list, graph_type, x_size, y_size):
    labels = {channel: channel_name for channel_name, channel in zip(y_names, y_channels)}
    labels[x_channel] = x_name
    if graph_type == "line":
        fig = px.line(data, x=x_channel, y=y_channels, width=x_size, height=y_size,
                      title='Interactive Line Chart', labels=labels)
    elif graph_type == "bar":
        fig = px.bar(data, x=x_name, y=y_names, width=x_size, height=y_size,
                     title='Interactive Bar Chart', labels=labels)
    elif graph_type == "scatter":
        fig = px.scatter(data, x=x_name, y=y_names, width=x_size, height=y_size,
                         title='Interactive Scatter Chart', labels=labels)
    # graph_json = fig.to_html()
    # graph_json = fig.to_json()
    graph_id = str(uuid.uuid4())
    fig.write_html(f"./graphs/{graph_type}_{graph_id}.html")
    return graph_id


def extract_data(chanel_json):
    values = [data_item['channels'][0]['value'] for data_item in chanel_json['data']]
    return values


def get_time_graph(start_time, data, graph_type, x_size, y_size):
    time_interval = timedelta(minutes=180)
    num_points = len(list(data.values())[0])
    times = [start_time + i * time_interval for i in range(num_points)]
    print(times)
    y_names = list(data.keys())
    data["Time"] = times
    if graph_type == "scatter":
        fig = px.scatter(data, x='Time', y=y_names, width=x_size, height=y_size, title='Data Over time')

    # Set X-axis tick positions and labels
    hourly_ticks = pd.date_range(start=start_time, end=start_time + num_points * time_interval, freq='1D')
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=hourly_ticks, ticktext=hourly_ticks.strftime('%Y-%m-%d %H:%M:%S')))
    graph = fig.to_html(include_plotlyjs='cdn', full_html=False)

    return graph


def validate_channels(channel_list):

    return any([all([channel in channel_config]) for channel_config in [hourly_channels, daily_channels,
                                                                        rain_channels_daily, rain_channels_monthly,
                                                                        rain_channels_yearly, radiation_channels]
                for channel in channel_list])



def get_range(start, end):
    if start and end:
        start, end = min(start, end), max(start, end)
        return [i for i in range(start, end + 1)]
    return None


def get_date_edges(interval_dict):
    intervals = {k: v for k, v in interval_dict.items() if v}
    start_date_dict = {}
    end_date_dict = {}
    for k, v in intervals.items():
        start_date_dict[k] = v["starting_from"]
        end_date_dict[k] = v["ending_at"]
    return start_date_dict, end_date_dict


def generate_filters(interval_dict, station_id):
    base_filter = {"stn_num": station_id}
    interval_filter = {k: v for k, v in interval_dict.items() if v}
    for key, value in interval_filter.items():
        if key != "hour":
            base_filter[key] = get_range(start=value["starting_from"], end=value["ending_at"])
    return base_filter


def transform_data(df, channels, interval_dict, cumulative):
    # filter data by datetime range
    start, end = get_date_edges(interval_dict=interval_dict)
    start_query = " & ".join([f"{key}>={value}" for key, value in start.items()])
    end_query = " & ".join([f"{key}<={value}" for key, value in end.items()])
    query = f"{start_query} or {end_query}"
    df = df.query(query)
    if len(channels) == 0:
        channels = df.columns.tolist()
    df = df[channels]
    if cumulative:
        cumulative_columns = [col for col in df.columns.tolist() if col not in non_cumulative_channels]
        df[cumulative_columns] = df[cumulative_columns].cumsum()

    return df

def remove_duplicate_columns(df):
    for column in df.columns:
        if column.endswith('_left'):
            original_column = column[:-5]  # Remove '_left' suffix
            if original_column + '_right' in df.columns:
                # Resolve duplicate by choosing one of the columns
                df[original_column] = df[column]
                # Drop both original columns
                df.drop(columns=[column, original_column + '_right'], inplace=True)
    return df
def join_dataframes(df_list):
    df_list = [df for df in df_list if not df.empty]
    if len(df_list) == 0:
        return pd.DataFrame()
    if len(df_list) == 1:
        return df_list[0]
    if len(df_list) == 2:
        df1 = df_list[0]
        df2 = df_list[1]
        df = pd.merge(df1, df2, how="inner", on="time_obs", suffixes=('_left', '_right'))
        return remove_duplicate_columns(df)
    else:
        df1 = df_list[0]
        df2 = df_list[1]
        df = pd.merge(df1, df2, how="inner", on="time_obs", suffixes=('_left', '_right'))
        df = remove_duplicate_columns(df)
        df_list = [df_list[i] for i in range(2, len(df_list))]
        df_list.append(df)
        return join_dataframes(df_list)


def map_channels_names(channel_list):
    pass


@router.post("/graphs/")
async def get_graph(request: GraphMeta):
    request = request.dict()
    time_interval = request["timeInterval"]
    # Create a new TimeInterval object
    new_time_interval = TimeInterval(
        startTime=time_interval["startTime"],
        endTime=time_interval["endTime"]
    )
    interval_dict = get_range_from_interval(new_time_interval.startTime, new_time_interval.endTime)
    stations_id = request["station"]
    filters = generate_filters(interval_dict=interval_dict, station_id=stations_id)
    y_channels = request["channelsY"]
    x_channels = request["channelX"]
    dataset = request["dataset"]

    if x_channels in y_channels:
        return JSONResponse({"error": "Channels X and Y can't have overlap"},
                            status_code=400)
    channels = y_channels + [x_channels]
    is_valid = validate_channels(channels)
    if not is_valid:
        return JSONResponse({"error": "Channels Not exists"}, status_code=400)

    df_list = [get_historical_data(resource_id=dataset_resource_id_map[dataset],
                                   filters=filters
                                   )]

    df = join_dataframes(df_list)

    if df.empty:
        return JSONResponse({"error": "No data"})
    df = transform_data(df=df, channels=channels, interval_dict=interval_dict, cumulative=request["cumulative"])

    graph = create_graph(data=df,
                         x_channel=x_channels,
                         y_channels=y_channels,
                         x_name=request["channelNameX"],
                         y_names=request["channelNamesY"],
                         graph_type=request["graphType"],
                         x_size=request["graphSizeX"],
                         y_size=request["graphSizeY"])
    return {"http_graph": graph}

def get_range_from_interval(start: datetime, end: datetime):
    diff = relativedelta(end, start)
    return {
        "year": {"starting_from": start.year, "ending_at": end.year},
        "month": {"starting_from": start.month, "ending_at": end.month},
        "day": {"starting_from": start.day, "ending_at": end.day},
        "hour": {"starting_from": start.hour, "ending_at": end.hour}
    }

request = GraphMeta(
    graphType="line",
    graphSizeX=800,
    graphSizeY=600,
    station=[241260, 190],
    isTime=True,
    channelX="time_obs",
    channelNameX="Time",
    channelsY=["rain_06_next", "tmp_air_max"],
    channelNamesY=["Rainfall", "Temperature_Air_Max"],
    dataset= Dataset.daily,
    timeInterval=TimeInterval(
        year=Range(starting_from=2020, ending_at=2023),
        month=Range(starting_from=1, ending_at=12),
        day=Range(starting_from=1, ending_at=31)
    ),
    cumulative=False
)
# import asyncio
# import webview
#
#
# def display_plot(html_content):
#     webview.create_window('Plot', html=html_content)
#     webview.start()


# graph = asyncio.run(get_graph(request))
