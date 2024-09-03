import datetime
import json
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
                             radiation_channels,rain_channels_monthly, non_cumulative_channels, dataset_resource_id_map,
                             dataset_filters)
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
    # Check if stn_num is in y_channels
    if 'stn_num' in y_channels:
        if graph_type == 'line':
        # Melt the DataFrame to long format for plotting
            melted_data = data.melt(id_vars=[x_channel, 'stn_num'],
                                    value_vars=[col for col in y_channels if col != 'stn_num'],
                                    var_name='channel', value_name='value')
            # Plot the data with color based on stn_num and line_dash based on the channel
            fig = px.line(melted_data, x=x_channel, y='value', color='stn_num', line_dash='channel',
                          title='Interactive Line Chart',
                          labels=labels,
                          width=x_size, height=y_size)

            # Update trace names in the legend
            fig.for_each_trace(lambda t: t.update(name=f"Station: {t.name.split(',')[0].strip().split('.')[0]} Channel: {labels[t.name.split(',')[-1].strip()]}"))
        else:
            return JSONResponse({"error": "Station comparison only in line plot"},
                            status_code=400)
    else:
        if len(y_channels) == 1:
            y_channels = y_channels[0]
        if graph_type == "line":
            fig = px.line(data, x=x_channel, y=y_channels, width=x_size, height=y_size,
                          title='Interactive Line Chart', labels=labels)
        elif graph_type == "bar":
            fig = px.bar(data, x=x_channel, y=y_channels, width=x_size, height=y_size,
                         title='Interactive Bar Chart', labels=labels)
        elif graph_type == "scatter":
            fig = px.scatter(data, x=x_channel, y=y_channels, width=x_size, height=y_size,
                             title='Interactive Scatter Chart', labels=labels)
        for trace in fig.data:
            trace.name = labels.get(trace.name, trace.name)
    font_size = 12 # Adjust this value for overall font size
    axis_title_font_size = 14
    legend_font_size = 6
    tick_font_size = 10
    fig.update_layout(
        font=dict(size=font_size),  # Overall font size
        title_font=dict(size=font_size + 2),  # Title font size
        xaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_font_size)),
        yaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_font_size)),
        legend=dict(font=dict(size=legend_font_size))
    )
    graph_json = fig.to_json()
    graph_id = str(uuid.uuid4())
    # with open(graph_id + '.json', 'w') as f:
    #     json.dump(graph_json, f)
    # fig.show()
    return graph_json


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
    channel_configs = [hourly_channels, daily_channels, rain_channels_daily, rain_channels_monthly, rain_channels_yearly,
              radiation_channels]
    for channel_config in channel_configs:
        match = all([channel in channel_config for channel in channel_list])
        if match:
            return True
    return False

def get_date_edges(interval_dict):
    intervals = {k: v for k, v in interval_dict.items() if v}
    start_date_dict = {}
    end_date_dict = {}
    for k, v in intervals.items():
        start_date_dict[k] = v["starting_from"]
        end_date_dict[k] = v["ending_at"]
    return start_date_dict, end_date_dict


def generate_filters(start_date, end_date, station_id, dataset_filters):
    base_filter = {"stn_num": station_id}
    current_date = start_date
    years = set()
    months = set()
    days = set()
    while current_date <= end_date:
        years.add(current_date.year)
        months.add(current_date.month)
        days.add(current_date.day)
        current_date += timedelta(days=1)
    candidate_filters = {"year": years, "month": months, "day": days, "time_obs": years}
    date_filter = {k: v for k, v in candidate_filters.items() if k in dataset_filters}
    base_filter.update(date_filter)
    return base_filter


def transform_data(df, channels, start_date, end_date, cumulative, dataset_type):
    # filter data by datetime range
    if dataset_type == "yearly_rain":
        start_time = start_date.year
        end_time = end_date.year
        df["time_obs"].astype(int)
    elif dataset_type == "monthly_rain":
        start_time = f"{start_date.year}-{start_date.month}"
        end_time = f"{end_date.year}-{end_date.month}"
        df['time_obs'] = pd.to_datetime(df["time_obs"], format="YYYY-MM")
    else:
        start_time = f"{start_date.year}-{start_date.month}-{start_date.day}"
        end_time = f"{end_date.year}-{end_date.month}-{end_date.day}"
        df['time_obs'] = pd.to_datetime(df["time_obs"])

    # start, end = get_date_edges(interval_dict=interval_dict)
    # start_query = " & ".join([f"{key}>={value}" for key, value in start.items()])
    # end_query = " & ".join([f"{key}<={value}" for key, value in end.items()])
    # query = f"({start_query}) and ({end_query})"
    # df = df.query(query)
    df = df[(df['time_obs'] >= start_time) & (df['time_obs'] <= end_time)]
    if len(channels) == 0:
        channels = df.columns.tolist()
    df = df[channels]
    channels_to_float = [c for c in channels if c != "time_obs"]
    df[channels_to_float] = df[channels_to_float].astype("float64")
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
    print(request)
    time_interval = request["timeInterval"]
    # Create a new TimeInterval object
    new_time_interval = TimeInterval(
        startTime=time_interval["startTime"],
        endTime=time_interval["endTime"]
    )
    stations_id = request["station"]
    dataset = request["dataset"]
    filters = generate_filters(start_date=new_time_interval.startTime,
                               end_date=new_time_interval.endTime,
                               station_id=stations_id,
                               dataset_filters=dataset_filters[dataset])
    y_channels = request["channelsY"]
    x_channels = request["channelX"]

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
    df = transform_data(df=df,
                        channels=channels,
                        start_date=new_time_interval.startTime,
                        end_date=new_time_interval.endTime,
                        cumulative=request["cumulative"],
                        dataset_type=request["dataset"])

    graph = create_graph(data=df,
                         x_name=request["channelNameX"],
                         y_names=request["channelNamesY"],
                         x_channel=x_channels,
                         y_channels=y_channels,
                         graph_type=request["graphType"],
                         x_size=request["graphSizeX"],
                         y_size=request["graphSizeY"])
    return JSONResponse(content=graph)


@router.get("/get-default-graph/")
async def get_default_graph():
    # TODO: to config
    with open("C:\\Users\\shemi\\thePeacocksEyeClient\\src\\assets\\graphs\\defaultGraph.json", 'r') as f:
        graph = json.load(f)
    return JSONResponse(content=graph)


def get_range_from_interval(start: datetime, end: datetime):
    return {
        "year": {"starting_from": start.year, "ending_at": end.year},
        "month": {"starting_from": start.month, "ending_at": end.month},
        "day": {"starting_from": start.day, "ending_at": end.day},
        # "hour": {"starting_from": start.hour, "ending_at": end.hour}
    }


# request = GraphMeta(
#     graphType="line",
#     graphSizeX=10
#     graphSizeY=10
#     station=248360
#     isTime=True
#     channelX=""
#     channelNameX: str
#     channelsY: List[str]
#     channelNamesY: List[str]
#     dataset: Dataset
#     timeInterval: TimeInterval
#     cumulative: bool
#
# )

# def get_graph(request: GraphMeta):
#     request = request.dict()
#     time_interval = request["timeInterval"]
#     # Create a new TimeInterval object
#     new_time_interval = TimeInterval(
#         startTime=time_interval["startTime"],
#         endTime=time_interval["endTime"]
#     )
#     # interval_dict = get_range_from_interval(new_time_interval.startTime, new_time_interval.endTime)
#     stations_id = request["station"]
#     dataset = request["dataset"]
#     filters = generate_filters(start_date=new_time_interval.startTime,
#                                end_date=new_time_interval.endTime,
#                                station_id=stations_id,
#                                dataset_filters=dataset_filters[dataset])
#     y_channels = request["channelsY"]
#     x_channels = request["channelX"]
#
#     if x_channels in y_channels:
#         return JSONResponse({"error": "Channels X and Y can't have overlap"},
#                             status_code=400)
#     channels = y_channels + [x_channels]
#     is_valid = validate_channels(channels)
#     if not is_valid:
#         return JSONResponse({"error": "Channels Not exists"}, status_code=400)
#
#     df_list = [get_historical_data(resource_id=dataset_resource_id_map[dataset],
#                                    filters=filters
#                                    )]
#
#     df = join_dataframes(df_list)
#
#     if df.empty or df[channels].dropna().empty:
#         return JSONResponse({"error": "No data"})
#     df = transform_data(df=df,
#                         channels=channels,
#                         start_date=new_time_interval.startTime,
#                         end_date=new_time_interval.endTime,
#                         cumulative=request["cumulative"],
#                         dataset_type=dataset)
#     print("Data shape: ", df.shape)
#     graph = create_graph(data=df,
#                          x_name=request["channelNameX"],
#                          y_names=request["channelNamesY"],
#                          x_channel=x_channels,
#                          y_channels=y_channels,
#                          graph_type=request["graphType"],
#                          x_size=request["graphSizeX"],
#                          y_size=request["graphSizeY"])

request_dict = {'graphType': 'line', 'graphSizeX': 600, 'graphSizeY': 400, 'station': [3014, 4798], 'isTime': True, 'channelX': 'time_obs', 'channelNameX': 'Date', 'channelsY': ["tmp_air_min", "tmp_air_max", "stn_num"], 'channelNamesY': ['MIN_TEMP',"MAX_TEMP", "Station"], 'dataset': 'daily', 'timeInterval': {'startTime': datetime.datetime(2022, 12, 31, 0, 0, tzinfo=datetime.timezone.utc), 'endTime': datetime.datetime(2024, 12, 30, 22, 0, tzinfo=datetime.timezone.utc)}, 'cumulative': False}
request = GraphMeta(**request_dict)
get_graph(request)
