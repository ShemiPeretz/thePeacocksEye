import logging
import os
import pprint
import traceback
from src.api.IMS_getters.raw_data import RawDataGetter
from request_classes import GraphMeta
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
load_dotenv()
router = APIRouter()

from fastapi import FastAPI, HTTPException
import pandas as pd
import plotly.express as px

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

def create_graph(data,x_name:str ,y_names:list, graph_type, x_size, y_size):

    # Create an interactive line chart using Plotly
    if graph_type == "line":
        fig = px.line(data, x=x_name, y=y_names,width=x_size, height=y_size, title='Interactive Line Chart')
    elif graph_type == "bar":
        fig = px.bar(data, x=x_name, y=y_names,width=x_size, height=y_size, title='Interactive Bar Chart')
    graph_json = fig.to_json()

    return graph_json
def extract_data(chanel_json):
    values = [data_item['channels'][0]['value'] for data_item in chanel_json['data']]
    return values



def get_time_graph(start_time,data, graph_type, x_size, y_size):

    time_interval = timedelta(minutes=10)
    num_points = len(list(data.values())[0])
    times = [start_time + i * time_interval for i in range(num_points)]
    y_names = list(data.keys())
    data["Time"] = times

    fig = px.scatter(data, x='Time', y=y_names,width=x_size, height=y_size, title='Data Over time')

    # Set X-axis tick positions and labels
    hourly_ticks = pd.date_range(start=start_time, end=start_time + num_points * time_interval, freq='1D')
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=hourly_ticks, ticktext=hourly_ticks.strftime('%Y-%m-%d %H:%M:%S')))
    return fig
@router.post("/graphs/")
async def get_graph(request: GraphMeta):
    """
    :param request:
        graphType: str
        graphSizeX: int
        graphSizeY: int
        region: int
        station: int
        isTime: bool
        channelX: int
        channelNamex: str
        channelsY: list[int]
        channelNamesY: list[str]
        timeInterval: TimeInterval
    :return:
    """
    request = request.dict()
    from_time, to_time = request["timeInterval"]
    from_year, from_month, from_day = from_time
    to_year, to_month, to_day = to_time
    data_request = {"request": "range",
                "data" : {
                    "range":{
                          "from":{
                                "year": from_year,
                                "month": from_month,
                                "day" :from_day
                                },
                            "to": {
                                "year": to_year,
                                "month": to_month,
                                "day": to_day
                            }

                          }
                   }
            }
    getter = RawDataGetter()
    y_channels = request["channelsY"]
    y_names = request["channelNamesY"]
    y_data = {y_names[i]:
                  extract_data(getter.get_channels(channel_id=y_channels[i],
                                      stations_id=request["station"],
                                      request=data_request))for i in range(len(y_names))}
    if request["isTime"]:
        starting_point = datetime(from_year,from_month,from_day, 0, 0, 0)
        graph = get_time_graph(start_time=starting_point,
                               data=y_data,
                               graph_type=request["graphType"],
                               x_size=request["graphSizeX"],
                               y_size=request["graphSizeY"])
    else:
        x_data = extract_data(getter.get_channels(channel_id=request["channelX"],
                                     stations_id=request["station"],
                                     request=data_request))
        data = y_data
        x_name = request["channelNameX"]
        data[x_name] = x_data
        graph = create_graph(data, x_name ,y_names, request["graphType"], request["graphSizeX"], request["graphSizeY"])

    return {"graph_json": graph}

#
# data = {"X" : ["22-10-2020","23-10-2020","24-10-2020","25-10-2020"], "Y1": [1,2,3,4]}
# fig = px.line(data, x='X', y=["Y1"],width=800, height=400, title='Interactive Line Chart')
# fig.show()
data_request = {"request": "range",
                "data" : {
                    "range":{
                          "from":{
                                "year": 2024,
                                "month": 3,
                                "day" :9
                                },
                            "to": {
                                "year": 2024,
                                "month": 3,
                                "day": 10
                            }

                          }
                   }
            }
# import pprint
# data = RawDataGetter().get_channels(stations_id=6,channel_id=1,request=data_request)
# pprint.pprint(data)
get_time_graph(datetime(2024, 1, 1, 0, 0, 0), data={ "y2": [i for i in range(1000)]},graph_type="s",x_size= 1000,y_size=1000)