import pprint

import dotenv
import requests
import os
import json
import logging
import traceback
import datetime
import pandas as pd
dotenv.load_dotenv()
TOKEN = os.getenv("TOKEN")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class RawDataGetter:
    def __init__(self):
        self.headers = {'Authorization': f'ApiToken {TOKEN}'}

    def get_regions(self, region_id=None):
        """

        :param region_id: int. current valid regions are 0 - 15
        :return: dict. data for each region is : region_id, region_name and list of stations.
        """

        if isinstance(region_id, int):
            url = f"https://api.ims.gov.il/v1/envista/regions/{region_id}"
        else:
            url = f"https://api.ims.gov.il/v1/envista/regions/"
        try:
            results = requests.request("GET", url=url, headers=self.headers)
            if results.status_code == 200:
                data = json.loads(results.text.encode("utf8"))
                return data
            elif results.status_code == 401:
                logging.warning("Not authorized to access data - regions")
                return None
            else:
                logging.warning(f"Unsuccessful to get data - regions, satus code: {results.status_code}. reason : {results.reason}")
                return None
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Network problem occurred while getting regions: %s", conn_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"Invalid request while trying to get region: %s", http_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout occurred while getting regions: %s", timeout_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.TooManyRedirects as redirects_err:
            logging.error(f"Too many redirects while getting regions: %s", redirects_err)
            logging.error(traceback.format_exc())
        except Exception as err:
            logging.error(f"Unexpected error occurred while getting regions: %s", err)
            logging.error(traceback.format_exc())
        return None

    def get_stations(self, station_id=None):
        """

        :param station_id: int. current valid stations are :
                [2, 6, 8, 10, 11, 13, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33, 35, 36, 41, 42,
                 43, 44, 45, 46, 54, 58, 59, 60, 62, 64, 65, 67, 69, 73, 74, 75, 77, 78, 79, 82, 85, 90, 98, 99,
                 106, 107, 112, 115, 121, 123, 124, 178, 186, 188, 202, 205, 206, 207, 208, 210, 211, 212, 218,
                 224, 227, 228, 232, 233, 236, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
                  251, 252, 257, 259, 263, 264, 265, 269, 270, 271, 274, 275, 276, 277, 278, 279, 280, 281, 282,283,
                  284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303,
                  304, 305, 306, 307, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 322, 323, 324, 325,
                   327, 328, 329, 330, 332, 333, 335, 336, 338, 343, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354,
                    355, 366, 367, 370, 373, 379, 380, 381, 411, 412, 443, 480, 498, 499]

        :return: dict. data for each station is : station_name, location and list of channels.
        """

        if isinstance(station_id, int):
            url = f"https://api.ims.gov.il/v1/envista/stations/{station_id}"
        else:
            url = f"https://api.ims.gov.il/v1/envista/stations/"
        try:
            results = requests.request("GET", url=url, headers=self.headers)
            if results.status_code == 200:
                data = json.loads(results.text.encode("utf8"))
                return data
            elif results.status_code == 401:
                logging.warning("Not authorized to access data - stations")
                return None
            else:
                logging.warning(f"Unsuccessful to get data - stations, satus code: {results.status_code}. reason : {results.reason}")
                return None
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Network problem occurred while getting stations: %s", conn_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"Invalid request while trying to get stations: %s", http_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout occurred while getting stations: %s", timeout_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.TooManyRedirects as redirects_err:
            logging.error(f"Too many redirects while getting stations: %s", redirects_err)
            logging.error(traceback.format_exc())
        except Exception as err:
            logging.error(f"Unexpected error occurred while getting stations: %s", err)
            logging.error(traceback.format_exc())
        return None

    def get_url_period(self,stations_id, channel_id, time_period):

        """

        :param stations_id: int.
        :param channel_id: int.
        :param time_period: str. one of daily, monthly, latest, earliest
        :return: str. url for daily data from station request
        """
        if not isinstance(stations_id, int):
            logging.warning("Station id must be int")
            return None
        elif time_period not in ["daily", "monthly", "latest", "earliest"]:
            logging.warning("Time period must be one of daily, monthly, latest, earliest")
            return None
        url = f"https://api.ims.gov.il/v1/envista/stations/{stations_id}/data/{channel_id}/{time_period}"
        return url

    def get_url_specific_day(self, stations_id, channel_id, year, month, day):
        # "https://api.ims.gov.il/v1/envista/stations/%7b%25ST_ID%25%7d/data/daily/%7b%25CH_ID%25%7d/YYYY/MM/DD"
        if all([isinstance(year, int), isinstance(month, int), isinstance(day, int)]):
            try:
                # Trying to build a datetime.datetime object to check date validity
                datetime__obj = datetime.datetime(year=year, month=month, day=day)
                if 1 <= day <= 9:
                    day = f"0{day}"
                if 1 <= month <= 9:
                    month = f"0{month}"
                url = f"https://api.ims.gov.il/v1/envista/stations/{stations_id}/data/{channel_id}/daily/{year}/{month}/{day}"
                return url
            except TypeError as type_err:
                logging.error(f"Invalid date, %s", type_err)
                return None
        else:
            logging.error("Invalid date params, expected all dates params to be int")
            logging.error(traceback.format_exc())
            return None

    def get_url_specific_month(self,stations_id, channel_id, year, month):

        if all([isinstance(year, int), isinstance(month, int)]):
            try:
                # Trying to build a datetime.datetime object to check date validity
                datetime_obj = datetime.datetime(year=year, month=month, day=1)
                if 0 <= month <= 9:
                    month = f"0{month}"
                url = f"https://api.ims.gov.il/v1/envista/stations/{stations_id}/data/{channel_id}/monthly/{year}/{month}"
                return url
            except TypeError as type_err:
                        logging.error(f"Invalid date, %s", type_err)
                        logging.error(traceback.format_exc())
                        return None

        else:
            logging.error("Invalid date params, expected all dates params to be int")
            logging.error(traceback.format_exc())
            return None

    def get_url_range(self, stations_id, channel_id, from_year,from_month, from_day, to_year, to_month, to_day):
        if all([isinstance(from_year,int), isinstance(from_month, int), isinstance(from_day, int),
                isinstance(to_year, int), isinstance(to_month, int), isinstance(to_day, int)]):
            try:
                from_date = datetime.datetime(year=from_year,month=from_month,day=from_day)
                to_date = datetime.datetime(year=to_year,month=to_month,day=to_day)
            except TypeError as type_err:
                logging.error("Could not set a date object due to invalid date parameters %s", type_err)
                logging.error(traceback.format_exc())
                return None
            if from_date < to_date:
                from_day, from_month ,to_day, to_month = list(map(lambda x: f"0{x}" if 0 <= x <=9 else x, [from_day, from_month ,to_day, to_month]))
                url = f"https://api.ims.gov.il/v1/envista/stations/{stations_id}/data/{channel_id}?from={from_year}/{from_month}/{from_day}" \
                      f"&to={to_year}/{to_month}/{to_day}"
                return url
            else:
                logging.error("Invalid date range, from date must be before end date ")
                return None
        else:
            logging.error("Invalid date params, expected all dates params to be int")
            logging.error(traceback.format_exc())
            return None

    def get_station_url(self,stations_id, channel_id, request):
        try:
            if request["request"] == "time_period":
                time_period = request["data"]["time_period"]
                return self.get_url_period(stations_id=stations_id, channel_id=channel_id,time_period=time_period)
            elif request["request"] == "specific_day":
                year, month, day = request["data"]["specific_day"]["year"], request["data"]["specific_day"]["month"],\
                                   request["data"]["specific_day"]["day"]
                return self.get_url_specific_day(stations_id=stations_id, channel_id=channel_id, year=year, month=month, day=day )
            elif request["request"] == "specific_month":
                    year, month = request["data"]["specific_month"]["year"], request["data"]["specific_month"]["month"]
                    return self.get_url_specific_month(stations_id=stations_id, channel_id=channel_id, year=year,month=month)
            elif request["request"] == "range":
                from_year, from_month, from_day = request["data"]["range"]["from"]["year"], \
                                                  request["data"]["range"]["from"]["month"], \
                                                  request["data"]["range"]["from"]["day"]
                to_year, to_month, to_day = request["data"]["range"]["to"]["year"],\
                                            request["data"]["range"]["to"]["month"],\
                                            request["data"]["range"]["to"]["day"]
                return self.get_url_range(stations_id=stations_id, channel_id=channel_id,
                                          from_year=from_year, from_month=from_month, from_day=from_day,
                                          to_year=to_year, to_month=to_month, to_day=to_day)
            else:
                logging.warning("Expected request to be one of:"
                                "current_day, current_month, latest, earliest,specific_day, specific_month or range."
                                f"Got : {request['request']}")
        except KeyError as key_err:
            logging.error("Expected year, month and day. %s ", key_err)
            logging.error(traceback.format_exc())
        except Exception as err:
            logging.error(f"Unexpected error occurred while stations url: %s", err)
            logging.error(traceback.format_exc())
        return None

    def get_channels(self, stations_id, channel_id, request):
        """
        :param stations_id: int (must)
        :param channel_id: int. (optional) if -1 all channels will be returned
        :param request: dict. the request for the period of time.
        :return: dict.

        request example :
        request = {"request": "specific_day",
        "data": {"time_period": <str> one of (daily, monthly, earliest, latest) ,
                "specific_day": {
                            "year": <int>,
                            "month": <int>,
                            "day": <int>

                            },
                "specific_month" : {
                            "year" : <int>,
                            "month" : <int>
                },
                "range" : {
                    "from" : {
                        "year" : <int>,
                        "month": <int>,
                        "day" : <int>
                    },
                    "to" : {
                        "year" : <int>,
                        "month" : <int>,
                        "day" : <int>
                    }

                        }
                }
        }
        """
        url = self.get_station_url(stations_id=stations_id,channel_id=channel_id,request=request)
        if url is None:
            return None
        try:
            results = requests.request("GET", url=url, headers=self.headers)
            results.encoding = results.apparent_encoding
            if results.status_code == 200:
                data = json.loads(results.text.encode(results.encoding))
                return data
            elif results.status_code == 401:
                logging.warning("Not authorized to access data - channels")
                return None
            else:
                logging.warning(f"Unsuccessful to get data - channels, satus code: {results.status_code}. reason : {results.reason}")
                return None
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Network problem occurred while getting channels: %s", conn_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"Invalid request while trying to get channel: %s", http_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout occurred while getting channels: %s", timeout_err)
            logging.error(traceback.format_exc())
        except requests.exceptions.TooManyRedirects as redirects_err:
            logging.error(f"Too many redirects while getting channels: %s", redirects_err)
            logging.error(traceback.format_exc())
        except Exception as err:
            logging.error(f"Unexpected error occurred while getting channels: %s", err)
            logging.error(traceback.format_exc())
        return None



    def get_channels_status(self, station_id):

        station_meta = self.get_stations(station_id)
        if station_meta is not None:

            channels_list = station_meta["monitors"]
            valid_channels = []
            invalid_channels =[]
            for channel in channels_list:
                if isinstance(channel["active"], bool) and channel["active"]:
                    valid_channels.append(channel["name"])
                else:
                    invalid_channels.append(channel["name"])
            return {"valid_channels": valid_channels, "invalid_channels" : invalid_channels}

        else:
            return None


    def pars_region_meta(self, region_id=None):
        """

        :param region_id:
        :return: dict

        Structure:
        region_id <int> : {station_id1 <int> : {'active' : <bool>,
                                                'location': {'latitude': <float>, 'longitude': <float>},
                                                'monitors': {'monitor_name1': {'active': <bool>, 'units': <str>},
                                                'monitor_name2': {'active': <bool>, 'units': <str>} ...
                                                },
                                                'name': <str>}
                            } ,...
                           'regionName': <str>
                           }
        """
        def pars_single_region(region_data):
            pars_data = {}
            region_id = region_data["regionId"]
            region_name = region_data["name"]
            pars_data[region_id] = {"regionName" : region_name}
            for station_data in region_data["stations"]:
                station_parsed_data = RawDataGetter.pars_stations_meta(station_data)
                if station_parsed_data is None:
                    continue

                pars_data[region_id][station_data["stationId"]] = station_parsed_data
            return pars_data

        regions_data = self.get_regions(region_id)
        final_parsed_data = None
        if isinstance(regions_data, list):
            for reg in regions_data:
                try:
                    pared_region = pars_single_region(region_data=reg)
                except Exception as err:
                    logging.warning("Could not pars region data", err)
                    continue
                if isinstance(final_parsed_data, dict):
                    final_parsed_data.update(pared_region)
                else:
                    final_parsed_data = pared_region
            return final_parsed_data
        elif isinstance(regions_data, dict):
            return pars_single_region(region_data=regions_data)
        else:
            return None

    @staticmethod
    def pars_stations_meta(station_data):
        if len(station_data["monitors"]) > 0:
            monitors = {}
            for monitor in station_data["monitors"]:
                monitors[monitor["name"]] = {"active": monitor["active"], "units": monitor["units"],"id": monitor["channelId"] }

            station_parsed_data = {k: v for k, v in station_data.items() if k in ["active",
                                                                                  "location",
                                                                                  "name",
                                                                                  ]}
            station_parsed_data["monitors"] = monitors
            return station_parsed_data
        else:
            return None
# if __name__ == '__main__':
#
#     getter = RawDataGetter()
#     headers = {'Authorization': f'ApiToken {TOKEN}'}
#     stations = [2, 6, 8, 10, 11, 13, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33, 35, 36, 41, 42,
#                      43, 44, 45, 46, 54, 58, 59, 60, 62, 64, 65, 67, 69, 73, 74, 75, 77, 78, 79, 82, 85, 90, 98, 99,
#                      106, 107, 112, 115, 121, 123, 124, 178, 186, 188, 202, 205, 206, 207, 208, 210, 211, 212, 218,
#                      224, 227, 228, 232, 233, 236, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
#                       251, 252, 257, 259, 263, 264, 265, 269, 270, 271, 274, 275, 276, 277, 278, 279, 280, 281, 282,283,
#                       284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303,
#                       304, 305, 306, 307, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 322, 323, 324, 325,
#                        327, 328, 329, 330, 332, 333, 335, 336, 338, 343, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354,
#                         355, 366, 367, 370, 373, 379, 380, 381, 411, 412, 443, 480, 498, 499]
#     request = {"request": "range",
#             "data": {"time_period": "daily",
#                     "specific_day": {
#                                 "year": 2024,
#                                 "month": 2,
#                                 "day": 1
#                                 },
#                     "specific_month" : {
#                                 "year" : 2024,
#                                 "month" : 11
#                     },
#                     "range" : {
#                         "from" : {
#                             "year" : 2024,
#                             "month": 1,
#                             "day" : 1
#                         },
#                         "to" : {
#                             "year" : 2024,
#                             "month" : 2,
#                             "day" : 2
#                         }
#
#                             }
#                     }
#
#             }
#     pprint.pprint(getter.get_stations(2))
#
# # def export_data(request):
# #     getter = RawDataGetter()
# #     stations = getter.get_stations()
# #     for i, station in enumerate(stations):
# #         data = {}
# #         station_name = station["name"]
# #         station_id = station["stationId"]
# #         try:
# #             for monitor in station["monitors"]:
# #                 data_from_channel = getter.get_channels(station_id, channel_id=monitor["channelId"], request=request)
# #                 if data_from_channel:
# #                     values = []
# #                     dates = []
# #                     for data_item in data_from_channel['data']:
# #                         value = data_item['channels'][0]['value']
# #                         date = data_item['datetime']
# #                         values.append(value)
# #                         dates.append(date)
# #                     data[monitor['name']] = values
# #             data["dates"] = dates
# #             df = pd.DataFrame(data)
# #             df.to_csv(f"{station_name}_nov.csv")
# #         except Exception as err:
# #             print(station_name + " has an error")
