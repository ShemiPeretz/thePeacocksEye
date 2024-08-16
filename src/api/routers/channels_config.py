from dotenv import load_dotenv
import os
load_dotenv()
HOURLY_RESOURCE_ID = os.getenv('HOURLY_RESOURCE_ID')
DAILY_RESOURCE_ID = os.getenv('DAILY_RESOURCE_ID')
RADIATION_RESOURCE_ID = os.getenv("RADIATION_RESOURCE_ID")
RAIN_DAILY_RESOURCE_ID = os.getenv('RAIN_DAILY_RESOURCE_ID')
RAIN_MONTHLY_RESOURCE_ID = os.getenv('RAIN_MONTHLY_RESOURCE_ID')
RAIN_YEARLY_RESOURCE_ID = os.getenv("RAIN_YEARLY_RESOURCE_ID")

hourly_channels = ['stn_num', 'time_obs',
                   'prs_stn', 'prs_sea_lvl',
                   'prs_lvl_hgt', 'tmp_air_dry',
                   'tmp_air_wet', 'tmp_dew_pnt',
                   'hmd_rlt', 'wind_dir', 'wind_spd',
                   'year', 'month', 'day',
                   'hour']

daily_channels = ['stn_num', 'time_obs',
                  'tmp_air_max', 'tmp_air_min',
                  'tmp_grass_min', 'sns_drt',
                  'rpr_gale', 'year',
                  'month', 'day']

# time_obs is the date.
rain_channels_daily = ['stn_num', 'time_obs', 'rain_06_next', 'year', 'month', 'day']

rain_channels_monthly = ["stn_num", "time_obs", "rain_ttl",
                         "rain_days_num", "year", "month",
                         "rain_max_day", "rain_max_val"]

rain_channels_yearly = ['stn_num', 'time_obs', 'rain_ttl']


radiation_channels = ['stn_num', 'time_obs', 'rad_type',
                      "rad_4", "rad_5",
                      "rad_6", "rad_7", "rad_8",
                      "rad_9", "rad_10", "rad_11",
                      "rad_12", "rad_13", "rad_14",
                      "rad_15", "rad_16", "rad_17",
                      "year", "month", "day"]

non_cumulative_channels = ['time_obs', 'stn_num', 'year', 'month', 'day', "hour", "rad_type"]

dataset_resource_id_map = {
    "daily": DAILY_RESOURCE_ID,
    "hourly": HOURLY_RESOURCE_ID,
    "radiation": RADIATION_RESOURCE_ID,
    "yearly_rain": RAIN_YEARLY_RESOURCE_ID,
    "monthly_rain": RAIN_MONTHLY_RESOURCE_ID,
    "daily_rain": RAIN_DAILY_RESOURCE_ID
}

dataset_filters = {
    "daily": ["year", "month", "day"],
    "hourly": ["year", "month", "day"],
    "radiation": ["year", "month", "day"],
    "yearly_rain": ["time_obs"],
    "monthly_rain": ["year", "month"],
    "daily_rain": ["year", "month", "day"]
}