import os

import pandas as pd
import numpy as np

from . import activity
from . import utils

def load_mnd_data(path, eer_func, window):
    """Process weight and calorie data from a MyNetDiary data export

    Arguments:
        path: Path to MyNetDiary export directory
        eer_func: Lambda wrapper for eer_male or eer_female (above)
                  with height and dob fields specified by caller
        window: Rolling window (in days) for averaging calculations

    Returns:
        df_weight: A Pandas dataframe of processed weight data
        df_calories: A Pandas dataframe of processed calorie data"""

    # Load MyNetDiary data
    mnd_data = utils.merge_excel_files(path)

    # Construct a table of actual & smoothed weights
    df_weight = pd.DataFrame()
    df_weight['Actual'] = mnd_data['Measurements'].query('Measurement == "Body Weight"').set_index('Date')['Value']
    smoothed_weight = df_weight['Actual'].resample('D').interpolate().rolling(5, center=True).mean()
    df_weight['Smoothed'] = smoothed_weight

    # Calculate weight loss rate by regressing over a rolling window
    df_weight['Rate'] = df_weight['Actual'].resample('D').interpolate().rolling(window, center=True).apply(
        lambda x: utils.time_series_linear_regression(x, 1, 'W')[0].iloc[0]['Rate'])

    # Construct a table of baseline (weight maintenance), exercise and food calories
    df_calories = pd.DataFrame()
    df_calories['Food'] = mnd_data['Food'].resample('D', on='Date & Time')['Calories, cals'].sum()
    df_calories['Exercise'] = mnd_data['Exercise'].resample('D', on='Date & Time')['Calories'].sum()
    df_calories['Baseline'] = eer_func(smoothed_weight)
    df_calories.index.rename('Date', inplace=True)
    df_calories.fillna(0, inplace=True)

    # Create an 'adjusted food' column that fills in a rolling average for missing days, then calculate net
    food_masked = df_calories['Food'].mask(df_calories['Food'] == 0)
    df_calories['Food Adj'] = food_masked.fillna(food_masked.rolling(window, min_periods=window//2,
                                                                     center=True).mean())
    df_calories['Net Daily'] = df_calories['Food Adj'] - df_calories['Baseline'] - df_calories['Exercise']

    return df_weight, df_calories

def load_strava_activities(path, recalculate=False):
    """Loads bicycling activity data from a Strava data export

    In addition to loading Strava's activities.csv, calculates certain
    additional metrics like FTP from raw FIT data. Since parsing FIT
    data takes some time, results for each FIT file are cached.

    Arguments:
        path: Path to Strava export directory
        recalculate: Pass true to recalculate all results, pass string
                     or iterable to recalculate only certain data files

    Returns:
        A Pandas dataframe of Strava bicycling activity data"""

    # Load activities.csv and filter out any non-bicycle activities
    csv = pd.read_csv(os.path.join(path, 'activities.csv'))
    csv.query('`Activity Type` in ["Ride", "Virtual Ride"]', inplace=True)

    # Run a set of calculations on all FIT files
    calcs = activity.process_activities(path, csv['Filename'].dropna(), recalculate)

    # Construct the return DataFrame, converting units as appropriate
    # Note timestamps are converted from UTC to local time when offset is available
    df = pd.DataFrame()
    df['Date'] = pd.to_datetime(csv['Activity Date']).add(calcs['UTC Offset'],
                                                          fill_value=np.timedelta64(0))
    df['Description'] = csv['Activity Name']
    df['Bicycle'] = csv['Activity Gear']
    df['Trainer'] = (csv['Activity Type'] == 'Virtual Ride') | (csv['Elevation Gain'] == 0)
    df['Distance'] = csv['Distance'] * 0.6213712 # Convert km to mi
    df['Elevation'] = csv['Elevation Gain'] / 0.3048 # Convert m to ft
    df['Elapsed Time'] = csv['Elapsed Time'] # In seconds
    df['Moving Time'] = csv['Moving Time'] # In seconds
    df['Observed FTP'] = calcs['Observed FTP'] # In watts
    df.set_index('Date', inplace=True, drop=True)

    return df.sort_index()
