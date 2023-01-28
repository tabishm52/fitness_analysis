"""Functions for processing MyNetDiary and Strava fitness data."""

import os

import pandas as pd

from . import activity
from . import utils


def calculate_weekly_rate(data):
    """Calculates average velocity of a time series on a per week basis."""

    return utils.time_series_linear_regression(data, 1, 'W')[0].iloc[0]['Rate']


def load_mnd_data(path, eer_func, window):
    """Process weight and calorie data from a MyNetDiary data export.

    Args:
        path: Path to MyNetDiary export directory.
        eer_func: Function object that wraps eer_male or eer_female with height
          and dob fields specified by caller.
        window: Rolling window (in days) for averaging calculations.

    Returns:
        A tuple of (weight, calories)

        weight: A Pandas DataFrame of processed weight data.
        calories: A Pandas DataFrame of processed calorie data.
    """

    # Load MyNetDiary data
    mnd_data = utils.merge_excel_files(path)

    # Construct a table of actual & smoothed weights
    weight = pd.DataFrame()
    weight['Actual'] = (
        mnd_data['Measurements']
        .query('Measurement == "Body Weight"')
        .set_index('Date')['Value']
    )
    smoothed_weight = (
        weight['Actual']
        .resample('D')
        .interpolate()
        .rolling(5, center=True)
        .mean()
    )
    weight['Smoothed'] = smoothed_weight

    # Calculate weight gain/loss rate over time
    weight['Rate'] = (
        weight['Actual']
        .resample('D')
        .interpolate()
        .rolling(window, center=True)
        .apply(calculate_weekly_rate)
    )

    # Construct a table of calorie information
    calories = pd.DataFrame()
    calories['Food'] = (
        mnd_data['Food']
        .resample('D', on='Date & Time')['Calories, cals']
        .sum()
    )
    calories['Exercise'] = (
        mnd_data['Exercise']
        .resample('D', on='Date & Time')['Calories']
        .sum()
    )
    calories['Baseline'] = eer_func(smoothed_weight)
    calories.index.rename('Date', inplace=True)
    calories.fillna(0, inplace=True)

    # Create an 'adjusted food' column that fills in a rolling average for
    # missing days, then calculate net calorie excess / deficit
    food_masked = (
        calories['Food']
        .mask(calories['Food'] == 0)
    )
    calories['Food Adj'] = (
        food_masked.fillna(
            food_masked
            .rolling(window, min_periods=window//2, center=True)
            .mean()
        )
    )
    calories['Net Daily'] = (
        calories['Food Adj']
        - calories['Baseline']
        - calories['Exercise']
    )

    return weight, calories


def load_strava_activities(path, recalculate=False):
    """Loads bicycling activity data from a Strava data export.

    In addition to loading Strava's activities.csv, calculates certain
    additional metrics from the underlying activity files. Since parsing all
    activity files takes some time, results are cached to a local file. If
    cached results are available, processing will be skipped and results from
    the cache will be returned instead.

    Args:
        path: Path to Strava export directory.
        recalculate: Pass True to recalculate all results, pass string or
          iterable of file name(s) to recalculate only certain data files.

    Returns:
        A Pandas DataFrame of Strava bicycling activity data.
    """

    # Load activities.csv and filter out any non-bicycle activities
    csv = pd.read_csv(os.path.join(path, 'activities.csv'))
    csv.query('`Activity Type` in ["Ride", "Virtual Ride"]', inplace=True)
    csv.reset_index(drop=True, inplace=True)

    # Run a set of calculations on all FIT files
    calcs = activity.process_activities(csv['Filename'], path, recalculate)

    # This is kinda ugly - convert all dates to local time (or a default) and
    # then drop the tzinfo so that weekly/daily calcs match Strava
    def activity_local_times():
        for i, _ in csv.iterrows():
            if (
                pd.isna(calcs.loc[i, 'Timezone'])
                or csv.loc[i, 'Activity Type'] == 'Virtual Ride'
            ):
                yield (
                    pd.to_datetime(csv.loc[i, 'Activity Date'])
                    .tz_localize('UTC')
                    .tz_convert('America/Los_Angeles')
                    .replace(tzinfo=None)
                )
            else:
                yield (
                    pd.to_datetime(csv.loc[i, 'Activity Date'])
                    .tz_localize('UTC')
                    .tz_convert(calcs.loc[i, 'Timezone'])
                    .replace(tzinfo=None)
                )

    # Construct the return DataFrame, converting units as appropriate
    df = pd.DataFrame()
    df['Date'] = pd.Series(activity_local_times(), csv.index)
    df['Description'] = csv['Activity Name']
    df['Bicycle'] = csv['Activity Gear']
    df['Trainer'] = ((csv['Activity Type'] == 'Virtual Ride') |
                     (~calcs['Has Location'] & ~csv['Filename'].isna()))
    df['Distance'] = csv['Distance'] * 0.6213712 # Convert km to mi
    df['Elevation'] = csv['Elevation Gain'] / 0.3048 # Convert m to ft
    df['Elapsed Time'] = csv['Elapsed Time'] # In seconds
    df['Moving Time'] = csv['Moving Time'] # In seconds
    df['Observed FTP'] = calcs['Observed FTP'] # In watts
    df['Filename'] = csv['Filename']
    df.set_index('Date', inplace=True)

    return df.sort_index()
