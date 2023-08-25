"""Functions for processing MyNetDiary fitness data."""

import numpy as np
import pandas as pd

from . import utils


def eer_male(weight, height, dob, pa=1.0):
    """Male estimated energy requirements (per day) from MyNetDiary.

    Args:
        weight: Pandas time series of weight measurements in pounds.
        height: Height, in inches.
        dob: Date of birth, as string or datetime64.
        pa: Activity level, 1.0 = sedentary, up to 1.45 for very active.

    Returns:
        Time series of calculated EER on dates of weight measurements.
    """

    # Calculate time series of age in fractional years
    age = (weight.index - np.datetime64(dob)).astype('timedelta64[D]') / 365.25

    # Perform male EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 662 - 9.53 * age + pa * (7.23 * weight + 13.71 * height)


def eer_female(weight, height, dob, pa=1.0):
    """Female estimated energy requirements (per day) from MyNetDiary.

    Args:
        weight: Pandas time series of weight measurements in pounds.
        height: Height, in inches.
        dob: Date of birth, as string or datetime64.
        pa: Activity level, 1.0 = sedentary, up to 1.45 for very active.

    Returns:
        Time series of calculated EER on dates of weight measurements.
    """

    # Calculate time series of age in fractional years
    age = (weight.index - np.datetime64(dob)).astype('timedelta64[D]') / 365.25

    # Perform female EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 354 - 6.91 * age + pa * (4.25 * weight + 18.44 * height)


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
        .apply(utils.calculate_weekly_rate)
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
