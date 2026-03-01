"""Functions for processing MyNetDiary fitness data."""

from collections.abc import Callable
from os import PathLike

import numpy as np
import pandas as pd

from . import utils


def eer_male(
    weight: pd.Series,
    height: float,
    dob: str | np.datetime64 | pd.Timestamp,
    pa: float = 1.0,
) -> pd.Series:
    """Male estimated energy requirements (per day) from MyNetDiary.

    Args:
        weight: Time-indexed weight measurements in pounds.
        height: Height, in inches.
        dob: Date of birth.
        pa: Activity level, 1.0 = sedentary, up to 1.45 for very active.

    Returns:
        Estimated daily energy requirement for each timestamp in ``weight``.
    """

    # Calculate time series of age in fractional years
    age = (weight.index - np.datetime64(dob)).days / 365.25

    # Perform male EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 662 - 9.53 * age + pa * (7.23 * weight + 13.71 * height)


def eer_female(
    weight: pd.Series,
    height: float,
    dob: str | np.datetime64 | pd.Timestamp,
    pa: float = 1.0,
) -> pd.Series:
    """Female estimated energy requirements (per day) from MyNetDiary.

    Args:
        weight: Time-indexed weight measurements in pounds.
        height: Height, in inches.
        dob: Date of birth.
        pa: Activity level, 1.0 = sedentary, up to 1.45 for very active.

    Returns:
        Estimated daily energy requirement for each timestamp in ``weight``.
    """

    # Calculate time series of age in fractional years
    age = (weight.index - np.datetime64(dob)).astype('timedelta64[D]') / 365.25

    # Perform female EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 354 - 6.91 * age + pa * (4.25 * weight + 18.44 * height)


def load_mnd_data(
    path: str | PathLike[str],
    eer_func: Callable[[pd.Series], pd.Series],
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process weight and calorie data from a MyNetDiary data export.

    Args:
        path: MyNetDiary export directory.
        eer_func: Callable that wraps eer_male or eer_female with height
          and dob fields specified by caller.
        window: Rolling window (in days) for averaging calculations.

    Returns:
        Tuple containing processed weight and calorie metrics.
    """

    # Load MyNetDiary data
    mnd_data = utils.merge_excel_files(path)

    # Construct a table of actual & smoothed weights
    weight = pd.DataFrame()
    weight['Actual'] = (
        mnd_data['Measurements']
        .query('Measurement == "Body Weight"')
        .set_index('Date')
        ['Value']
        .resample('D')
        .mean()
    )
    weight['Smoothed'] = (
        weight['Actual']
        .rolling(7, min_periods=2, center=True)
        .mean()
    )

    # Calculate weight gain/loss rate over time
    weight['Rate'] = (
        weight['Actual']
        .rolling(window, min_periods=window//2, center=True)
        .apply(lambda x: utils.time_series_linear_rate(x.dropna(), 'W'))
    )

    # Construct a table of calorie information
    calories = pd.DataFrame()
    calories['Food'] = (
        mnd_data['Food']
        .resample('D', on='Date & Time')
        ['Calories, cals']
        .sum(min_count=1)
    )
    calories['Exercise'] = (
        mnd_data['Exercise']
        .resample('D', on='Date & Time')
        ['Calories']
        .sum()
    )
    calories['Exercise'] = calories['Exercise'].fillna(0)
    calories['Baseline'] = (
        eer_func(
            weight['Smoothed']
            .ffill()
            .reindex(calories.index, method='ffill')
        )
    )
    calories.index.rename('Date', inplace=True)

    # Create an 'adjusted food' column that fills in a rolling average for
    # days where no food was logged
    calories['Food Adj'] = (
        calories['Food'].fillna(
            calories['Food']
            .rolling(window, min_periods=window//2, center=True)
            .mean()
        )
    )

    # Calculate net calorie balance for each day
    calories['Net Daily'] = (
        calories['Food Adj']
        - calories['Baseline']
        - calories['Exercise']
    )

    # Calculate rolling average of net calorie balance
    calories['Net Recorded'] = (
        calories['Net Daily']
        .rolling(window, min_periods=window//2, center=True)
        .mean()
    )

    # Convert observed weight gain/loss in lbs/week to calories/day
    calories['Net Observed'] = 500 * weight['Rate']

    # Calculate "accuracy" of calorie counting relative to actual weight loss
    avg_food_recorded = (
        calories['Food Adj']
        .rolling(window, min_periods=window//2, center=True)
        .mean()
    )
    avg_exercise = (
        calories['Exercise']
        .rolling(window, min_periods=window//2, center=True)
        .mean()
    )
    avg_consumption_observed = (
        calories['Baseline']
        + avg_exercise
        + calories['Net Observed']
    )
    calories['Accuracy'] = avg_food_recorded / avg_consumption_observed

    return weight, calories
