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
    age = (weight.index - np.datetime64(dob)).astype("timedelta64[D]") / 365.25

    # Perform female EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 354 - 6.91 * age + pa * (4.25 * weight + 18.44 * height)


def _ewm_min_periods_from_halflife(
    halflife: str | pd.Timedelta,
    coverage: float,
    floor: int = 2,
) -> int:
    """Derive EWM ``min_periods`` from half-life and target weight coverage.

    Args:
        halflife: EWM half-life (for example, ``'3D'``).
        coverage: Target cumulative EWM weight mass in ``(0, 1)``.
        floor: Minimum returned value.

    Returns:
        Integer ``min_periods`` aligned to the specified half-life.
    """

    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")

    halflife_days = pd.to_timedelta(halflife) / pd.Timedelta("1D")
    if halflife_days <= 0:
        raise ValueError("halflife must be positive")

    daily_decay = 0.5 ** (1 / halflife_days)
    periods = int(np.ceil(np.log(1 - coverage) / np.log(daily_decay)))

    return max(floor, periods)


def load_mnd_data(
    path: str | PathLike[str],
    eer_func: Callable[[pd.Series], pd.Series],
    weight_halflife: str = "3D",
    calorie_halflife: str = "9D",
    rate_window_days: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process weight and calorie data from a MyNetDiary data export.

    Args:
        path: MyNetDiary export directory.
        eer_func: Callable that wraps eer_male or eer_female with height
            and dob fields specified by caller.
        weight_halflife: Exponential half-life for weight smoothing.
        calorie_halflife: Exponential half-life for calorie smoothing.
        rate_window_days: Rolling window (days) for weight-rate regression.

    Returns:
        Tuple containing processed weight and calorie metrics.
    """

    # Load MyNetDiary data
    mnd_data = utils.merge_excel_files(path)

    # Tuned averaging controls
    weight_coverage = 0.50
    calorie_coverage = 0.67

    weight_min_periods = _ewm_min_periods_from_halflife(
        weight_halflife,
        coverage=weight_coverage,
    )
    calorie_min_periods = _ewm_min_periods_from_halflife(
        calorie_halflife,
        coverage=calorie_coverage,
    )

    rate_min_periods = max(2, rate_window_days // 2)

    # Construct a table of actual & smoothed weights
    weight = pd.DataFrame()
    weight["Actual"] = (
        mnd_data["Measurements"]
        .query('Measurement == "Body Weight"')
        .set_index("Date")["Value"]
        .resample("D")
        .mean()
    )
    weight["Smoothed"] = (
        weight["Actual"]
        .ewm(
            halflife=weight_halflife,
            times=weight.index,
            min_periods=weight_min_periods,
        )
        .mean()
    )

    # Calculate weight gain/loss rate over time
    weight["Rate"] = (
        weight["Smoothed"]
        .rolling(
            rate_window_days,
            min_periods=rate_min_periods,
            center=True,
        )
        .apply(lambda x: utils.time_series_linear_rate(x.dropna(), "W"))
    )

    # Construct a table of calorie information
    calories = pd.DataFrame()
    calories["Food"] = (
        mnd_data["Food"]
        .resample("D", on="Date & Time")["Calories, cals"]
        .sum(min_count=1)
    )
    calories["Exercise"] = (
        mnd_data["Exercise"].resample("D", on="Date & Time")["Calories"].sum()
    )
    calories["Exercise"] = calories["Exercise"].fillna(0)
    calories["Baseline"] = eer_func(
        weight["Smoothed"].reindex(calories.index, method="ffill")
    )
    calories.index.rename("Date", inplace=True)

    # Create an 'adjusted food' column that fills in a rolling average for
    # days where no food was logged
    calories["Food Adj"] = calories["Food"].fillna(
        calories["Food"]
        .ewm(
            halflife=calorie_halflife,
            times=calories.index,
            min_periods=calorie_min_periods,
        )
        .mean()
    )

    # Calculate net calorie balance for each day
    calories["Net Daily"] = (
        calories["Food Adj"] - calories["Baseline"] - calories["Exercise"]
    )

    # Calculate rolling average of net calorie balance
    calories["Net Recorded"] = (
        calories["Net Daily"]
        .ewm(
            halflife=calorie_halflife,
            times=calories.index,
            min_periods=calorie_min_periods,
        )
        .mean()
    )

    # Convert observed weight gain/loss in lbs/week to calories/day
    calories["Net Observed"] = 500 * weight["Rate"]

    # Calculate "accuracy" of calorie counting relative to actual weight loss
    avg_food_recorded = (
        calories["Food Adj"]
        .ewm(
            halflife=calorie_halflife,
            times=calories.index,
            min_periods=calorie_min_periods,
        )
        .mean()
    )
    avg_exercise = (
        calories["Exercise"]
        .ewm(
            halflife=calorie_halflife,
            times=calories.index,
            min_periods=calorie_min_periods,
        )
        .mean()
    )
    avg_consumption_observed = (
        calories["Baseline"] + avg_exercise + calories["Net Observed"]
    )
    calories["Accuracy"] = avg_food_recorded / avg_consumption_observed

    return weight, calories
