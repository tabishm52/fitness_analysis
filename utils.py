"""Common utility functions for fitness_analysis module."""

import os
from collections.abc import Iterable
from os import PathLike

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics

import pwlf
import timezonefinder

import activity_parser

# Global objects that can be used throughout the fitness_analysis module
parser = activity_parser.ActivityParser()
tz_finder = timezonefinder.TimezoneFinder()


def merge_excel_files(path: str | PathLike[str]) -> dict[str, pd.DataFrame]:
    """Loads and merges data from all Excel files in a directory.

    Loads all sheets of all [xls,xlsx] files in a directory and merges them into
    one mapping keyed by sheet name. Identically named sheets from each Excel
    file are concatenated in alphabetical order of Excel file name.

    Args:
        path: Directory of Excel files.

    Returns:
        A mapping of merged sheet data keyed by sheet name.
    """

    data: dict[str, pd.DataFrame] = {}
    files = os.listdir(path)
    files.sort()

    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() not in ['.xls', '.xlsx']:
            continue

        excel = pd.read_excel(os.path.join(path, f), sheet_name=None)
        nonempty_sheets = (k for k in excel if not excel[k].empty)
        for k in nonempty_sheets:
            if k in data:
                data[k] = (
                    pd.concat([data[k], excel[k]])
                    .reset_index(drop=True)
                )
            else:
                data[k] = excel[k]

    return data


def pwlf_wrapper(
    series: pd.Series,
    units: str,
    breaks: Iterable[pd.Timestamp] | None = None,
    num_segments: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Worker function for piecewise linear regression on time-indexed data."""

    # Convert time index to a uint array (in us) for scipy calculations
    time_us = (
        series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)
    )

    # Calculate the conversion factor to desired rate units
    c = np.uint64(np.timedelta64(np.timedelta64(1, units), 'us'))

    # Do a multi-segment piecewise linear regression. Use specified breakpoints
    # if provided, otherwise let pwlf pick the breakpoints.
    model = pwlf.PiecewiseLinFit(time_us, series.values)
    if breaks is not None:
        breaks_us = np.array(breaks).astype('datetime64[us]').astype(np.uint64)
        model.fit_with_breaks(breaks_us)
    else:
        breaks_us = model.fit(num_segments)

    # Construct the DataFrame of regression results
    regression = model.predict(time_us)
    result = pd.DataFrame(index=breaks_us.astype('datetime64[us]'))
    result['Value'] = model.predict(breaks_us)
    result['Rate'] = c * np.append(model.calc_slopes(), [np.nan])
    r2 = sklearn.metrics.r2_score(series.values, regression)

    return result, r2


def time_series_piecewise_regression(
    series: pd.Series,
    num_segments: int,
    units: str,
) -> tuple[pd.DataFrame, float]:
    """Perform a piecewise linear regression on a time series.

    Args:
        series: Time-indexed values on which to perform regression.
        num_segments: Number of segments for piecewise regression.
        units: Time unit to use for calculating slope at each breakpoint
            (e.g., 'D' means calculate rate per day).

    Returns:
        Tuple of (result, r2).

        result: Time-indexed breakpoints with value and slope
            calculated for each breakpoint.
        r2: Calculated R^2 score of model fit.
    """

    return pwlf_wrapper(series, units, num_segments=num_segments)


def time_series_piecewise_regression_with_breaks(
    series: pd.Series,
    breaks: Iterable[pd.Timestamp],
    units: str,
) -> tuple[pd.DataFrame, float]:
    """Perform a piecewise linear regression with specified breakpoints.

    Args:
        series: Time-indexed values on which to perform regression.
        breaks: Iterable of Timestamps of breakpoints for piecewise regression.
        units: Time unit to use for calculating slope at each breakpoint
            (e.g., 'D' means calculate rate per day).

    Returns:
        Tuple of (result, r2).

        result: Time-indexed breakpoints with value and slope
            calculated for each breakpoint.
        r2: Calculated R^2 score of model fit.
    """

    return pwlf_wrapper(series, units, breaks=breaks)


def time_series_linear_rate(series: pd.Series, units: str) -> float:
    """Calculate the slope of the linear regression of a time series.

    Args:
        series: Time-indexed values on which to perform regression.
        units: Time unit to use for calculating slope (e.g., 'D' means calculate
          rate per day).

    Returns:
        Calculated slope as a rate per ``units``.
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)

    # Do an ordinary linear regression
    model = sklearn.linear_model.LinearRegression()
    model.fit(x.reshape(-1, 1), series.values)

    # Get the conversion factor to desired rate units
    c = np.uint64(np.timedelta64(np.timedelta64(1, units), 'us'))

    return c * model.coef_[0]


def time_series_constant_regression(
    series: pd.Series,
    num_segments: int,
) -> tuple[pd.Series, float]:
    """Fit a segmented constant model to a time series.

    Args:
        series: Time-indexed values on which to perform regression.
        num_segments: Number of segments for model.

    Returns:
        Tuple of (result, r2).

        result: Time-indexed breakpoints with fitted constant value
            at each breakpoint. Use result.resample().ffill() to plot.
        r2: Calculated R^2 score of model fit.
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)

    # Do a multi-segment constant regression
    model = pwlf.PiecewiseLinFit(x, series.values, degree=0)
    breakpoints = model.fit(num_segments)
    regression = model.predict(x)

    # Construct the Series of regression results
    result = pd.Series(
        model.predict(breakpoints),
        index=breakpoints.astype('datetime64[us]'),
    )
    result = result.shift(-1, fill_value=result.iloc[-1])
    r2 = sklearn.metrics.r2_score(series.values, regression)

    return result, r2


def infer_timezone(records: pd.DataFrame) -> str | None:
    """Determine the timezone of an activity from its GPS location data.

    Finds the first valid latitude and longitude in the data and calculates the
    timezone for that GPS location.

    Args:
        records: Activity data to process.

    Returns:
        Timezone name, or None if no valid latitude/longitude is available
        or no timezone match is found.
    """

    try:
        points = records[['latitude', 'longitude']]
    except KeyError:
        return None

    idx = points.apply(pd.Series.first_valid_index).max()
    if idx is None:
        return None

    lat, lng = points.loc[idx]
    if pd.isna(lat) or pd.isna(lng):
        return None

    return tz_finder.timezone_at(lng=lng, lat=lat)


def identify_inactive_periods(
    series: pd.Series,
    activity_threshold: float,
    min_duration: pd.Timedelta,
) -> pd.Series:
    """Identify periods where values in a time series are not changing.

    Returns a boolean mask identifying periods where the velocity of
    ``series`` (in units per second) remains below ``activity_threshold`` for
    at least ``min_duration``.

    Args:
        series: Time-indexed values.
        activity_threshold: Velocity, in units per second, below which values
            of ``series`` are considered inactive.
        min_duration: Minimum inactivity span to flag.

    Returns:
        Time-indexed boolean mask.
    """

    # Calculate the derivative of series values, in units per second.
    # (this will be noisy but works well enough for our purposes)
    velocity = series.resample('s').interpolate().diff()
    below_threshold = velocity < activity_threshold

    # Split series into segments where velocity remains above or below
    # threshold, then calculate each segment duration.
    group_ids = below_threshold.diff().cumsum()
    grouped = velocity.groupby(group_ids, sort=False)
    durations = group_ids.map(grouped.count())

    # Return True where velocity stays below the threshold for min_duration.
    return (durations >= min_duration.total_seconds()) & below_threshold
