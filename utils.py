"""Common utility functions for fitness_analysis module."""

import os

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


def merge_excel_files(path):
    """Loads and merges data from all Excel files in a directory.

    Loads all sheets of all [xls,xlsx] files in a directory and merges them into
    a single dictionary of DataFrames. Identically-named sheets from each Excel
    file will be concatenated in alphabetical order of the Excel file names.

    Args:
        path: Path to directory of Excel files.

    Returns:
        A dict of DataFrames.
    """

    data = {}
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
                data[k] = pd.concat([data[k], excel[k]]).reset_index(drop=True)
            else:
                data[k] = excel[k]

    return data


def time_series_piecewise_regression(series, num_segments, units):
    """Perform a piecewise linear regression on a time series.

    Args:
        series: Time-indexed Series on which to perform regression.
        num_segments: Number of segments for piecewise regression.
        units: Time unit to use for calculating slope at each breakpoint
          (e.g., 'D' means calculate rate per day).

    Returns:
        Tuple of (result, r2).

        result: Time-indexed DataFrame of breakpoints, with value and slope
          calculated for each breakpoint.
        r2: Calculated R^2 score of model fit.
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)

    # Do a multi-segment piecewise linear regression
    model = pwlf.PiecewiseLinFit(x, series.values)
    breakpoints = model.fit(num_segments)
    regression = model.predict(x)

    # Get the conversion factor to desired rate units
    c = np.uint64(np.timedelta64(np.timedelta64(1, units), 'us'))

    # Construct the DataFrame of regression results
    result = pd.DataFrame(index=breakpoints.astype('datetime64[us]'))
    result['Value'] = model.predict(breakpoints)
    result['Rate'] = c * np.append(model.calc_slopes(), [np.NaN])
    r2 = sklearn.metrics.r2_score(series.values, regression)

    return result, r2


def time_series_linear_rate(series, units):
    """Calculate the slope of the linear regression of a time series.

    Args:
        series: Time-indexed Series on which to perform regression.
        units: Time unit to use for calculating slope (e.g., 'D' means calculate
          rate per day).

    Returns:
        Calculated slope as a rate per 'units'
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)

    # Do an ordinary linear regression
    model = sklearn.linear_model.LinearRegression()
    model.fit(x.reshape(-1, 1), series.values)

    # Get the conversion factor to desired rate units
    c = np.uint64(np.timedelta64(np.timedelta64(1, units), 'us'))

    return c * model.coef_[0]


def time_series_constant_regression(series, num_segments):
    """Fit a segmented constant model to a time series.

    Args:
        series: Time-indexed Series on which to perform regression.
        num_segments: Number of segments for model.

    Returns:
        Tuple of (result, r2).

        result: Time-indexed Series of breakpoints, with constant value fitted
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
        index=breakpoints.astype('datetime64[us]')
    )
    result = result.shift(-1, fill_value=result.iloc[-1])
    r2 = sklearn.metrics.r2_score(series.values, regression)

    return result, r2


def infer_timezone(records):
    """Determine the timezone of an activity from its GPS location data

    Finds the first valid latitude and longitude in the data and calculates the
    timezone for that GPS location.

    Args:
        records: The activity data to process

    Returns:
        Timezone string, or None if no timezone has been matched
    """

    points = records[['latitude', 'longitude']]
    idx = points.apply(pd.Series.first_valid_index).max()
    lat, lng = points.loc[idx]
    return tz_finder.timezone_at(lng=lng, lat=lat)


def identify_inactive_periods(series, activity_threshold, min_duration):
    """Identify periods where the value of a time series is not changing
    
    Returns a Boolean Series identifying the periods where the velocity of
    'series' (in units per second) remains below 'activity_threshold' for at
    least a 'min_duration' span of time.

    Args:
        series: Time-indexed Series.
        activity_threshold: Velocity, in units per second, below which values
          of 'series' are considered inactive.
        min_duration: Timedelta object, should be greater than 1 second.
    
    Returns:
        Time-indexed Series of Boolean values.
    """

    # Calculate the derivative of the values of series, in units per second
    # (this will be noisy but works well enough for our purposes)
    velocity = series.resample('s').interpolate().diff()
    below_threshold = velocity < activity_threshold

    # Split series into segments where velocity remains above or below
    # threshold, then calculate the duration of each segment
    group_ids = below_threshold.diff().cumsum()
    grouped = velocity.groupby(group_ids, sort=False)
    durations = group_ids.map(grouped.count())

    # Return True where series < activity_threshold for at least min_duration
    return (durations >= min_duration.seconds) & below_threshold
