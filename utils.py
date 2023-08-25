"""Common utility functions for fitness_analysis module."""

import os

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import sklearn.metrics

import pwlf


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
        for k in excel:
            if k in data:
                data[k] = pd.concat([data[k], excel[k]]).reset_index(drop=True)
            else:
                data[k] = excel[k]

    return data


def time_series_linear_regression(series, num_segments, units):
    """Perform a piecewise linear regression on a time series.

    Args:
        series: Pandas time series on which to perform regression.
        num_segments: Number of segments for piecewise regression.
        units: Time unit to use for calculating slope at each breakpoint
          (e.g., 'D' means calculate rate per day).

    Returns:
        A tuple of (df, r2).

        df: Dataframe (time-indexed) of breakpoints, with value and slope
          calculated for each breakpoint.
        r2: Calculated R^2 score of model fit.
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)

    # Do a multi-segment piecewise linear regression
    piecewise = pwlf.PiecewiseLinFit(x, series.values)
    breakpoints = piecewise.fit(num_segments)
    values = piecewise.predict(breakpoints)
    slopes = np.append(piecewise.calc_slopes(), [np.NaN])
    regression = piecewise.predict(x)

    # Get the conversion factor to desired rate units
    c = np.uint64(np.timedelta64(np.timedelta64(1, units), 'us'))

    # Construct a table of overall regression results
    df = pd.DataFrame(index=breakpoints.astype('datetime64[us]'))
    df['Value'] = values
    df['Rate'] = slopes * c
    r2 = sklearn.metrics.r2_score(series.values, regression)

    return df, r2


def calculate_weekly_rate(data):
    """Calculates average velocity of a time series on a per week basis."""

    return time_series_linear_regression(data, 1, 'W')[0].iloc[0]['Rate']


def time_series_constant_regression(series, num_segments):
    """Fit a segmented constant model to a time series.

    Args:
        series: Pandas time series on which to perform regression.
        num_segments: Number of segments for model.

    Returns:
        Pandas series (time-indexed) of breakpoints, with constant
        value fitted at each breakpoint.
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)

    # Do the multi-segment constant regression and return the result
    piecewise = pwlf.PiecewiseLinFit(x, series.values, degree=0)
    breakpoints = piecewise.fit(num_segments)
    return pd.Series(shift(piecewise.predict(breakpoints), -1, cval=np.NaN),
                     index=breakpoints.astype('datetime64[us]'))

