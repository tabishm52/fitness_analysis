import os
import hashlib
import gzip

import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.metrics
from scipy.ndimage.interpolation import shift

import pwlf
import fitanalysis

# A list of files where I happened to record janky power data
BAD_POWER_DATA = ['activities/3799945079.fit.gz']

def _get_cache_path():
    """Utility function to retrieve path to cached results"""
    return os.path.splitext(__file__)[0] + '.cache'

def merge_excel_files(path):
    """Loads and merges data from all Excel files in a directory

    Loads all sheets of all [xls,xlsx] files in a directory and merges
    them into a single dictionary of DataFrames. Identically-named
    sheets from each Excel file will be concatenated in alphabetical
    order of the Excel file names.

    Arguments:
        path: Path to directory of Excel files

    Returns:
        A dictionary of DataFrames"""

    data = dict()
    files = os.listdir(path)
    files.sort()

    for f in files:
        (root, ext) = os.path.splitext(f)
        if ext != '.xls' and ext != '.xlsx':
            continue

        excel = pd.read_excel(os.path.join(path, f), sheet_name=None)
        for k in excel:
            if k in data:
                data[k] = pd.concat([data[k], excel[k]]).reset_index(drop=True)
            else:
                data[k] = excel[k]

    return data

def time_series_linear_regression(series, num_segments, units):
    """Perform a piecewise linear regression on a time series

    Arguments:
        series: Pandas time series on which to perform regression
        num_segments: Number of segments for piecewise regression
        units: Time unit to use for calculating slope at each
               breakpoint (e.g., 'D' means calculate rate per day)

    Returns:
        df: Dataframe (time-indexed) of breakpoints, with value and
            slope calculated for each breakpoint
        r2: Calculated R^2 score of model fit
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)

    if num_segments == 1:
        # Do an ordinary linear regression
        model = sklearn.linear_model.LinearRegression()
        model.fit(x[:, np.newaxis], series.values)
        breakpoints = np.array([x[0], x[-1]])
        values = model.predict(breakpoints[:, np.newaxis])
        slopes = np.array([model.coef_[0], np.NaN])
        regression = model.predict(x[:, np.newaxis])

    else:
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

def time_series_constant_regression(series, num_segments):
    """Fit a segmented constant model to a time series

    Arguments:
        series: Pandas time series on which to perform regression
        num_segments: Number of segments for model

    Returns:
        Pandas series (time-indexed) of breakpoints, with constant
        value fitted at each breakpoint
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = series.index.to_numpy().astype('datetime64[us]').astype(np.uint64)

    # Do the multi-segment constant regression and return the result
    piecewise = pwlf.PiecewiseLinFit(x, series.values, degree=0)
    breakpoints = piecewise.fit(num_segments)
    return pd.Series(index=breakpoints.astype('datetime64[us]'),
                     data=shift(piecewise.predict(breakpoints), -1, cval=np.NaN))

def eer_male(weight, height, dob, pa=1.0):
    """Male estimated energy requirements (per day) from MyNetDiary

    Arguments:
        weight: Pandas time series of weight measurements in pounds
        height: Height, in inches
        dob: Date of birth, as string or datetime64
        pa: Activity level, 1.0 = sedentary, up to 1.45 for very active

    Returns:
        Time series of calculated EER on dates of weight measurements
    """

    # Calculate time series of age in days
    age = (weight.index - np.datetime64(dob)).astype('timedelta64[D]') / 365.25

    # Perform male EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 662 - 9.53 * age + pa * (7.23 * weight + 13.71 * height)

def eer_female(weight, height, dob, pa=1.0):
    """Female estimated energy requirements (per day) from MyNetDiary

    Arguments:
        weight: Pandas time series of weight measurements in pounds
        height: Height, in inches
        dob: Date of birth, as string or datetime64
        pa: Activity level, 1.0 = sedentary, up to 1.45 for very active

    Returns:
        Time series of calculated EER on dates of weight measurements
    """

    # Calculate time series of age in days
    age = (weight.index - np.datetime64(dob)).astype('timedelta64[D]') / 365.25

    # Perform female EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 354 - 6.91 * age + pa * (4.25 * weight + 18.44 * height)

def _process_fit_data(path, fname, cache=None):
    """Run calculations on a FIT file for load_strava_activities()"""

    # Calculate hash of file - to make sure cached results are valid
    full_path = os.path.join(path, fname)
    with open(full_path, "rb") as f:
        hash = hashlib.blake2b(f.read(), digest_size=8).hexdigest()

    # Short-circuit and return cached results if available
    try:
        if cache is not None and cache.loc[fname, 'Hash'] == hash:
            return cache.loc[fname].to_numpy()
    except KeyError:
        pass

    # Strava seems to gzip some but not all files; run gunzip if needed
    root, ext = os.path.splitext(fname)
    if ext == '.gz':
        file = gzip.open(full_path)
        root, ext = os.path.splitext(root)
    else:
        file = full_path

    # Return null data if this is not a FIT file (e.g. for GPX files)
    if ext != '.fit':
        return hash, np.timedelta64(0), np.NaN

    # Load the FIT file for analysis
    # Note: This is some rando code from github and is super slow
    try:
	    fitfile = fitanalysis.Activity(file)
    except TypeError:
        # fitanalysis seems to puke on non-Garmin FIT files
        return hash, np.timedelta64(0), np.NaN

    # Calculate UTC offset if available in FIT file
    try:
        m = next(fitfile.get_messages('activity'))
        offset = np.timedelta64(m.get('local_timestamp').value - m.get('timestamp').value)
    except AttributeError:
        offset = np.timedelta64(0)

    # Calculate 20-min average power (FTP) if available in FIT file
    if fitfile.has_power and fname not in BAD_POWER_DATA:
        p = fitfile.power.droplevel(level='block')
        ftp = np.max(fitanalysis.util.moving_average(p, 20*60))
    else:
        ftp = np.NaN

    return hash, offset, ftp

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

    # Load cached calculation results if available
    try:
        cache = pd.read_hdf(_get_cache_path(), 'strava_calcs')
        if isinstance(recalculate, str):
            cache = cache[~cache.index.isin([recalculate])]
        elif hasattr(recalculate, '__iter__'):
            cache = cache[~cache.index.isin(recalculate)]
        elif recalculate:
            cache = None
    except FileNotFoundError:
        cache = None

    # Run calculations on all FIT files, leveraging cached results
    calcs = pd.DataFrame()
    calcs['Filename'] = csv['Filename'].dropna()
    calcs['Hash'], calcs['UTC Offset'], calcs['Observed FTP'] = \
        zip(*calcs['Filename'].map(lambda x: _process_fit_data(path, x, cache)))

    # Add new results to cache, deduplicate and save
    new_cache = pd.concat([calcs.set_index('Filename', drop=True), cache])
    new_cache = new_cache[~new_cache.index.duplicated()].sort_index()
    new_cache.to_hdf(_get_cache_path(), 'strava_calcs', mode='w')

    # Construct the return DataFrame, converting units as appropriate
    # Note timestamps are converted from UTC to local time when offset is available
    df = pd.DataFrame()
    df['Date'] = pd.to_datetime(csv['Activity Date']) + calcs['UTC Offset']
    df['Description'] = csv['Activity Name']
    df['Bicycle'] = csv['Activity Gear']
    df['Distance'] = csv['Distance'] * 0.6213712 # Convert km to mi
    df['Elevation'] = csv['Elevation Gain'] / 0.3048 # Convert m to ft
    df['Elapsed Time'] = csv['Elapsed Time'] # In seconds
    df['Moving Time'] = csv['Moving Time'] # In seconds
    df['Observed FTP'] = calcs['Observed FTP'] # In watts
    df.set_index('Date', inplace=True, drop=True)

    return df.sort_index()
