"""Functions for processing Strava bicycling activities."""

import functools
import hashlib
import multiprocessing
import os

import numpy as np
import pandas as pd

from . import utils


def get_cache_path():
    """Utility function to retrieve path to cached results."""

    return os.path.join(os.path.dirname(__file__), 'cache.csv')


def process_one_activity(filename, path, cache=None):
    """Run calculations on a single activity."""

    # Manual activities in Strava don't have a related activity file
    if pd.isna(filename):
        return {
            'Filename': filename,
            'Hash': np.nan,
            'Timezone': np.nan,
            'Has Location': False,
            'Max Heart Rate': np.nan,
            'Max Avg Power': np.nan,
        }

    # Calculate hash of file - to make sure cached results are valid
    full_path = os.path.join(path, filename)
    with open(full_path, 'rb') as f:
        hash_val = hashlib.blake2b(f.read(), digest_size=8).hexdigest()

    # Short-circuit and return cached results if available
    try:
        if cache is not None and cache.at[filename, 'Hash'] == hash_val:
            return {
                'Filename': filename,
                'Hash': hash_val,
                'Timezone': cache.at[filename, 'Timezone'],
                'Has Location': cache.at[filename, 'Has Location'],
                'Max Heart Rate': cache.at[filename, 'Max Heart Rate'],
                'Max Avg Power': cache.at[filename, 'Max Avg Power'],
            }
    except KeyError:
        pass

    # Load time-series data from the file
    records, _, _ = utils.parser.parse(full_path)

    # Determine timezone using first valid lat,lng position in the file
    try:
        timezone = utils.infer_timezone(records)
        has_location = True
    except KeyError:
        # No valid lat,lng data in file
        timezone = np.nan
        has_location = False

    # Calculate maximum heart rate observed during activity
    try:
        max_hr = records['heart_rate'].max()
    except KeyError:
        # No heart rate data in file
        max_hr = np.nan

    # Calculate maximum average power over 20 minutes during activity
    try:
        max_avg_power = (
            records['power']
            .resample('S')
            .ffill()
            .rolling(20*60)
            .mean()
            .max()
        )
    except KeyError:
        # No power data in file
        max_avg_power = np.nan

    return {
        'Filename': filename,
        'Hash': hash_val,
        'Timezone': timezone,
        'Has Location': has_location,
        'Max Heart Rate': max_hr,
        'Max Avg Power': max_avg_power,
    }


def process_activities(files, path, recalculate=False):
    """Run calculations on a set of activities.

    Calculates metrics on a set of activity files. Loading of activities in FIT
    format can be slow, so results are cached to a local file. If cached results
    are available, processing will be skipped and results from the cache will be
    returned instead.

    Args:
        files: List of activity files to process.
        path: Path to directory containing the files.
        recalculate: Pass True to recalculate all results, pass file name or
          iterable of file names to recalculate only certain data files.

    Returns:
        DataFrame of calculation results
    """

    # Load cached calculation results if available
    try:
        cache = pd.read_csv(get_cache_path()).set_index('Filename')
        if isinstance(recalculate, str):
            cache = cache[~cache.index.isin([recalculate])]
        elif hasattr(recalculate, '__iter__'):
            cache = cache[~cache.index.isin(recalculate)]
        elif recalculate:
            cache = None
    except FileNotFoundError:
        cache = None

    # Run calculations on all activity files, leveraging cached results
    processor = functools.partial(process_one_activity, path=path, cache=cache)
    with multiprocessing.Pool() as p:
        calcs = pd.DataFrame(p.map(processor, files), index=files.index)

    # Add new results to cache, deduplicate and save
    new_cache = pd.concat([calcs.set_index('Filename'), cache])
    new_cache = new_cache[~new_cache.index.isna()]
    new_cache = new_cache[~new_cache.index.duplicated()]
    new_cache.sort_index().to_csv(get_cache_path())

    return calcs


def load_strava_activities(path, home_tz, recalculate=False):
    """Loads bicycling activity data from a Strava data export.

    In addition to loading Strava's activities.csv, calculates certain
    additional metrics from the underlying activity files. Since parsing all
    activity files takes some time, results are cached to a local file. If
    cached results are available, processing will be skipped and results from
    the cache will be returned instead.

    Args:
        path: Path to Strava export directory.
        home_tz: Timezone to use as a default for activities on a trainer or
          for activities where location data is not available.
        recalculate: Pass True to recalculate all results, pass file name or
          iterable of file names to recalculate only certain data files.

    Returns:
        DataFrame of Strava bicycling activity data.
    """

    # Load activities.csv and filter out any non-bicycle activities
    csv = (
        pd.read_csv(os.path.join(path, 'activities.csv'))
        .query('`Activity Type` in ["Ride", "Virtual Ride"]')
    )

    # Set the UTC date and time of the activity as the index
    csv['Activity Date'] = (
        pd.to_datetime(csv['Activity Date'], format='%b %d, %Y, %I:%M:%S %p')
    )
    csv.set_index('Activity Date', inplace=True)

    # Run a set of calculations on all activity files
    calcs = process_activities(csv['Filename'], path, recalculate)

    # Infer activities that were performed on a stationary trainer
    calcs['Trainer'] = (
        (csv['Activity Type'] == 'Virtual Ride')
        | (~calcs['Has Location'] & ~csv['Filename'].isna())
    )

    # Calculate the local date and time for each activity, subbing in a default
    # timezone for trainer rides or if timezone info is not available
    mask = calcs['Trainer'] | calcs['Timezone'].isna()
    calcs['Timezone Used'] = calcs['Timezone'].mask(mask, home_tz)
    calcs['Local Date'] = [
        date.tz_localize('UTC').tz_convert(tz).tz_localize(None)
        for date, tz in zip(calcs.index, calcs['Timezone Used'])
    ]

    # Construct the return DataFrame, converting units as appropriate
    df = pd.DataFrame()
    df['Date'] = calcs['Local Date']
    df['Description'] = csv['Activity Name']
    df['Bicycle'] = csv['Activity Gear']
    df['Trainer'] = calcs['Trainer']
    df['Commute'] = csv['Commute']
    df['Distance'] = csv['Distance'] * 0.6213712 # Convert km to mi
    df['Elevation'] = csv['Elevation Gain'] / 0.3048 # Convert m to ft
    df['Elapsed Time'] = csv['Elapsed Time'] # In seconds
    df['Moving Time'] = csv['Moving Time'] # In seconds
    df['Max Heart Rate'] = calcs['Max Heart Rate'] # In beats per minute
    df['Max Avg Power (20 min)'] = calcs['Max Avg Power'] # In watts
    df['Filename'] = csv['Filename']

    return df.set_index('Date').sort_index()
