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


def process_one_activity(fname, path, cache=None):
    """Run calculations on a single Strava activity."""

    # Manual activities in Strava don't have a related activity file
    if pd.isna(fname):
        return {
            'Filename': fname,
            'Hash': np.NaN,
            'Timezone': np.NaN,
            'Has Location': False,
            'Observed FTP': np.NaN,
        }

    # Calculate hash of file - to make sure cached results are valid
    full_path = os.path.join(path, fname)
    with open(full_path, 'rb') as f:
        hash_val = hashlib.blake2b(f.read(), digest_size=8).hexdigest()

    # Short-circuit and return cached results if available
    try:
        if cache is not None and cache.loc[fname, 'Hash'] == hash_val:
            return {
                'Filename': fname,
                'Hash': hash_val,
                'Timezone': cache.loc[fname, 'Timezone'],
                'Has Location': cache.loc[fname, 'Has Location'],
                'Observed FTP': cache.loc[fname, 'Observed FTP'],
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
        timezone = np.NaN
        has_location = False

    # Estimate FTP by calculating maximum 20-minute effort in file
    try:
        observed_ftp = (
            np.max(
                records['power']
                .resample('S')
                .ffill()
                .rolling(20*60)
                .mean()
            )
        )
    except KeyError:
        # No power data in file
        observed_ftp = np.NaN

    return {
        'Filename': fname,
        'Hash': hash_val,
        'Timezone': timezone,
        'Has Location': has_location,
        'Observed FTP': observed_ftp,
    }


def process_activities(files, path, recalculate=False):
    """Run calculations on a set of activities.

    Calculates metrics on a set of FIT, TCX and/or GPX activities. Since
    processing a large number of activities can be slow, results are cached to a
    local file. If cached results are available, processing will be skipped and
    results from the cache will be returned instead.

    Args:
        files: List of FIT, TCX and/or GPX files to process.
        path: Path to directory containing the files.
        recalculate: Pass True to recalculate all results, pass string or
          iterable of file name(s) to recalculate only certain data files.

    Returns:
        A Pandas DataFrame of calculation results
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

    # Run calculations on all FIT files, leveraging cached results
    processor = functools.partial(process_one_activity, path=path, cache=cache)
    with multiprocessing.Pool() as p:
        calcs = pd.DataFrame(p.map(processor, files), index=files.index)

    # Add new results to cache, deduplicate and save
    new_cache = pd.concat([calcs.set_index('Filename'), cache])
    new_cache = new_cache[~new_cache.index.isna()]
    new_cache = new_cache[~new_cache.index.duplicated()]
    new_cache.sort_index().to_csv(get_cache_path())

    return calcs


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
    calcs = process_activities(csv['Filename'], path, recalculate)

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
