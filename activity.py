"""Functions for calculating metrics on Strava bicycling activities."""

import functools
import hashlib
import multiprocessing
import os

import numpy as np
import pandas as pd
import timezonefinder

from . import parse_activity


activity_parser = parse_activity.ActivityParser()
tz_finder = timezonefinder.TimezoneFinder()


def get_cache_path():
    """Utility function to retrieve path to cached results."""

    return os.path.join(os.path.dirname(__file__), 'cache.csv')


def process_one_activity(fname, path, cache=None):
    """Run calculations on a single Strava activity."""

    # Calculate hash of file - to make sure cached results are valid
    full_path = os.path.join(path, fname)
    with open(full_path, 'rb') as f:
        h = hashlib.blake2b(f.read(), digest_size=8).hexdigest()

    # Short-circuit and return cached results if available
    try:
        if cache is not None and cache.loc[fname, 'Hash'] == h:
            return cache.loc[fname].to_numpy()
    except KeyError:
        pass

    # Load time-series data from the file
    records, _, _ = activity_parser.parse(full_path)

    # Determine timezone using first valid lat,lng position in the file
    try:
        points = records[['latitude', 'longitude']]
        idx = points.apply(pd.Series.first_valid_index).max()
        lat, lng = points.loc[idx]
        tz = tz_finder.timezone_at(lng=lng, lat=lat)
    except KeyError:
        # No valid lat,lng data in file
        tz = np.NaN

    # Estimate FTP by calculating maximum 20-minute effort in file
    try:
        ftp = (
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
        ftp = np.NaN

    return h, tz, ftp


def process_activities(files, path, recalculate=False):
    """Run calculations on a set of activities.

    Calculates metrics on a set of FIT, TCX and/or GPX activities. Since
    processing a large number of activities can be slow, results are cached to a
    local file. If cached results are available, processing will be skipped and
    results from the cache will be returned instead.

    Args:
        files: List of FIT, TCX and/or GPX files to process
        path: Path to directory containing the files
        recalculate: Pass True to recalculate all results, pass string or
          iterable to recalculate only certain data files

    Returns:
        A Pandas dataframe of calculation results
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
    calcs = pd.DataFrame()
    calcs['Filename'] = files
    processor = functools.partial(process_one_activity, path=path, cache=cache)
    with multiprocessing.Pool() as p:
        results = p.map(processor, calcs['Filename'])
    calcs['Hash'], calcs['Timezone'], calcs['Observed FTP'] = zip(*results)

    # Add new results to cache, deduplicate and save
    new_cache = pd.concat([calcs.set_index('Filename'), cache])
    new_cache = new_cache[~new_cache.index.duplicated()].sort_index()
    new_cache.to_csv(get_cache_path())

    return calcs
