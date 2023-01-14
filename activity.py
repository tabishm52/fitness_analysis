import os
import hashlib

import multiprocessing
import functools

import pandas as pd
import numpy as np

import timezonefinder

from .parse_fit import parse_fit
from .parse_tcx_gpx import parse_tcx, parse_gpx


def get_cache_path():
    """Utility function to retrieve path to cached results"""
    return os.path.join(os.path.dirname(__file__), 'cache.csv')


def process_one_activity(fname, path, cache=None):
    """Run calculations on a single Strava activity"""

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

    # Strava's data export seems to gzip some but not all files
    root, ext = os.path.splitext(fname)
    if ext == '.gz':
        _, ext = os.path.splitext(root)

    # Parse data and identify relevant columns
    if ext == '.fit':
        records, _, _ = parse_fit(full_path)
        lat_col = 'position_lat'
        lng_col = 'position_long'
        power_col = 'power'
    elif ext == '.tcx':
        records, _, _ = parse_tcx(full_path)
        lat_col = 'LatitudeDegrees'
        lng_col = 'LongitudeDegrees'
        power_col = 'Watts'
    elif ext == '.gpx':
        records, _, _ = parse_gpx(full_path)
        lat_col = 'lat'
        lng_col = 'lon'
        power_col = 'pow' # Actually GPX files don't seem to have power
    else:
        raise RuntimeError(f"File type not supported for {fname}")

    # Determine timezone using first valid lat,lng position in the file
    try:
        tf = timezonefinder.TimezoneFinder()
        points = records[[lat_col, lng_col]]
        idx = points.apply(pd.Series.first_valid_index).max()
        lat, lng = points.loc[idx]
        tz = tf.timezone_at(lng=lng, lat=lat)
    except KeyError:
        # No valid lat,lng data in file
        tz = np.NaN

    # Estimate FTP by calculating maximum 20-minute effort in file
    try:
        ftp = np.max(
                records[power_col]
                .resample('S')
                .ffill()
                .rolling(20*60)
                .mean()
            )
    except KeyError:
        # No power data in file
        ftp = np.NaN

    return hash, tz, ftp


def process_activities(path, files, recalculate=False):
    """Run calculations on a set of Strava activities"""

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
