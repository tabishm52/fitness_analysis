import os
import hashlib

import multiprocessing
import functools

import pandas as pd
import numpy as np

from .parse_fit import parse_fit
from .parse_tcx_gpx import parse_tcx

def get_cache_path():
    """Utility function to retrieve path to cached results"""
    return os.path.join(os.path.dirname(__file__), 'cache.hdf')

def process_fit_data(points, laps, extra):
    """Run calculations on data from a FIT file"""

    # Calculate UTC offset if available in FIT file
    # Note some files have a nonsense local_timestamp, so throw those out
    try:
        offset = np.timedelta64(extra['local_timestamp'] - \
                 extra['timestamp'].replace(tzinfo=None))
        if np.abs(offset) > np.timedelta64(1, 'D'):
            offset = np.timedelta64(0)
    except KeyError:
        offset = np.timedelta64(0)

    # Calculate 20-min average power (FTP) if available in FIT file
    if 'power' in points.columns:
        ftp = np.max(points.power.resample('S').ffill().rolling(20*60).mean())
    else:
        ftp = np.NaN

    return offset, ftp

def process_tcx_data(points, laps, extra):
    """Run calculations on data from a TCX file"""

    # Calculate 20-min average power (FTP) if available in TCX file
    if 'Watts' in points.columns:
        ftp = np.max(points.Watts.resample('S').ffill().rolling(20*60).mean())
    else:
        ftp = np.NaN

    return np.timedelta64(0), ftp

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

    # Return null data if this is not a FIT file (e.g. for GPX files)
    if ext == '.fit':
        points, laps, extra = parse_fit(full_path)
        offset, ftp = process_fit_data(points, laps, extra)
    elif ext == '.tcx':
        points, laps, extra = parse_tcx(full_path)
        offset, ftp = process_tcx_data(points, laps, extra)
    else:
        offset = np.timedelta64(0)
        ftp = np.NaN

    return hash, offset, ftp

def process_activities(path, files, recalculate=False):
    """Run calculations on a set of Strava activities"""

    # Load cached calculation results if available
    try:
        cache = pd.read_hdf(get_cache_path(), 'strava_calcs')
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
    calcs['Hash'], calcs['UTC Offset'], calcs['Observed FTP'] = zip(*results)

    # Add new results to cache, deduplicate and save
    new_cache = pd.concat([calcs.set_index('Filename', drop=True), cache])
    new_cache = new_cache[~new_cache.index.duplicated()].sort_index()
    new_cache.to_hdf(get_cache_path(), 'strava_calcs', mode='w')

    return calcs
