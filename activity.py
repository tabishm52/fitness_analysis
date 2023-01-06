import os
import hashlib
import gzip

import multiprocessing
import functools

import pandas as pd
import numpy as np

import fitanalysis

# A list of files where I happened to record janky power data
BAD_POWER_DATA = ['activities/3799945079.fit.gz']

def get_cache_path():
    """Utility function to retrieve path to cached results"""
    return os.path.join(os.path.dirname(__file__), 'cache.hdf')

def process_fit_file(file):
    """Run calculations on a single FIT file"""

    # Load the FIT file for analysis
    # Note: This is some rando code from github and is super slow
    try:
        fitfile = fitanalysis.Activity(file)
    except TypeError:
        # fitanalysis seems to puke on non-Garmin FIT files
        return np.timedelta64(0), np.NaN

    # Calculate UTC offset if available in FIT file
    try:
        m = next(fitfile.get_messages('activity'))
        offset = np.timedelta64(m.get('local_timestamp').value - m.get('timestamp').value)
    except AttributeError:
        offset = np.timedelta64(0)

    # Calculate 20-min average power (FTP) if available in FIT file
    if fitfile.has_power:
        p = fitfile.power.droplevel(level='block')
        ftp = np.max(fitanalysis.util.moving_average(p, 20*60))
    else:
        ftp = np.NaN

    return offset, ftp

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

    # Strava seems to gzip some but not all files; run gunzip if needed
    root, ext = os.path.splitext(fname)
    if ext == '.gz':
        file = gzip.open(full_path)
        root, ext = os.path.splitext(root)
    else:
        file = full_path

    # Return null data if this is not a FIT file (e.g. for GPX files)
    if ext == '.fit':
        offset, ftp = process_fit_file(file)
    else:
        offset = np.timedelta64(0)
        ftp = np.NaN

    # Drop FTP if in list of bad files
    if fname in BAD_POWER_DATA:
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
