import os
import hashlib
import gzip

import multiprocessing
import functools

import pandas as pd
import numpy as np

import fitdecode

# A list of files where I happened to record janky power data
BAD_POWER_DATA = ['activities/3799945079.fit.gz']

def get_cache_path():
    """Utility function to retrieve path to cached results"""
    return os.path.join(os.path.dirname(__file__), 'cache.hdf')

def copy_fit_frames(fit_file):
    """Copy FIT data from a file-like object into a list of frames"""
    fit = fitdecode.FitReader(fit_file, processor=fitdecode.StandardUnitsDataProcessor())
    return [f for f in fit
            if f.frame_type == fitdecode.FIT_FRAME_DATA and f.mesg_type is not None]

def extract_fit_dicts(frames, name):
    """Extract FIT data (of a given name) from a list of frames into a generator of dicts"""
    return (dict((d.name, d.value) for d in f.fields if d.field is not None)
            for f in frames if f.name == name)

def load_fit_file(file, gzipped=False):
    """Load a FIT file into a DataFrame of records and dicts of session and activity data"""

    # Load the file
    if gzipped:
        with gzip.open(file) as fit_file:
            frames = copy_fit_frames(fit_file)
    else:
        with open(file) as fit_file:
            frames = copy_fit_frames(fit_file)

    # Compose records into a DataFrame - the actual time-series data from the FIT file
    # Note FIT files occasionally have duplicate timestamps, just drop those
    records = pd.DataFrame(extract_fit_dicts(frames, 'record')).set_index('timestamp', drop=True)
    records = records[~records.index.duplicated()]

    # 'session' and 'activity' are summary data, usually at the end of the file
    session = next(extract_fit_dicts(reversed(frames), 'session'))
    activity = next(extract_fit_dicts(reversed(frames), 'activity'))

    return records, session, activity

def process_fit_data(records, session, activity):
    """Run calculations on data from a FIT file"""

    # Calculate UTC offset if available in FIT file
    # Note some files have a nonsense local_timestamp, so throw those out
    try:
        offset = np.timedelta64(activity['local_timestamp'] - \
                 activity['timestamp'].replace(tzinfo=None))
        if np.abs(offset) > np.timedelta64(1, 'D'):
            offset = np.timedelta64(0)
    except KeyError:
        offset = np.timedelta64(0)

    # Calculate 20-min average power (FTP) if available in FIT file
    if 'power' in records.columns:
        ftp = np.max(records.power.resample('S').ffill().rolling(20*60).mean())
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

    # Strava's data export seems to gzip some but not all files
    root, ext = os.path.splitext(fname)
    if ext == '.gz':
        gzipped = True
        root, ext = os.path.splitext(root)
    else:
        gzipped = False

    # Return null data if this is not a FIT file (e.g. for GPX files)
    if ext == '.fit':
        records, session, activity = load_fit_file(full_path, gzipped)
        offset, ftp = process_fit_data(records, session, activity)
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
