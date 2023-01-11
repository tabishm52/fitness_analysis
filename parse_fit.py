import pandas as pd
import numpy as np

import gzip

import fitdecode

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
