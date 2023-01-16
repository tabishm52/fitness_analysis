"""Functions for parsing FIT files into Pandas DataFrames."""

import gzip
import os

import pandas as pd

import fitdecode


def copy_fit_frames(fit_file):
    """Copies FIT data from a file-like object into a list of frames."""

    fit = fitdecode.FitReader(fit_file,
                              processor=fitdecode.StandardUnitsDataProcessor())
    return [f for f in fit if f.frame_type == fitdecode.FIT_FRAME_DATA and
                              f.mesg_type is not None]


def extract_fit_dicts(frames, names):
    """Yields dicts of frame data from FIT frames of given names."""

    for f in frames:
        if f.name in names:
            yield dict(
                (d.name, d.value)
                for d in f.fields
                if d.field is not None
            )


def parse_fit(file):
    """Loads a FIT activity into Pandas DataFrames.

    FIT data fields that are marked as 'unknown' by fitdecode are dropped during
    import. Assumes that the FIT file is all one activity, i.e. "chained" FIT
    files will be merged into one set of return values, possibly over-writing
    some fields.

    Args:
        file: File-like or path-like object. A path-like argument ending in
          '.gz' will be unzipped before processing.

    Returns:
        A tuple of (records, laps, extra)

        records: Time-indexed DataFrame of sensor data recorded during activity
        laps: DataFrame of lap information from the activity
        extra: Dict of selected additional information from the activity
    """

    try:
        _, ext = os.path.splitext(file)
        is_gzipped = True if ext.lower() == '.gz' else False
        is_path = True
    except TypeError:
        is_path = False

    if is_path:
        if is_gzipped:
            with gzip.open(file) as fit_file:
                frames = copy_fit_frames(fit_file)
        else:
            with open(file, 'rb') as fit_file:
                frames = copy_fit_frames(fit_file)
    else:
        frames = copy_fit_frames(file)

    # Note FIT files occasionally have duplicate timestamps, just drop those
    records = pd.DataFrame(extract_fit_dicts(frames, ['record']))
    records = records.set_index('timestamp')
    records = records[~records.index.duplicated()]

    laps = pd.DataFrame(extract_fit_dicts(frames, ['lap']))

    # This is a bit clumsy - if there is more than one frame of a given type or
    # fields with the same name across frame types, then you'll get whichever
    # value appears last in the FIT file
    extra = dict()
    extra_names = ['file_id', 'sport', 'session', 'activity']
    for d in extract_fit_dicts(frames, extra_names):
        extra.update(d)

    return records, laps, extra
