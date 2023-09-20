"""Functions for processing Strava bicycling commutes."""

import multiprocessing
import os

import pandas as pd

from . import utils


def load_and_split_activities(files, path, delta):
    """Splits a set of activities and yields record data from each activity.

    Generator function that loads data from a list of activity files and then
    splits the data from each activity whenever the time gap between two
    adjacent rows of data exceeds 'delta'.

    Args:
        files: List of activity files to process.
        path: Path to directory containing the files.
        delta: Size of time gap at which to split activities.

    Yields:
        DataFrames of sensor records from each (potentially split) activity.
    """

    # Loading FIT files is slow, so parallelize reading of files
    file_paths = (os.path.join(path, filename) for filename in files)
    with multiprocessing.Pool() as p:
        data = p.map(utils.parser.parse, file_paths)

    for records, _, _ in data:
        # Split records into separate groups wherever there is a difference
        # greater than 'delta' between the indices of two adjacent rows
        group_ids = (records.index.to_series().diff() > delta).cumsum()
        grouped = records.groupby(group_ids, sort=False)

        # Yield each group separately
        for _, group in grouped:
            yield group


def process_one_commute(records):
    """Calculate summary metrics for one commute activity"""

    # Compute the date of the activity in local time
    timezone = utils.infer_timezone(records)
    date = records.index[0].tz_convert(timezone).tz_localize(None)

    # Mark morning vs afternoon activities
    if date.hour < 12:
        direction = 'Morning'
    else:
        direction = 'Afternoon'

    # Determine total distance of activity in miles
    distance = 0.6213712 * (
        records['distance'].max()
        - records['distance'].min()
    )

    # Elapsed time is the difference between the first and last timestamps
    elapsed_time = records.index[-1] - records.index[0]

    # Either look up speed, or calculate it as derivative of distance
    # The derivative is super noisy but good enough to calculate moving time
    try:
        speed = records['speed'].resample('S').interpolate()
    except KeyError:
        speed = records['distance'].resample('S').interpolate().diff() * 3600.0

    # Moving time defined as whenever speed > 1 mile per hour
    moving_time = pd.Timedelta(speed[speed > 1.609344].count(), 's')

    return {
        'Date': date,
        'Direction': direction,
        'Distance': distance,
        'Elapsed Time': elapsed_time,
        'Moving Time': moving_time,
    }


def load_process_commutes(files, path):
    """Load and calculate summary metrics for a set of commute activities.

    Splits commute activities into separate activities for each direction of the
    commute, and then calculates metrics on each commute direction. The activity
    files can be a mix of commutes recorded as separate one-way activities and
    commutes recorded as a single round-trip activity, with a pause of at least
    one hour between the two directions of the commute.

    Args:
        files: List of activity files to process.
        path: Path to directory containing the files.

    Returns:
        DataFrame of commutes with related activity metrics.
    """

    records = load_and_split_activities(files, path, pd.Timedelta(1, 'h'))
    return pd.DataFrame(map(process_one_commute, records)).set_index('Date')
