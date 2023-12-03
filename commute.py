"""Functions for processing Strava bicycling commutes."""

import multiprocessing
import os

import pandas as pd

from . import utils


def process_one_commute(activity, records):
    """Calculate summary metrics for one commute activity."""

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
    elapsed_time = (
        records.index[-1]
        - records.index[0]
    )

    # Try to look up speed, else calculate as smoothed derivative of distance
    try:
        speed = (
            records['speed']
            .resample('S')
            .interpolate()
        )
    except KeyError:
        speed = 3600.0 * (
            records['distance']
            .resample('S')
            .interpolate()
            .diff()
            .rolling(3)
            .mean()
        )

    # Moving time defined as whenever speed > 1 km per hour
    moving_time = pd.Timedelta(speed[speed > 1.0].count(), 's')

    return {
        'Date': date,
        'Description': activity['Description'],
        'Direction': direction,
        'Distance': distance,
        'Elapsed Time': elapsed_time,
        'Moving Time': moving_time,
        'Filename': activity['Filename'],
    }


def split_and_process_commutes(activities, path, delta):
    """Splits activities and yields summary metrics on each split.

    Generator function that splits the data from a set of activities whenever
    the time gap between two adjacent rows of data exceeds 'delta', and then
    yields commute summary metrics on each split.

    Args:
        activity: DataFrame of activity files to load and process.
        path: Path to Strava export directory.
        delta: Size of time gap at which to split activities.

    Yields:
        Dicts of summary metrics from each split.
    """

    # Loading FIT files is slow, so parallelize reading of files
    file_paths = (os.path.join(path, f) for f in activities['Filename'])
    with multiprocessing.Pool() as p:
        data = p.map(utils.parser.parse, file_paths)

    activity_rows = (row for _, row in activities.iterrows())
    commute_records = (records for records, _, _ in data)

    for activity, records in zip(activity_rows, commute_records):
        # Split records into separate groups wherever there is a difference
        # greater than 'delta' between the indices of two adjacent rows
        group_ids = (records.index.to_series().diff() > delta).cumsum()
        grouped = records.groupby(group_ids, sort=False)

        # Yield metrics on each split separately
        for _, group in grouped:
            yield process_one_commute(activity, group)


def load_commute_activities(activities, path, delta=pd.Timedelta(1, 'h')):
    """Calculate summary metrics for a set of commute activities.

    Splits commute activities into separate activities for each direction of
    the commute, and then calculates metrics on each commute direction.

    The activity files can be a mix of commutes recorded as separate one-way
    activities and commutes recorded as a single round-trip activity, with a
    pause of at least 'delta' between the two directions of the commute.

    Args:
        activity: DataFrame of activity files to load and process, usually a
          filtered list from load_strava_activities().
        path: Path to Strava export directory.

    Returns:
        DataFrame of commute activities with summary metrics.
    """

    results = split_and_process_commutes(activities, path, delta)
    return pd.DataFrame(results).set_index('Date')
