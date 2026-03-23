"""Functions for processing Strava bicycling commutes."""

import multiprocessing
import os
from collections.abc import Iterator
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd

from . import utils

_NOON = 12


def process_one_commute(
    activity: pd.Series,
    records: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate summary metrics for one commute activity."""

    # Compute the date of the activity in local time
    timezone = utils.infer_timezone(records)
    if timezone is None:
        date = records.index[0].tz_localize(None)
    else:
        date = records.index[0].tz_convert(timezone).tz_localize(None)

    # Mark morning vs afternoon activities
    if date.hour < _NOON:
        direction = "Morning"
    else:
        direction = "Afternoon"

    # Elapsed time is the difference between the first and last timestamps
    elapsed_time = records.index[-1] - records.index[0]

    # Some activities (e.g., GPX files) do not contain 'distance' information
    if "distance" not in records.columns:
        distance = np.nan
        moving_time = np.nan

    else:
        # Determine total distance of activity in miles
        distance = 0.6213712 * (
            records["distance"].max() - records["distance"].min()
        )

        # Calculate moving time by excluding periods where speed is less than
        # 1 km per hour for at least 10 seconds
        inactive_periods = utils.identify_inactive_periods(
            records["distance"], 1.0 / 3600.0, pd.Timedelta(10, "s")
        )
        moving_time = pd.Timedelta((~inactive_periods).sum(), "s")

    return {
        "Date": date,
        "Description": activity["Description"],
        "Direction": direction,
        "Distance": distance,
        "Elapsed Time": elapsed_time,
        "Moving Time": moving_time,
        "Filename": activity["Filename"],
    }


def split_and_process_commutes(
    activities: pd.DataFrame,
    path: str | PathLike[str],
    delta: pd.Timedelta,
) -> Iterator[dict[str, Any]]:
    """Splits activities and yields summary metrics on each split.

    Generator function that splits the data from a set of activities whenever
    the time gap between two adjacent rows of data exceeds 'delta', and then
    yields commute summary metrics on each split.

    Args:
        activities: Activity files to load and process.
        path: Strava export directory.
        delta: Size of time gap at which to split activities.

    Yields:
        Records with summary metrics from each split.
    """

    # Loading FIT files is slow, so parallelize reading of files
    file_names = activities["Filename"][activities["Filename"].notna()]
    file_paths = (os.path.join(path, f) for f in file_names)
    with multiprocessing.Pool() as p:
        data = p.map(utils.parser.parse, file_paths)

    activity_rows = (row for _, row in activities.iterrows())
    commute_records = (records for records, _, _ in data)

    for activity, all_records in zip(activity_rows, commute_records):
        # Drop periods of inactivity, to cover the cases where the GPS was left
        # on all day rather than being paused between commute segments
        try:
            inactive_periods = utils.identify_inactive_periods(
                all_records["distance"], 2.5 / 3600.0, delta
            )
            active_periods = (
                (~inactive_periods).reindex(all_records.index).fillna(True)
            )
            records = all_records[active_periods]
        except KeyError:
            records = all_records

        # Split records into separate groups wherever there is a difference
        # greater than 'delta' between the indices of two adjacent rows
        group_ids = (records.index.to_series().diff() > delta).cumsum()
        grouped = records.groupby(group_ids, sort=False)

        # Yield metrics on each split separately
        for _, group in grouped:
            yield process_one_commute(activity, group)


def load_commute_activities(
    activities: pd.DataFrame,
    path: str | PathLike[str],
    delta: pd.Timedelta = pd.Timedelta(90, "m"),
) -> pd.DataFrame:
    """Calculate summary metrics for a set of commute activities.

    Identifies commute activities, splits commutes into separate activities for
    each direction of the commute, and then calculates metrics on each commute
    direction.

    The activity files can be a mix of commutes recorded as separate one-way
    activities and commutes recorded as a single round-trip activity, with a
    pause of at least 'delta' between the two directions of the commute.

    Args:
        activities: Activity files to load and process.
        path: Strava export directory.
        delta: Size of time gap at which to split activities.

    Returns:
        Summary metrics from commute activities indexed by local Date.
    """

    commutes = activities.loc[activities["Commute"]]
    results = split_and_process_commutes(commutes, path, delta)
    data = pd.DataFrame(results)

    if data.empty:
        columns = [
            "Description",
            "Direction",
            "Distance",
            "Elapsed Time",
            "Moving Time",
            "Filename",
        ]
        return pd.DataFrame(columns=columns, index=pd.Index([], name="Date"))

    return data.set_index("Date")
