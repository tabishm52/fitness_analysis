"""Functions for processing Strava bicycling activities."""

import functools
import hashlib
import multiprocessing
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import utils

CACHE_FNAME = "bicycle_cache.csv"


def process_one_activity(
    filename: str | float,
    path: str | PathLike[str],
    cache: pd.DataFrame | None,
) -> dict[str, Any]:
    """Run calculations on a single activity."""

    # Manual activities in Strava don't have a related activity file
    if pd.isna(filename):
        return {
            "filename": filename,
            "hash": np.nan,
            "timezone": np.nan,
            "has_location": False,
            "max_heart_rate": np.nan,
            "max_avg_power": np.nan,
        }

    # Calculate hash of file - to make sure cached results are valid
    full_path = Path(path) / filename
    with open(full_path, "rb") as f:
        hash_val = hashlib.blake2b(f.read(), digest_size=8).hexdigest()

    # Short-circuit and return cached results if available
    try:
        if cache is not None and cache.at[filename, "hash"] == hash_val:
            return {
                "filename": filename,
                "hash": hash_val,
                "timezone": cache.at[filename, "timezone"],
                "has_location": cache.at[filename, "has_location"],
                "max_heart_rate": cache.at[filename, "max_heart_rate"],
                "max_avg_power": cache.at[filename, "max_avg_power"],
            }
    except KeyError:
        pass

    # Load time-series data from the file
    records, _, _ = utils.parser.parse(full_path)

    # Determine timezone using the first valid latitude/longitude position.
    timezone = utils.infer_timezone(records)
    has_location = timezone is not None
    if timezone is None:
        timezone = np.nan

    # Calculate maximum heart rate observed during activity
    try:
        max_hr = records["heart_rate"].max()
    except KeyError:
        # No heart rate data in file
        max_hr = np.nan

    # Calculate maximum average power over 20 minutes during activity
    try:
        max_avg_power = (
            records["power"].resample("s").ffill().rolling(20 * 60).mean().max()
        )
    except KeyError:
        # No power data in file
        max_avg_power = np.nan

    return {
        "filename": filename,
        "hash": hash_val,
        "timezone": timezone,
        "has_location": has_location,
        "max_heart_rate": max_hr,
        "max_avg_power": max_avg_power,
    }


def process_activities(
    files: pd.Series,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Run calculations on a set of activities.

    Calculates metrics on a set of activity files. Loading of activities in FIT
    format can be slow, so results are cached to a local file. If cached results
    are available, processing will be skipped and results from the cache will be
    returned instead.

    Args:
        files: List of activity files to process.
        path: Directory containing the files.
        cache_dir: Optional directory for cache of results.

    Returns:
        Calculation results aligned to the index of ``files``.
    """

    # Load cached calculation results if available
    if cache_dir is not None:
        cache_path = Path(cache_dir) / CACHE_FNAME
        try:
            cache = pd.read_csv(cache_path).set_index("filename")
        except FileNotFoundError:
            cache = None
    else:
        cache = None

    # Run calculations on all activity files, leveraging cached results
    processor = functools.partial(process_one_activity, path=path, cache=cache)
    with multiprocessing.Pool() as p:
        calcs = pd.DataFrame(p.map(processor, files), index=files.index)

    if cache_dir is not None:
        # Add new results to cache, deduplicate and save
        if cache is not None:
            new_cache = pd.concat([calcs.set_index("filename"), cache])
        else:
            new_cache = calcs.set_index("filename")
        new_cache = new_cache[~new_cache.index.isna()]
        new_cache = new_cache[~new_cache.index.duplicated()]
        new_cache.sort_index().to_csv(cache_path)

    return calcs


def load_strava_activities(
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads bicycling activity data from a Strava data export.

    In addition to loading Strava's activities.csv, calculates certain
    additional metrics from the underlying activity files. Since parsing all
    activity files takes some time, results are cached to a local file. If
    cached results are available, processing will be skipped and results from
    the cache will be returned instead.

    Args:
        path: Strava export directory.
        home_tz: Fallback timezone used when location data are unavailable
            (for example, trainer rides). Accepts either a single timezone
            value or a callable that returns timezone values per activity.
        cache_dir: Optional location for cache of results.

    Returns:
        Tuple of (activities, weekly_sums) where activities is the processed
        Strava bicycling activity data and weekly_sums is a DataFrame of
        distance, elevation, and time metrics resampled to weekly totals.
    """

    # Load activities.csv and filter out any non-bicycle activities
    csv = pd.read_csv(Path(path) / "activities.csv").query(
        '`Activity Type` in ["Ride", "Virtual Ride"]'
    )

    # Set the UTC date and time of the activity as the index
    csv["Activity Date"] = pd.to_datetime(
        csv["Activity Date"], format="%b %d, %Y, %I:%M:%S %p"
    )
    csv.set_index("Activity Date", inplace=True)

    # Run a set of calculations on all activity files
    calcs = process_activities(csv["Filename"], path, cache_dir)

    # Infer activities that were performed on a stationary trainer
    calcs["trainer"] = (csv["Activity Type"] == "Virtual Ride") | (
        ~calcs["has_location"] & ~csv["Filename"].isna()
    )

    # Calculate the local date and time for each activity, subbing in a default
    # timezone for trainer rides or if timezone info is not available
    mask = calcs["trainer"] | calcs["timezone"].isna()
    calcs["timezone_used"] = calcs["timezone"].mask(mask, home_tz)
    calcs["local_date"] = [
        date.tz_localize("UTC").tz_convert(tz).tz_localize(None)
        for date, tz in zip(calcs.index, calcs["timezone_used"])
    ]

    # Construct the return DataFrame, converting units as appropriate
    df = pd.DataFrame()
    df["date"] = calcs["local_date"]
    df["description"] = csv["Activity Name"]
    df["bicycle"] = csv["Activity Gear"]
    df["trainer"] = calcs["trainer"]
    df["commute"] = csv["Commute"]
    df["distance"] = csv["Distance"] * utils.KM_TO_MI
    df["elevation"] = csv["Elevation Gain"] * utils.M_TO_FT
    df["elapsed_time"] = csv["Elapsed Time"]  # In seconds
    df["moving_time"] = csv["Moving Time"]  # In seconds
    df["max_heart_rate"] = calcs["max_heart_rate"]  # In beats per minute
    df["estimated_ftp"] = calcs["max_avg_power"]  # In watts
    df["filename"] = csv["Filename"]

    activities = df.set_index("date").sort_index()

    weekly_metrics = ["distance", "elevation", "elapsed_time", "moving_time"]
    weekly_sums = activities[weekly_metrics].resample("W-SUN").sum()

    return activities, weekly_sums
