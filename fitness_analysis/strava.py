"""Functions for processing Strava bicycling activities."""

from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import utils

ACTIVITIES_FNAME = "activities.csv"
ACTIVITIES_CACHE_FNAME = "activities_cache.csv"


def check_activities_cache(
    filename: str | float,
    cache: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Check whether a single activity is in the cache.

    No validation is performed — activity files are assumed immutable after
    Strava export. Returns the cached result dict on a hit, or None on a miss.
    NaN filenames (activities with no file) always return a result immediately.
    """

    if pd.isna(filename):
        return {
            "filename": filename,
            "timezone": np.nan,
            "has_location": False,
            "max_heart_rate": np.nan,
            "estimated_ftp": np.nan,
        }

    if cache is not None and filename in cache.index:
        cached = cache.loc[filename]
        return {
            "filename": filename,
            "timezone": cached["timezone"],
            "has_location": cached["has_location"],
            "max_heart_rate": cached["max_heart_rate"],
            "estimated_ftp": cached["estimated_ftp"],
        }

    return None


def process_one_activity(
    filename: str | float,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Parse a single activity file and compute metrics."""

    full_path = Path(path) / filename
    records = utils.parse_record_cached(full_path, cache_dir)

    timezone = utils.infer_timezone(records)
    has_location = timezone is not None
    if timezone is None:
        timezone = np.nan

    if "heart_rate" in records.columns:
        max_hr = records["heart_rate"].max()
    else:
        max_hr = np.nan

    if "power" in records.columns:
        estimated_ftp = (
            records["power"].resample("s").ffill().rolling(20 * 60).mean().max()
        )
    else:
        estimated_ftp = np.nan

    return {
        "filename": filename,
        "timezone": timezone,
        "has_location": has_location,
        "max_heart_rate": max_hr,
        "estimated_ftp": estimated_ftp,
    }


def process_activities(
    files: pd.Series,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Run calculations on a set of activities.

    Computes per-activity metrics (timezone, heart rate, estimated FTP) using a
    two-level caching strategy:

    1. **Activities cache** (``ACTIVITIES_CACHE_FNAME``): stores computed
       metrics. Activities found here are returned immediately with no
       further work.
    2. **Records cache** (parquet files, one per activity): stores the parsed
       FIT/TCX/GPX records. For activities not in the activities cache, cold
       files (no parquet yet) are parsed in parallel and written to the records
       cache first, then metrics are computed serially from the warm cache.

    Args:
        files: List of activity files to process.
        path: Directory containing the files.
        cache_dir: Optional directory for both caches. If omitted, no caching
            is performed and all files are parsed and processed on every call.

    Returns:
        Calculation results aligned to the index of ``files``.
    """

    cache_path = Path(cache_dir) / ACTIVITIES_CACHE_FNAME if cache_dir else None
    cache = None
    if cache_path and cache_path.exists():
        cache = pd.read_csv(cache_path).set_index("filename")

    results = [check_activities_cache(f, cache) for f in files]
    misses = [f for f, r in zip(files, results) if r is None]

    if misses:
        if cache_dir:
            utils.warm_records_cache(misses, path, cache_dir)

        miss_map = {f: process_one_activity(f, path, cache_dir) for f in misses}
        results = (
            r if r is not None else miss_map[f] for f, r in zip(files, results)
        )

    calcs = pd.DataFrame(results, index=files.index)

    if cache_path is not None:
        # Merge new results with existing cache, drop NaN-filename rows and dups
        if cache is not None:
            new_cache = pd.concat([calcs.set_index("filename"), cache])
        else:
            new_cache = calcs.set_index("filename")
        new_cache = new_cache[~new_cache.index.isna()]
        new_cache = new_cache[~new_cache.index.duplicated()]
        new_cache.sort_index().to_csv(cache_path)

    return calcs


def invalidate_activities_cache(
    cache_dir: str | PathLike[str],
    files: Iterable[str] | None = None,
) -> None:
    """Invalidate the activities cache.

    If ``files`` is None, deletes the entire cache file. Otherwise removes
    only the entries for the given activity filenames, leaving the rest intact.

    Args:
        cache_dir: Cache directory passed to ``load_strava_activities``.
        files: Activity filenames to remove. If None, the whole cache is
            cleared.
    """

    cache_path = Path(cache_dir) / ACTIVITIES_CACHE_FNAME
    if not cache_path.exists():
        return
    if files is None:
        cache_path.unlink()
        return
    cache = pd.read_csv(cache_path).set_index("filename")
    cache = cache.drop(index=[f for f in files if f in cache.index])
    cache.sort_index().to_csv(cache_path)


def load_strava_activities_raw(
    path: str | PathLike[str],
) -> pd.DataFrame:
    """Load raw bicycling activity data from a Strava export CSV.

    Reads ``ACTIVITIES_FNAME`` and filters to Ride and Virtual Ride activities
    without computing any additional metrics from the underlying activity files.

    Args:
        path: Strava export directory.

    Returns:
        Raw activity data indexed by UTC activity date.
    """

    csv = pd.read_csv(Path(path) / ACTIVITIES_FNAME).query(
        '`Activity Type` in ["Ride", "Virtual Ride"]'
    )
    csv["Activity Date"] = pd.to_datetime(
        csv["Activity Date"], format="%b %d, %Y, %I:%M:%S %p"
    )
    return csv.set_index("Activity Date")


def load_strava_activities(
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load bicycling activity data from a Strava export directory.

    Reads ``ACTIVITIES_FNAME``, filters to Ride and Virtual Ride activities,
    and computes additional per-activity metrics (timezone, max heart rate,
    estimated FTP) from the underlying activity files. Results are cached so
    that subsequent calls are fast even for large exports.

    Args:
        path: Strava export directory.
        home_tz: Fallback timezone for activities without GPS location data
            (e.g. trainer rides). Either a fixed timezone string or a callable
            that accepts a Series and returns per-activity timezone values.
        cache_dir: Optional directory for cached results. If omitted, activity
            files are parsed on every call.

    Returns:
        Tuple of (activities, weekly_sums). ``activities`` contains one row per
        ride with distance, elevation, time, heart rate, and FTP metrics indexed
        by local date. ``weekly_sums`` contains distance, elevation, and time
        totals resampled to weekly (Sunday) buckets.
    """

    csv = load_strava_activities_raw(path)
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
    df["estimated_ftp"] = calcs["estimated_ftp"]  # In watts
    df["filename"] = csv["Filename"]

    activities = df.set_index("date").sort_index()

    weekly_metrics = ["distance", "elevation", "elapsed_time", "moving_time"]
    weekly_sums = activities[weekly_metrics].resample("W-SUN").sum()

    return activities, weekly_sums
