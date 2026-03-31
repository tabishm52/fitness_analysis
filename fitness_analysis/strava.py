"""Functions for processing Strava bicycling activities."""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import records, utils

ACTIVITIES_FNAME = "activities.csv"
ACTIVITIES_CACHE_FNAME = "activities_cache.csv"


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _cache_df_to_dict(cache: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Convert a cache DataFrame to a filename-keyed dict of result dicts."""

    return {
        filename: {
            "filename": filename,
            "timezone": row["timezone"],
            "has_location": row["has_location"],
            "max_heart_rate": row["max_heart_rate"],
            "estimated_ftp": row["estimated_ftp"],
        }
        for filename, row in cache.iterrows()
    }


def invalidate_activities_cache(
    files: Iterable[str] | None,
    cache_dir: str | PathLike[str],
) -> None:
    """Invalidate the activities cache.

    If ``files`` is None, deletes the entire cache file. Otherwise removes
    only the entries for the given activity filenames, leaving the rest intact.

    Args:
        files: Activity filenames to remove. If None, the whole cache is
            cleared.
        cache_dir: Cache directory passed to ``load_strava_activities``.
    """

    cache_path = Path(cache_dir) / ACTIVITIES_CACHE_FNAME
    if not cache_path.exists():
        return

    if files is None:
        cache_path.unlink()
        return

    cache = pd.read_csv(cache_path, index_col="filename")
    cache = cache[~cache.index.isin(list(files))]
    cache.sort_index().to_csv(cache_path)


# ---------------------------------------------------------------------------
# Activity processing
# ---------------------------------------------------------------------------


def process_activity_file(
    filename: str | float,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Parse a single activity file and compute metrics."""

    full_path = Path(path) / filename
    activity_records = records.parse_record_cached(full_path, cache_dir)

    timezone = utils.infer_timezone(activity_records)
    has_location = timezone is not None
    if timezone is None:
        timezone = np.nan

    if "heart_rate" in activity_records.columns:
        max_hr = activity_records["heart_rate"].max()
    else:
        max_hr = np.nan

    if "power" in activity_records.columns:
        # 20-minute mean power as a proxy for FTP
        estimated_ftp = (
            activity_records["power"]
            .resample("s")
            .ffill()
            .rolling(20 * 60)
            .mean()
            .max()
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

    Computes per-activity metrics. When ``cache_dir`` is provided, uses a
    two-level caching strategy:

    1. **Activities cache** (``ACTIVITIES_CACHE_FNAME``): stores computed
       metrics. Activities found here are returned immediately with no
       further work.
    2. **Records cache** (parquet files, one per activity): stores the parsed
       FIT/TCX/GPX records. Warmed via ``warm_records_cache``, which
       parallelizes parsing of cold files when the batch is large enough.

    Without a cache directory, all files are parsed on every call, with
    parallelization applied directly for large batches.

    Args:
        files: List of activity files to process.
        path: Directory containing the files.
        cache_dir: Optional directory for both caches. If omitted, no caching
            is performed and all files are parsed and processed on every call.

    Returns:
        Calculation results aligned to the index of ``files``.
    """

    if cache_dir is None:
        with ProcessPoolExecutor() as ex:
            results = ex.map(partial(process_activity_file, path=path), files)
        return pd.DataFrame(results, index=files.index)

    cache_path = Path(cache_dir) / ACTIVITIES_CACHE_FNAME
    cache_df = (
        pd.read_csv(cache_path, index_col="filename")
        if cache_path.exists()
        else None
    )
    cache = _cache_df_to_dict(cache_df) if cache_df is not None else {}

    no_file_result = {
        "timezone": np.nan,
        "has_location": False,
        "max_heart_rate": np.nan,
        "estimated_ftp": np.nan,
    }
    results = [
        {"filename": f, **no_file_result} if pd.isna(f) else cache.get(f)
        for f in files
    ]
    misses = [f for f, r in zip(files, results) if r is None]

    if not misses:
        return pd.DataFrame(results, index=files.index)

    records.warm_records_cache(misses, path, cache_dir)
    miss_map = dict(
        zip(misses, (process_activity_file(f, path, cache_dir) for f in misses))
    )

    calcs = pd.DataFrame(
        (r if r is not None else miss_map[f] for f, r in zip(files, results)),
        index=files.index,
    )

    new_rows = pd.DataFrame(miss_map.values()).set_index("filename")
    updated = (
        pd.concat([cache_df, new_rows]) if cache_df is not None else new_rows
    )
    updated.sort_index().to_csv(cache_path)

    return calcs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


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

    df = pd.DataFrame()
    df["date"] = calcs["local_date"]
    df["description"] = csv["Activity Name"]
    df["bicycle"] = csv["Activity Gear"]
    df["trainer"] = calcs["trainer"]
    df["commute"] = csv["Commute"]
    df["distance"] = csv["Distance"] * utils.KM_TO_MI
    df["elevation"] = csv["Elevation Gain"] * utils.M_TO_FT
    df["elapsed_time"] = pd.to_timedelta(csv["Elapsed Time"], unit="s")
    df["moving_time"] = pd.to_timedelta(csv["Moving Time"], unit="s")
    df["max_heart_rate"] = calcs["max_heart_rate"]
    df["estimated_ftp"] = calcs["estimated_ftp"]
    df["filename"] = csv["Filename"]

    activities = df.set_index("date").sort_index()

    weekly_metrics = ["distance", "elevation", "elapsed_time", "moving_time"]
    weekly_sums = activities[weekly_metrics].resample("W-SUN").sum()

    return activities, weekly_sums
