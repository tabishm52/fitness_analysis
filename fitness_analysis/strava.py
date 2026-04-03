"""Functions for processing Strava bicycling activities."""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import records, routes, utils
from .routes import ROUTE_CLUSTER_CONFIG_RAW, RouteClusterConfig

ACTIVITIES_FNAME = "activities.csv"
ACTIVITIES_CACHE_FNAME = "activities_cache.csv"


@dataclass
class ActivitiesConfig:
    """Configuration parameters for ``load_strava_activities``.

    Attributes:
        ftp_window_s: Rolling window in seconds for FTP estimation from power
            data.
        ftp_factor: Fraction of the rolling-window mean power used as the FTP
            estimate. The standard 20-minute protocol uses 0.95.
        weekly_anchor: Pandas offset alias for weekly resampling. ``'W-SUN'``
            matches Strava's weekly metrics.
        clustering: Route clustering parameters. If None, ``cluster_id`` and
            ``cluster_name`` columns are not added to returned activities.
    """

    ftp_window_s: int = 20 * 60
    ftp_factor: float = 0.95
    weekly_anchor: str = "W-SUN"
    clustering: RouteClusterConfig | None = field(
        default_factory=lambda: ROUTE_CLUSTER_CONFIG_RAW
    )


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


def parse_activity_file(
    filename: str | float,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: ActivitiesConfig,
) -> dict[str, Any]:
    """Parse a single activity file and compute metrics."""

    activity_records = records.parse_record_cached(filename, path, cache_dir)

    timezone = utils.infer_timezone(activity_records)
    has_location = timezone is not None
    if timezone is None:
        timezone = np.nan

    if "heart_rate" in activity_records.columns:
        max_hr = activity_records["heart_rate"].max()
    else:
        max_hr = np.nan

    if "power" in activity_records.columns:
        estimated_ftp = (
            activity_records["power"]
            .resample("s")
            .ffill()
            .rolling(config.ftp_window_s)
            .mean()
            .max()
        ) * config.ftp_factor
    else:
        estimated_ftp = np.nan

    return {
        "filename": filename,
        "timezone": timezone,
        "has_location": has_location,
        "max_heart_rate": max_hr,
        "estimated_ftp": estimated_ftp,
    }


def load_file_metrics(
    files: pd.Series,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: ActivitiesConfig,
    cache_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Load per-activity metrics, using a pre-loaded cache when provided.

    Args:
        files: Activity filenames aligned to the CSV index.
        path: Strava export directory.
        cache_dir: Records parquet cache directory.
        config: Activities configuration.
        cache_df: Already-loaded activities cache indexed by filename. Pass
            None to skip cache lookup (all files are processed).

    Returns:
        Tuple of:
        - ``results``: Metrics DataFrame aligned to ``files.index``.
        - ``miss_map``: Newly computed rows keyed by filename. Empty when
          every entry was a cache hit or had no file.
    """

    if cache_df is None:
        with ProcessPoolExecutor() as ex:
            results = list(
                ex.map(
                    partial(
                        parse_activity_file,
                        path=path,
                        cache_dir=None,
                        config=config,
                    ),
                    files,
                )
            )
        return pd.DataFrame(results, index=files.index), {}

    cached = _cache_df_to_dict(cache_df) if not cache_df.empty else {}

    no_file_result = {
        "timezone": np.nan,
        "has_location": False,
        "max_heart_rate": np.nan,
        "estimated_ftp": np.nan,
    }
    rows = [
        {"filename": f, **no_file_result} if pd.isna(f) else cached.get(f)
        for f in files
    ]
    misses = [f for f, r in zip(files, rows) if r is None]

    if misses:
        records.warm_records_cache(misses, path, cache_dir)
        miss_map = {
            f: parse_activity_file(f, path, cache_dir, config) for f in misses
        }
        rows = [
            r if r is not None else miss_map[f] for f, r in zip(files, rows)
        ]
    else:
        miss_map = {}

    return pd.DataFrame(rows, index=files.index), miss_map


def build_activity_columns(
    csv: pd.DataFrame,
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None,
    config: ActivitiesConfig,
) -> pd.DataFrame:
    """Compute all derived per-activity columns with a cache lookup.

    Reads the activities cache, computes file metrics (timezone, heart rate,
    FTP), infers trainer status, calculates local dates, and optionally runs
    route clustering. Writes updated cache at the end.

    Args:
        csv: Raw Strava CSV indexed by UTC activity date.
        path: Strava export directory.
        home_tz: Fallback timezone for activities without GPS data.
        cache_dir: Optional cache directory. If None, no caching is performed.
        config: Activities configuration.

    Returns:
        Computed columns aligned to the index of ``csv``.
    """

    cache_path = Path(cache_dir) / ACTIVITIES_CACHE_FNAME if cache_dir else None
    if cache_path is None:
        cache_df = None  # no caching at all
    elif cache_path.exists():
        cache_df = pd.read_csv(cache_path, index_col="filename")
        cache_df = cache_df[cache_df.index.notna()]
    else:
        cache_df = pd.DataFrame()  # cache dir set but file not yet created

    calcs, miss_map = load_file_metrics(
        csv["Filename"], path, cache_dir, config, cache_df
    )

    calcs["trainer"] = (csv["Activity Type"] == "Virtual Ride") | (
        ~calcs["has_location"] & ~csv["Filename"].isna()
    )

    mask = calcs["trainer"] | calcs["timezone"].isna()
    calcs["timezone_used"] = calcs["timezone"].mask(mask, home_tz)
    calcs["local_date"] = [
        date.tz_localize("UTC").tz_convert(tz).tz_localize(None)
        for date, tz in zip(calcs.index, calcs["timezone_used"])
    ]

    cluster_miss = False
    if config.clustering is not None:
        clusters, cluster_miss = routes.cluster_routes_cached(
            csv,
            path,
            cache_dir,
            cache_df=(
                cache_df
                if cache_df is not None and not cache_df.empty
                else None
            ),
            config=config.clustering,
        )
        calcs["cluster_id"] = clusters["cluster_id"]
        calcs["cluster_name"] = clusters["cluster_name"]

    if cache_path is not None and (miss_map or cluster_miss):
        if miss_map:
            new_rows = pd.DataFrame(miss_map.values()).set_index("filename")
            updated = (
                pd.concat([cache_df, new_rows])
                if cache_df is not None and not cache_df.empty
                else new_rows
            )
        else:
            updated = cache_df.copy()

        if cluster_miss:
            fn = csv["Filename"].dropna()
            fn = fn[~fn.duplicated()]
            updated["cluster_id"] = pd.Series(
                calcs.loc[fn.index, "cluster_id"].values, index=fn.values
            )
            updated["cluster_name"] = pd.Series(
                calcs.loc[fn.index, "cluster_name"].values, index=fn.values
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
    config: ActivitiesConfig | None = None,
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
        config: Optional configuration. Defaults to ``ActivitiesConfig()``.

    Returns:
        Tuple of:
        - ``activities``: One row per ride with distance, elevation, time,
          heart rate, and FTP metrics indexed by local date.
        - ``weekly_sums``: Distance, elevation, and time totals resampled to
          weekly (Sunday) buckets.
    """

    if config is None:
        config = ActivitiesConfig()

    csv = load_strava_activities_raw(path)
    calcs = build_activity_columns(csv, path, home_tz, cache_dir, config)

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
    if config.clustering is not None:
        df["cluster_id"] = calcs["cluster_id"]
        df["cluster_name"] = calcs["cluster_name"]
    df["filename"] = csv["Filename"]

    activities = df.set_index("date").sort_index()

    weekly_metrics = ["distance", "elevation", "elapsed_time", "moving_time"]
    weekly_sums = (
        activities[weekly_metrics].resample(config.weekly_anchor).sum()
    )

    return activities, weekly_sums
