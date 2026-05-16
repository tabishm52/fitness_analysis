"""Functions for processing Strava bicycling activities."""

import contextlib
import dataclasses
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite_utils

from . import cache_db, records, routes, utils

ACTIVITIES_FNAME = "activities.csv"


@dataclass
class ActivitiesConfig:
    """Configuration parameters for ``load_strava_activities``.

    Attributes:
        power_curve_max_s: Upper bound window size in seconds for power curves.
        power_curve_density: Logspace samples per decade for power curve
            windows.
        ftp_window_s: Rolling window in seconds for FTP estimation from power
            data.
        ftp_factor: Fraction of the rolling-window mean power used as the FTP
            estimate. The standard 20-minute protocol uses 0.95.
        weekly_anchor: Pandas offset alias for weekly resampling. ``'W-SUN'``
            matches Strava's weekly metrics.
        clustering: Route clustering parameters. If None, ``cluster_id`` and
            ``cluster_name`` columns are not added to returned activities.
    """

    power_curve_max_s: int = 12 * 3600
    power_curve_density: int = 50
    ftp_window_s: int = 20 * 60
    ftp_factor: float = 0.95
    weekly_anchor: str = "W-SUN"
    clustering: routes.RouteClusterConfig | None = dataclasses.field(
        default_factory=routes.RouteClusterConfig
    )

    def __post_init__(self) -> None:
        if self.clustering is not None:
            self.clustering.raw_csv = True


@dataclass
class ActivityMetrics:
    """Per-activity metrics computed from a single activity file.

    Attributes:
        filename: Activity filename, or NaN for fileless activities.
        timezone: IANA timezone inferred from GPS data, or ``None`` when absent.
        has_location: Whether the activity has GPS location data.
        max_heart_rate: Maximum heart rate in bpm, or ``None``.
        estimated_ftp: Estimated FTP in watts, or ``None``.
    """

    filename: str | float
    timezone: str | None = None
    has_location: bool = False
    max_heart_rate: float | None = None
    power_windows: np.ndarray | None = None
    power_curve: np.ndarray | None = None
    estimated_ftp: float | None = None

    def __post_init__(self) -> None:
        """Coerce numpy scalars to Python types; cast has_location to bool."""
        self.has_location = bool(self.has_location)
        self.max_heart_rate = cache_db.to_sql(self.max_heart_rate)
        self.estimated_ftp = cache_db.to_sql(self.estimated_ftp)

    def to_db_dict(self) -> dict:
        """Return a dict suitable for ``db["activities"].upsert()``."""
        return {
            "filename": self.filename,
            "segment": -1,
            "timezone": self.timezone,
            "has_location": int(self.has_location),
            "max_heart_rate": self.max_heart_rate,
            "power_windows": (
                json.dumps(self.power_windows.tolist())
                if self.power_windows is not None
                else None
            ),
            "power_curve": (
                json.dumps(self.power_curve.tolist())
                if self.power_curve is not None
                else None
            ),
            "estimated_ftp": self.estimated_ftp,
        }

    @classmethod
    def from_db_dict(cls, row: dict) -> ActivityMetrics:
        """Construct from a ``db["activities"].rows_where()`` row dict."""
        excluded = {"power_windows", "power_curve"}
        field_names = {f.name for f in dataclasses.fields(cls)} - excluded
        obj = cls(**{name: row.get(name) for name in field_names})

        if row.get("power_windows") is not None:
            obj.power_windows = np.array(json.loads(row["power_windows"]))
            obj.power_curve = np.array(json.loads(row["power_curve"]))

        return obj


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def load_activities_cache(
    cache_dir: str | PathLike[str],
) -> dict[str, ActivityMetrics]:
    """Read the activities cache from the database.

    Args:
        cache_dir: Cache directory containing the SQLite database.

    Returns:
        Dict mapping each activity filename to its cached metrics.
    """
    with cache_db.open_db(cache_dir) as db:
        return {
            row["filename"]: ActivityMetrics.from_db_dict(row)
            for row in db["activities"].rows
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
    if not cache_db.db_path(cache_dir).exists():
        return

    with cache_db.open_db(cache_dir) as db:
        if files is None:
            with db.conn:
                db["activities"].drop()
                db["cluster_fingerprints"].delete_where(
                    "table_name = ?", ["activities"]
                )
        else:
            files_list = list(files)
            marks = ",".join("?" * len(files_list))
            with db.conn:
                db["activities"].delete_where(
                    f"filename IN ({marks})", files_list
                )
                db["cluster_fingerprints"].delete_where(
                    "table_name = ?", ["activities"]
                )


# ---------------------------------------------------------------------------
# Activity processing
# ---------------------------------------------------------------------------


def parse_activity_file(
    filename: str | float,
    activity_records: pd.DataFrame,
    config: ActivitiesConfig,
    db: sqlite_utils.Database | None = None,
) -> ActivityMetrics:
    """Compute metrics from a pre-loaded activity records DataFrame.

    Args:
        filename: Activity filename, or NaN for activities without a file.
        activity_records: Parsed records DataFrame for the activity.
        config: Activities configuration.
        db: Optional open database. When provided, the computed metrics are
            inserted into the cache as each file is parsed.

    Returns:
        Computed metrics for the activity.
    """
    timezone = utils.infer_timezone(activity_records)
    has_location = timezone is not None

    if "heart_rate" in activity_records.columns:
        max_hr = activity_records["heart_rate"].max()
    else:
        max_hr = np.nan

    power_windows_arr = None
    power_curve_arr = None
    estimated_ftp = np.nan

    if "power" in activity_records.columns:
        power_windows_arr = utils.power_curve_windows(
            config.power_curve_max_s, config.power_curve_density
        )
        power_curve_arr = utils.compute_power_curve(
            activity_records["power"], power_windows_arr
        )
        if power_curve_arr is not None:
            valid = ~np.isnan(power_curve_arr)
            power_curve_arr = power_curve_arr[valid]
            power_windows_arr = power_windows_arr[valid]
            estimated_ftp = (
                np.interp(
                    config.ftp_window_s, power_windows_arr, power_curve_arr
                )
                * config.ftp_factor
            )
        else:
            # No power curve, so no windows to pair with
            power_windows_arr = None

    result = ActivityMetrics(
        filename=filename,
        timezone=timezone,
        has_location=has_location,
        max_heart_rate=max_hr,
        power_windows=power_windows_arr,
        power_curve=power_curve_arr,
        estimated_ftp=estimated_ftp,
    )

    if db is not None:
        db["activities"].upsert(result.to_db_dict(), pk=("filename", "segment"))

    return result


def load_file_metrics(
    files: pd.Series,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: ActivitiesConfig,
    cache: dict[str, ActivityMetrics] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, None], pd.DataFrame | None]]:
    """Compute per-activity metrics, using a pre-loaded cache when provided.

    Cache misses are computed and written to the DB as each file is processed.

    Args:
        files: Activity filenames aligned to the CSV index.
        path: Strava export directory.
        cache_dir: Records parquet cache directory.
        config: Activities configuration.
        cache: Activities cache keyed by filename, or ``None`` to skip caching.

    Returns:
        Tuple of:
        - Metrics DataFrame aligned to ``files.index``.
        - Preloaded coords dict keyed by ``(filename, None)`` for each cache
          miss, ready to pass to ``routes.cluster_routes_cached``.
    """
    if cache is not None:
        rows = [
            ActivityMetrics(filename=f) if pd.isna(f) else cache.get(f)
            for f in files
        ]
    else:
        rows = [
            ActivityMetrics(filename=f) if pd.isna(f) else None for f in files
        ]

    misses = [f for f, r in zip(files, rows) if r is None]
    miss_dfs = records.load_activity_records(misses, None, path, cache_dir)

    if misses:
        ctx = (
            cache_db.open_db(cache_dir)
            if cache_dir is not None
            else contextlib.nullcontext()
        )
        with ctx as db:
            miss_map = {
                f: parse_activity_file(f, df, config, db)
                for f, df in zip(misses, miss_dfs)
            }

        rows = [
            r if r is not None else miss_map[f] for f, r in zip(files, rows)
        ]

    preloaded_coords = {
        (f, None): records.coords_from_records(df)
        for f, df in zip(misses, miss_dfs)
    }

    return pd.DataFrame(rows, index=files.index), preloaded_coords


def build_activity_columns(
    csv: pd.DataFrame,
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None,
    config: ActivitiesConfig,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Compute all derived per-activity columns with a cache lookup.

    Reads the activities cache, computes file metrics (timezone, heart rate,
    FTP), infers trainer status, and calculates local dates. Cache misses are
    written to the DB as each file is parsed. Optionally runs route clustering.

    Args:
        csv: Raw Strava CSV indexed by UTC activity date.
        path: Strava export directory.
        home_tz: Fallback timezone for activities without GPS data.
        cache_dir: Optional cache directory. If None, no caching is performed.
        config: Activities configuration.

    Returns:
        Tuple of:
        - ``calcs``: Computed columns aligned to the index of ``csv``.
        - ``clusters``: Cluster assignments including position columns, or
          ``None`` when clustering is disabled.
    """
    cache = load_activities_cache(cache_dir) if cache_dir is not None else None

    calcs, preloaded_coords = load_file_metrics(
        csv["Filename"], path, cache_dir, config, cache
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

    if config.clustering is None:
        return calcs, None

    clusters = routes.cluster_routes_cached(
        csv,
        None,
        path,
        cache_dir,
        "activities",
        preloaded_coords,
        config.clustering,
    )
    return calcs, clusters


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
        - ``activities``: Summary metrics indexed by local date.
        - ``weekly_sums``: Metrics resampled to weekly buckets.
    """
    if config is None:
        config = ActivitiesConfig()

    csv = load_strava_activities_raw(path)
    calcs, clusters = build_activity_columns(
        csv, path, home_tz, cache_dir, config
    )

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
    if clusters is not None:
        df["cluster_id"] = clusters["cluster_id"]
        df["cluster_name"] = clusters["cluster_name"]
        if config.clustering.geocoding is not None:
            df["start_address"] = clusters["start_address"]
            df["end_address"] = clusters["end_address"]
    df["filename"] = csv["Filename"]

    activities = df.set_index("date").sort_index()

    weekly_metrics = ["distance", "elevation", "elapsed_time", "moving_time"]
    weekly_sums = (
        activities[weekly_metrics].resample(config.weekly_anchor).sum()
    )

    return activities, weekly_sums


def load_power_curves(
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None = None,
    config: ActivitiesConfig | None = None,
) -> pd.DataFrame:
    """Load power curves for all activities, computing and caching as needed.

    Args:
        path: Strava export directory.
        home_tz: Fallback timezone for activities without GPS location data.
            Either a fixed timezone string or a callable that accepts a Series
            and returns per-activity timezone values.
        cache_dir: Optional cache directory. If None, curves are computed on
            every call without caching.
        config: Activities configuration. Defaults to ``ActivitiesConfig()``.

    Returns:
        DataFrame indexed by local activity date with timedelta columns for
        each window duration. Activities without power data are omitted.
    """
    if config is None:
        config = ActivitiesConfig()

    csv = load_strava_activities_raw(path)
    calcs, _ = build_activity_columns(csv, path, home_tz, cache_dir, config)

    rows_dict = {}
    for local_date, metrics in zip(calcs["local_date"], calcs.itertuples()):
        if metrics.power_windows is None:
            continue
        rows_dict[local_date] = pd.Series(
            metrics.power_curve,
            index=pd.to_timedelta(metrics.power_windows, unit="s"),
        )

    if not rows_dict:
        return pd.DataFrame()

    return pd.DataFrame(rows_dict).T
