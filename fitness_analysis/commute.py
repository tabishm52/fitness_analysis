"""Functions for processing Strava bicycling commutes."""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import records, strava, utils

COMMUTES_CACHE_FNAME = "commutes_cache.csv"


@dataclass
class CommuteConfig:
    """Configuration parameters for ``load_commute_activities``.

    Attributes:
        delta: Time gap at which to split round-trip activities.
        inactive_speed: Speed (km/s) below which GPS is considered left on
            between commute segments.
        stopped_speed: Speed (km/s) below which the rider is considered
            stopped for moving time calculation.
        min_stop_duration: Minimum stop duration to exclude from moving time.
        morning_cutoff_hour: Hour (0-23) before which a commute is classified
            as morning; at or after which it is classified as afternoon.
    """

    delta: pd.Timedelta = pd.Timedelta(90, "m")
    inactive_speed: float = 2.5 / 3600.0
    stopped_speed: float = 1.0 / 3600.0
    min_stop_duration: pd.Timedelta = pd.Timedelta(10, "s")
    morning_cutoff_hour: int = 12


def process_commute_file(
    activity: pd.Series,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
) -> list[dict[str, Any]]:
    """Calculate summary metrics for one commute activity.

    Loads the activity file, filters out long inactive periods, splits on gaps
    greater than ``config.delta``, and returns metrics for each split.

    Args:
        activity: Activity row from the Strava CSV.
        path: Strava export directory.
        cache_dir: Optional cache directory for parsed activity records.
        config: Configuration parameters.

    Returns:
        List of metric dicts, one per commute split.
    """

    activity_records = records.parse_record_cached(
        activity["Filename"], path, cache_dir
    )

    # Drop periods of inactivity, to cover the cases where the GPS was left
    # on all day rather than being paused between commute segments
    if "distance" in activity_records.columns:
        inactive = utils.identify_inactive_periods(
            activity_records["distance"], config.inactive_speed, config.delta
        )
        active = (~inactive).reindex(activity_records.index).fillna(True)
        activity_records = activity_records[active]

    group_ids = (
        activity_records.index.to_series().diff() > config.delta
    ).cumsum()

    results = []
    for _, group in activity_records.groupby(group_ids, sort=False):
        timezone = utils.infer_timezone(group)
        ts = group.index[0]
        date = (ts.tz_convert(timezone) if timezone else ts).tz_localize(None)

        direction = (
            "Morning" if date.hour < config.morning_cutoff_hour else "Afternoon"
        )
        elapsed_time = group.index[-1] - group.index[0]

        if "distance" not in group.columns:
            distance = np.nan
            moving_time = np.nan
        else:
            dist = group["distance"]
            distance = utils.KM_TO_MI * (dist.max() - dist.min())
            inactive_periods = utils.identify_inactive_periods(
                group["distance"],
                config.stopped_speed,
                config.min_stop_duration,
            )
            moving_time = pd.Timedelta((~inactive_periods).sum(), "s")

        results.append(
            {
                "date": date,
                "description": activity["Activity Name"],
                "direction": direction,
                "distance": distance,
                "elapsed_time": elapsed_time,
                "moving_time": moving_time,
                "filename": activity["Filename"],
            }
        )

    return results


def process_commute_csv(
    activity: pd.Series,
    utc_date: pd.Timestamp,
    tz: str,
    config: CommuteConfig,
) -> dict[str, Any]:
    """Derive commute metrics for an activity with no file.

    Uses Strava CSV fields directly in place of parsed activity records.

    Args:
        activity: Activity row from the Strava CSV.
        utc_date: UTC start time of the activity (from the CSV index).
        tz: Timezone used to convert the UTC date to local time.
        config: Configuration parameters.

    Returns:
        Metric dict for the commute.
    """

    date = utc_date.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    direction = (
        "Morning" if date.hour < config.morning_cutoff_hour else "Afternoon"
    )

    return {
        "date": date,
        "description": activity["Activity Name"],
        "direction": direction,
        "distance": activity["Distance"] * utils.KM_TO_MI,
        "elapsed_time": pd.Timedelta(activity["Elapsed Time"], "s"),
        "moving_time": pd.Timedelta(activity["Moving Time"], "s"),
        "filename": activity["Filename"],
    }


def _cache_df_to_dict(
    cache: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Convert a cache DataFrame to a filename-keyed dict of result lists.

    Performs type conversions (strings to Timestamps/Timedeltas) once at load
    time so per-filename lookups are O(1) dict gets with no further parsing.
    """

    result: dict[str, list[dict[str, Any]]] = {}
    for filename, group in cache.groupby(level=0):
        result[filename] = [
            {
                "date": pd.Timestamp(r["date"]),
                "description": r["description"],
                "direction": r["direction"],
                "distance": r["distance"],
                "elapsed_time": pd.Timedelta(r["elapsed_time_s"], "s"),
                "moving_time": (
                    pd.Timedelta(r["moving_time_s"], "s")
                    if pd.notna(r["moving_time_s"])
                    else np.nan
                ),
                "filename": filename,
            }
            for _, r in group.iterrows()
        ]
    return result


def invalidate_commutes_cache(
    files: Iterable[str] | None,
    cache_dir: str | PathLike[str],
) -> None:
    """Invalidate the commutes cache.

    If ``files`` is None, deletes the entire cache file. Otherwise removes
    only the entries for the given activity filenames, leaving the rest intact.

    Args:
        files: Activity filenames to remove. If None, the whole cache is
            cleared.
        cache_dir: Cache directory passed to ``load_commute_activities``.
    """

    cache_path = Path(cache_dir) / COMMUTES_CACHE_FNAME
    if not cache_path.exists():
        return

    if files is None:
        cache_path.unlink()
        return

    cache = pd.read_csv(cache_path, index_col="filename")
    cache = cache[~cache.index.isin(list(files))]
    cache.to_csv(cache_path)


def process_commutes(
    file_commutes: pd.DataFrame,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Run calculations on a set of file-based commute activities.

    Computes per-activity commute splits. When ``cache_dir`` is provided, uses
    a two-level caching strategy:

    1. **Commutes cache** (``COMMUTES_CACHE_FNAME``): stores computed splits.
       Activities found here are returned immediately with no further work.
    2. **Records cache** (parquet files, one per activity): stores parsed
       FIT/TCX/GPX records. Warmed via ``warm_records_cache``, which
       parallelizes parsing of cold files when the batch is large enough.

    Without a cache directory, all files are parsed on every call, with
    parallelization applied directly for large batches.

    Args:
        file_commutes: Commute activities with a non-NaN ``Filename`` column.
        path: Strava export directory.
        cache_dir: Optional directory for both caches.
        config: Configuration parameters.

    Returns:
        Dict mapping each filename to its list of commute split dicts.
    """

    files = file_commutes["Filename"]

    if cache_dir is None:
        rows = list(file_commutes.iterrows())
        fn = partial(process_commute_file, path=path, config=config)
        with ProcessPoolExecutor() as ex:
            splits_list = list(ex.map(fn, [activity for _, activity in rows]))
        return {
            activity["Filename"]: splits
            for (_, activity), splits in zip(rows, splits_list)
        }

    cache_path = Path(cache_dir) / COMMUTES_CACHE_FNAME
    cache_df: pd.DataFrame | None = (
        pd.read_csv(cache_path, index_col="filename")
        if cache_path.exists()
        else None
    )
    cache = _cache_df_to_dict(cache_df) if cache_df is not None else None

    cache_results = {f: (cache or {}).get(f) for f in files}
    misses = [f for f, r in cache_results.items() if r is None]

    if not misses:
        return cache_results

    records.warm_records_cache(misses, path, cache_dir)

    miss_rows = file_commutes[file_commutes["Filename"].isin(misses)]
    for _, activity in miss_rows.iterrows():
        cache_results[activity["Filename"]] = process_commute_file(
            activity, path, cache_dir, config
        )

    new_rows = pd.DataFrame(
        [
            {
                "date": split["date"],
                "description": split["description"],
                "direction": split["direction"],
                "distance": split["distance"],
                "elapsed_time_s": split["elapsed_time"].total_seconds(),
                "moving_time_s": (
                    split["moving_time"].total_seconds()
                    if isinstance(split["moving_time"], pd.Timedelta)
                    else np.nan
                ),
            }
            for f in misses
            for split in (cache_results[f] or [])
        ],
        index=pd.Index(
            [f for f in misses for split in (cache_results[f] or [])],
            name="filename",
        ),
    )

    updated = pd.concat([c for c in [cache_df, new_rows] if c is not None])
    updated = updated[updated.index.notna()]
    updated.sort_values("date").sort_index(kind="stable").to_csv(cache_path)

    return cache_results


def load_commute_activities(
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None = None,
    config: CommuteConfig | None = None,
) -> pd.DataFrame:
    """Calculate summary metrics for a set of commute activities.

    Loads bicycling activities from a Strava export, filters to commutes,
    and returns per-commute summary metrics. Activities recorded as a single
    file are split on any gap greater than ``config.delta``, producing one
    entry per segment.

    Activities without a FIT/GPX file are treated as simple one-way commutes,
    deriving metrics from the Strava CSV fields and using ``home_tz`` to
    determine the local date.

    When ``cache_dir`` is provided, results are cached in
    ``COMMUTES_CACHE_FNAME`` so subsequent calls are fast even for large
    exports.

    Args:
        path: Strava export directory.
        home_tz: Timezone for commutes without GPS location data. Either a
            fixed timezone string or a callable that accepts a Series and
            returns per-activity timezone values.
        cache_dir: Optional directory for cached results. If omitted, activity
            files are parsed on every call.
        config: Optional config parameters. Defaults to ``CommuteConfig()``.

    Returns:
        Summary metrics from commute activities indexed by local date.
    """

    if config is None:
        config = CommuteConfig()

    csv = strava.load_strava_activities_raw(path)
    commutes = csv[csv["Commute"].fillna(False)]

    file_mask = commutes["Filename"].notna()
    file_commutes = commutes[file_mask]
    csv_commutes = commutes[~file_mask]

    cache_results = process_commutes(file_commutes, path, cache_dir, config)

    tz_series = pd.Series(np.nan, index=csv_commutes.index, dtype=object)
    tz_series = tz_series.mask(tz_series.isna(), home_tz)

    results: list[dict[str, Any]] = []
    for utc_date, activity in commutes.iterrows():
        fn = activity["Filename"]
        if pd.isna(fn):
            results.append(
                process_commute_csv(
                    activity, utc_date, tz_series.loc[utc_date], config
                )
            )
        else:
            results.extend(cache_results[fn] or [])

    if not results:
        columns = [
            "description",
            "direction",
            "distance",
            "elapsed_time",
            "moving_time",
            "filename",
        ]
        return pd.DataFrame(columns=columns, index=pd.Index([], name="date"))

    return pd.DataFrame(results).set_index("date")
