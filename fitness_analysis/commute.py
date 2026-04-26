"""Functions for processing Strava bicycling commutes."""

import sqlite3
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd

from . import cache_db, records, routes, strava, utils


@dataclass
class CommuteConfig:
    """Configuration parameters for ``load_commute_activities``.

    Attributes:
        delta: Time gap at which to split round-trip activities.
        inactive_speed: Speed (km/hr) below which GPS is considered left on
            between commute segments.
        stopped_speed: Speed (km/hr) below which the rider is considered
            stopped for moving time calculation.
        min_stop_duration: Minimum stop duration to exclude from moving time.
        morning_cutoff_hour: Hour (0-23) before which a commute is classified
            as morning; at or after which it is classified as afternoon.
        clustering: Route clustering parameters. If None, ``cluster_id`` and
            ``cluster_name`` columns are not added to returned activities.
    """

    delta: pd.Timedelta = pd.Timedelta(90, "m")
    inactive_speed: float = 2.5
    stopped_speed: float = 1.0
    min_stop_duration: pd.Timedelta = pd.Timedelta(10, "s")
    morning_cutoff_hour: int = 12
    clustering: routes.RouteClusterConfig | None = field(
        default_factory=routes.RouteClusterConfig
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def load_commutes_cache(
    cache_dir: str | PathLike[str],
) -> dict[str, list[dict[str, Any]]]:
    """Read the commutes cache from the database.

    Args:
        cache_dir: Cache directory containing the SQLite database.

    Returns:
        Dict mapping each activity filename to its list of cached split dicts.
    """

    with cache_db.open_db(cache_dir) as conn:
        rows = conn.execute(
            "SELECT filename, segment, date, description, direction,"
            " distance, elapsed_time_s, moving_time_s FROM commutes"
            " ORDER BY filename"
        ).fetchall()

    result = {}
    for fn, seg, date, desc, direction, dist, elapsed_s, moving_s in rows:
        result.setdefault(fn, []).append(
            {
                "date": pd.Timestamp(date),
                "description": desc,
                "direction": direction,
                "distance": dist,
                "elapsed_time_s": elapsed_s,
                "moving_time_s": moving_s,
                "filename": fn,
                "segment": cache_db.segment_from_db(seg),
            }
        )

    return result


def invalidate_commutes_cache(
    files: Iterable[str] | None,
    cache_dir: str | PathLike[str],
) -> None:
    """Invalidate the commutes cache.

    If ``files`` is None, clears the entire commutes table and deletes any
    segment parquets for multi-segment activities. Otherwise removes only the
    entries for the given activity filenames, deleting their segment parquets
    as well.

    Args:
        files: Activity filenames to remove. If None, the whole cache is
            cleared.
        cache_dir: Cache directory passed to ``load_commute_activities``.
    """

    if not cache_db.db_path(cache_dir).exists():
        return

    with cache_db.open_db(cache_dir) as conn:
        if files is None:
            rows = conn.execute(
                "SELECT filename, segment FROM commutes WHERE segment != -1"
            ).fetchall()
        else:
            files_list = list(files)
            marks = ",".join("?" * len(files_list))
            rows = conn.execute(
                "SELECT filename, segment FROM commutes"
                f" WHERE filename IN ({marks}) AND segment != -1",
                files_list,
            ).fetchall()

        if rows:
            fns, segs = zip(*rows)
            records.invalidate_records_cache(fns, segs, cache_dir)

        if files is None:
            conn.execute("DELETE FROM commutes")
        else:
            conn.execute(
                f"DELETE FROM commutes WHERE filename IN ({marks})",
                files_list,
            )


# ---------------------------------------------------------------------------
# Activity processing
# ---------------------------------------------------------------------------


def segment_metrics(
    activity: pd.Series,
    group: pd.DataFrame,
    seg_idx: int | None,
    config: CommuteConfig,
) -> dict[str, Any]:
    """Compute summary metrics for one commute segment.

    Args:
        activity: Activity row from the Strava CSV.
        group: Activity records for this segment.
        seg_idx: 1-based segment index, or ``None`` for single-segment
            activities.
        config: Configuration parameters.

    Returns:
        Metric dict for the segment.
    """

    timezone = utils.infer_timezone(group)
    ts = group.index[0]
    date = (ts.tz_convert(timezone) if timezone else ts).tz_localize(None)
    direction = (
        "Morning" if date.hour < config.morning_cutoff_hour else "Afternoon"
    )
    elapsed_time_s = (group.index[-1] - group.index[0]).total_seconds()

    if "distance" not in group.columns:
        distance, moving_time_s = np.nan, np.nan
    else:
        dist = group["distance"]
        distance = utils.KM_TO_MI * (dist.max() - dist.min())
        inactive_periods = utils.identify_inactive_periods(
            group["distance"],
            config.stopped_speed / 3600.0,
            config.min_stop_duration,
        )
        moving_time_s = float((~inactive_periods).sum())

    return {
        "date": date,
        "description": activity["Activity Name"],
        "direction": direction,
        "distance": distance,
        "elapsed_time_s": elapsed_time_s,
        "moving_time_s": moving_time_s,
        "filename": activity["Filename"],
        "segment": seg_idx,
    }


def parse_commute_file(
    activity: pd.Series,
    path: str | PathLike[str],
    config: CommuteConfig,
    cache_dir: str | PathLike[str] | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Calculate summary metrics for one commute activity.

    Loads the activity file, filters out long inactive periods, splits on gaps
    greater than ``config.delta``, and returns metrics for each split.

    When ``cache_dir`` is set and the file produces multiple splits, each
    split's records are written to the parquet cache. When ``conn`` is
    provided, the computed splits are inserted into the cache as each file is
    parsed.

    Args:
        activity: Activity row from the Strava CSV.
        path: Strava export directory.
        config: Configuration parameters.
        cache_dir: Optional cache directory for parsed activity records.
        conn: Optional open DB connection.

    Returns:
        List of metric dicts, one per commute split.
    """

    activity_records = records.parse_record_cached(
        activity["Filename"], None, path, cache_dir
    )

    # Drop periods of inactivity, to cover the cases where the GPS was left
    # on all day rather than being paused between commute segments
    if "distance" in activity_records.columns:
        inactive = utils.identify_inactive_periods(
            activity_records["distance"],
            config.inactive_speed / 3600.0,
            config.delta,
        )
        active = (~inactive).reindex(activity_records.index).fillna(True)
        activity_records = activity_records[active]

    group_ids = (
        activity_records.index.to_series().diff() > config.delta
    ).cumsum()
    groups = list(activity_records.groupby(group_ids, sort=False))

    results = []
    for i, (_, group) in enumerate(groups):
        seg_idx = None if len(groups) == 1 else i + 1
        if cache_dir is not None and seg_idx is not None:
            records.cache_record(
                group, activity["Filename"], seg_idx, cache_dir
            )
        results.append(segment_metrics(activity, group, seg_idx, config))

    if conn is not None:
        conn.executemany(
            "INSERT OR REPLACE INTO commutes"
            " (filename, segment, date, description, direction,"
            " distance, elapsed_time_s, moving_time_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    activity["Filename"],
                    cache_db.segment_to_db(split["segment"]),
                    str(split["date"]),
                    split["description"],
                    split["direction"],
                    cache_db.to_sql(split["distance"]),
                    cache_db.to_sql(split["elapsed_time_s"]),
                    cache_db.to_sql(split["moving_time_s"]),
                )
                for split in results
            ],
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
        "elapsed_time_s": float(activity["Elapsed Time"]),
        "moving_time_s": float(activity["Moving Time"]),
        "filename": activity["Filename"],
        "segment": None,
    }


def load_commute_splits(
    file_commutes: pd.DataFrame,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
    cache: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Compute per-commute metrics, using a pre-loaded cache when provided.

    Cache misses are computed and written to the DB as each file is parsed.

    Args:
        file_commutes: Commute activities with a non-NaN ``Filename`` column.
        path: Strava export directory.
        cache_dir: Optional directory for the records parquet cache.
        config: Configuration parameters.
        cache: Commutes cache keyed by filename, or ``None`` to skip caching.

    Returns:
        Dict mapping each filename to its list of commute split dicts.
    """

    files = file_commutes["Filename"]

    if cache is None:
        rows = list(file_commutes.iterrows())
        fn = partial(parse_commute_file, path=path, config=config)
        with ProcessPoolExecutor() as ex:
            splits_list = list(ex.map(fn, (activity for _, activity in rows)))
        return {
            activity["Filename"]: splits
            for (_, activity), splits in zip(rows, splits_list)
        }

    splits = {f: cache.get(f) for f in files}
    misses = [f for f, r in splits.items() if r is None]

    if not misses:
        return splits

    records.warm_records_cache(misses, None, path, cache_dir)

    miss_rows = file_commutes[file_commutes["Filename"].isin(misses)]

    with cache_db.open_db(cache_dir) as conn:
        for _, activity in miss_rows.iterrows():
            splits[activity["Filename"]] = parse_commute_file(
                activity, path, config, cache_dir, conn
            )

    return splits


def build_commute_columns(
    commutes: pd.DataFrame,
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
) -> list[dict[str, Any]]:
    """Compute all derived per-commute columns with a cache lookup.

    Reads the commutes cache, processes misses, and optionally runs route
    clustering. Cache misses are written to the DB as each file is parsed.

    Args:
        commutes: Commute activities from the Strava CSV.
        path: Strava export directory.
        home_tz: Timezone for commutes without GPS location data. Either a
            fixed timezone string or a callable that accepts a Series and
            returns per-activity timezone values.
        cache_dir: Optional cache directory. If ``None``, no caching is
            performed.
        config: Configuration parameters.

    Returns:
        List of result dicts, one per commute split.
    """

    cache = load_commutes_cache(cache_dir) if cache_dir is not None else None

    file_mask = commutes["Filename"].notna()
    file_splits = load_commute_splits(
        commutes[file_mask], path, cache_dir, config, cache
    )

    csv_commutes = commutes[~file_mask]
    tz_series = pd.Series(np.nan, index=csv_commutes.index, dtype=object)
    tz_series = tz_series.mask(tz_series.isna(), home_tz)

    csv_splits = {
        utc_date: process_commute_csv(
            activity, utc_date, tz_series.loc[utc_date], config
        )
        for utc_date, activity in csv_commutes.iterrows()
    }

    results = []
    for utc_date, activity in commutes.iterrows():
        fn = activity["Filename"]
        if pd.isna(fn):
            results.append(csv_splits[utc_date])
        else:
            results.extend(file_splits[fn])

    if config.clustering is not None:
        clusters = routes.cluster_routes_cached(
            pd.DataFrame(results),
            [r["segment"] for r in results],
            path,
            cache_dir,
            "commutes",
            config.clustering,
        )
        for i, r in enumerate(results):
            r["cluster_id"] = clusters["cluster_id"].iat[i]
            r["cluster_name"] = clusters["cluster_name"].iat[i]

    return results


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


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
    results = build_commute_columns(commutes, path, home_tz, cache_dir, config)

    if not results:
        columns = [
            "description",
            "direction",
            "distance",
            "elapsed_time",
            "moving_time",
            "filename",
            "segment",
        ]
        if config.clustering is not None:
            columns += ["cluster_id", "cluster_name"]
        return pd.DataFrame(columns=columns, index=pd.Index([], name="date"))

    df = (
        pd.DataFrame(results)
        .set_index("date")
        .assign(
            elapsed_time=lambda d: pd.to_timedelta(
                d["elapsed_time_s"], unit="s"
            ),
            moving_time=lambda d: pd.to_timedelta(
                d["moving_time_s"].fillna(0), unit="s"
            ).where(d["moving_time_s"].notna()),
            segment=lambda d: d["segment"].astype("Int64"),
        )
        .drop(columns=["elapsed_time_s", "moving_time_s"])
    )
    if config.clustering is not None:
        df["cluster_id"] = df["cluster_id"].astype("Int64")
    return df
