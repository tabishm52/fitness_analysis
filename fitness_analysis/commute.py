"""Functions for processing Strava bicycling commutes."""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import records, routes, strava, utils

COMMUTES_CACHE_FNAME = "commutes_cache.csv"


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


def _cache_df_to_dict(
    cache: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Convert a cache DataFrame to a filename-keyed dict of result lists."""

    result = {}
    for filename, group in cache.groupby(level=0):
        result[filename] = [
            {
                "date": pd.Timestamp(r["date"]),
                "description": r["description"],
                "direction": r["direction"],
                "distance": r["distance"],
                "elapsed_time_s": r["elapsed_time_s"],
                "moving_time_s": r["moving_time_s"],
                "filename": filename,
                "segment": (
                    None
                    if pd.isna(r.get("segment", np.nan))
                    else int(r["segment"])
                ),
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
    For targeted invalidation, also deletes any segment parquets written for
    multi-segment activities and clears cluster columns from surviving rows.

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
    files_list = list(files)
    to_remove = cache[cache.index.isin(files_list)]

    # Delete segment parquets for multi-segment activities being invalidated.
    if "segment" in to_remove.columns:
        segs = to_remove["segment"].dropna()
        if not segs.empty:
            records.invalidate_records_cache(
                segs.index,
                segs.astype(int),
                cache_dir,
            )

    surviving = cache[~cache.index.isin(files_list)]
    surviving.sort_index().to_csv(cache_path)


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
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
) -> list[dict[str, Any]]:
    """Calculate summary metrics for one commute activity.

    Loads the activity file, filters out long inactive periods, splits on gaps
    greater than ``config.delta``, and returns metrics for each split.

    When ``cache_dir`` is set and the file produces multiple splits, each
    split's records are written to the parquet cache via ``cache_record``.

    Args:
        activity: Activity row from the Strava CSV.
        path: Strava export directory.
        cache_dir: Optional cache directory for parsed activity records.
        config: Configuration parameters.

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
    cache_df: pd.DataFrame | None,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame | None]:
    """Load commute splits for a set of file-based activities.

    Computes per-activity commute splits. When ``cache_df`` is provided,
    uses the commutes cache for hits and processes only misses. When
    ``cache_df`` is ``None`` (no caching), all files are parsed in parallel.

    Args:
        file_commutes: Commute activities with a non-NaN ``Filename`` column.
        path: Strava export directory.
        cache_dir: Optional directory for the records parquet cache.
        config: Configuration parameters.
        cache_df: Already-loaded commutes cache indexed by filename, an empty
            DataFrame when the cache file does not yet exist, or ``None`` to
            skip caching entirely.

    Returns:
        Tuple of:
        - ``splits``: Dict mapping each filename to its list of commute split
          dicts.
        - ``new_rows``: DataFrame of newly computed rows to append to the
          cache, indexed by filename. ``None`` when there are no new rows.
    """

    files = file_commutes["Filename"]

    if cache_df is None:
        rows = list(file_commutes.iterrows())
        fn = partial(parse_commute_file, path=path, config=config)
        with ProcessPoolExecutor() as ex:
            splits_list = list(ex.map(fn, (activity for _, activity in rows)))
        return (
            {
                activity["Filename"]: splits
                for (_, activity), splits in zip(rows, splits_list)
            },
            None,
        )

    cache = _cache_df_to_dict(cache_df) if not cache_df.empty else {}

    splits = {f: cache.get(f) for f in files}
    misses = [f for f, r in splits.items() if r is None]

    if not misses:
        return splits, None

    records.warm_records_cache(misses, None, path, cache_dir)

    miss_rows = file_commutes[file_commutes["Filename"].isin(misses)]
    for _, activity in miss_rows.iterrows():
        splits[activity["Filename"]] = parse_commute_file(
            activity, path, cache_dir, config
        )

    new_rows = pd.DataFrame(
        split for f in misses for split in splits[f]
    ).set_index("filename")

    return splits, new_rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_cluster_columns(  # noqa: PLR0913
    file_results: list[dict[str, Any]],
    csv_results: list[dict[str, Any]],
    cache_df: pd.DataFrame | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
) -> tuple[pd.DataFrame | None, bool]:
    """Run route clustering and assign cluster columns into results in place.

    Returns ``(None, False)`` when there is nothing to cluster (no file results
    or no cache directory). CSV-only entries in ``csv_results`` are always
    filled with ``(pd.NA, None)``.

    Args:
        file_results: Commute result dicts with a non-NaN filename.
        csv_results: Commute result dicts without a file (CSV-only).
        cache_df: Already-loaded commutes cache, or ``None`` to skip lookup.
        path: Strava export directory.
        cache_dir: Records cache directory.
        config: Commute configuration.

    Returns:
        Tuple of:
        - ``clusters``: ``cluster_id`` and ``cluster_name`` columns aligned to
          ``file_results``, or ``None`` when skipped.
        - ``cluster_miss``: True when clustering was recomputed.
    """

    for r in csv_results:
        r["cluster_id"], r["cluster_name"] = pd.NA, None

    if not file_results or cache_dir is None:
        for r in file_results:
            r["cluster_id"], r["cluster_name"] = pd.NA, None
        return None, False

    clusters, cluster_miss = routes.cluster_routes_cached(
        pd.DataFrame(file_results),
        [r["segment"] for r in file_results],
        path,
        cache_dir,
        cache_df=cache_df,
        config=config.clustering,
    )

    for i, r in enumerate(file_results):
        r["cluster_id"] = clusters["cluster_id"].iat[i]
        r["cluster_name"] = clusters["cluster_name"].iat[i]

    return clusters, cluster_miss


def build_commute_columns(
    commutes: pd.DataFrame,
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
) -> list[dict[str, Any]]:
    """Compute all derived per-commute columns with a cache lookup.

    Reads the commutes cache, processes misses, and optionally runs route
    clustering. Writes updated cache at the end.

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

    cache_path = Path(cache_dir) / COMMUTES_CACHE_FNAME if cache_dir else None
    if cache_path is None:
        cache_df = None
    elif cache_path.exists():
        cache_df = pd.read_csv(cache_path, index_col="filename")
    else:
        cache_df = pd.DataFrame()

    file_mask = commutes["Filename"].notna()
    file_splits, new_rows = load_commute_splits(
        commutes[file_mask], path, cache_dir, config, cache_df
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

    cluster_miss = False
    if config.clustering is not None:
        file_results = [r for r in results if pd.notna(r["filename"])]
        csv_results = [r for r in results if pd.isna(r["filename"])]
        clusters, cluster_miss = load_cluster_columns(
            file_results,
            csv_results,
            cache_df,
            path,
            cache_dir,
            config,
        )

    if cache_path is not None and (new_rows is not None or cluster_miss):
        if new_rows is not None:
            updated = (
                pd.concat([cache_df, new_rows])
                if cache_df is not None and not cache_df.empty
                else new_rows
            )
        else:
            updated = cache_df.copy()

        if cluster_miss:
            cluster_map = {
                routes.cluster_cache_key(r["filename"], r["segment"]): (
                    clusters["cluster_id"].iat[i],
                    clusters["cluster_name"].iat[i],
                )
                for i, r in enumerate(file_results)
            }

            updated["cluster_id"], updated["cluster_name"] = zip(
                *(
                    cluster_map.get(
                        routes.cluster_cache_key(fn, row.get("segment")),
                        (pd.NA, None),
                    )
                    for fn, row in updated.iterrows()
                )
            )

        updated.sort_values("date").sort_index(kind="stable").to_csv(cache_path)

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
            moving_time=lambda d: pd.to_timedelta(d["moving_time_s"], unit="s"),
            segment=lambda d: d["segment"].astype("Int64"),
        )
        .drop(columns=["elapsed_time_s", "moving_time_s"])
    )
    if config.clustering is not None:
        df["cluster_id"] = df["cluster_id"].astype("Int64")
    return df
