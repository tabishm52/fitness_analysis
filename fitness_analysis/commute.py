"""Functions for processing Strava bicycling commutes."""

import dataclasses
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from os import PathLike

import numpy as np
import pandas as pd
import sqlite_utils
from sklearn.preprocessing import StandardScaler

from . import cache_db, changepoints, records, routes, strava, utils


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
        span_detection: Span detection parameters. Requires ``clustering``
            to be non-None. If None, span detection is skipped.
    """

    delta: pd.Timedelta = pd.Timedelta(90, "m")
    inactive_speed: float = 2.5
    stopped_speed: float = 1.0
    min_stop_duration: pd.Timedelta = pd.Timedelta(10, "s")
    morning_cutoff_hour: int = 12
    clustering: routes.RouteClusterConfig | None = dataclasses.field(
        default_factory=routes.RouteClusterConfig
    )
    span_detection: changepoints.SpanConfig | None = dataclasses.field(
        default_factory=changepoints.SpanConfig
    )


@dataclass(kw_only=True)
class CommuteMetrics:
    """Cached metrics for one commute segment.

    Attributes:
        date: Local date and time of the segment start.
        description: Activity name from Strava.
        direction: ``"Morning"`` or ``"Afternoon"``.
        distance: Distance in miles, or ``None`` when GPS distance is absent.
        elapsed_time_s: Total elapsed time in seconds.
        moving_time_s: Moving time in seconds, or ``None`` when absent.
        filename: Activity filename, or NaN for fileless activities.
        segment: 1-based split index, or ``None`` for single-segment activities.
    """

    date: pd.Timestamp
    description: str
    direction: str
    distance: float | None = None
    elapsed_time_s: float
    moving_time_s: float | None = None
    filename: str | float
    segment: int | None = None

    def __post_init__(self) -> None:
        """Coerce numpy scalars to Python-native types."""
        self.distance = cache_db.to_sql(self.distance)
        self.elapsed_time_s = cache_db.to_sql(self.elapsed_time_s)
        self.moving_time_s = cache_db.to_sql(self.moving_time_s)

    def to_db_dict(self) -> dict:
        """Return a dict suitable for ``db["commutes"].upsert()``."""
        return {
            "date": str(self.date),
            "description": self.description,
            "direction": self.direction,
            "distance": self.distance,
            "elapsed_time_s": self.elapsed_time_s,
            "moving_time_s": self.moving_time_s,
            "filename": self.filename,
            "segment": cache_db.segment_to_db(self.segment),
        }

    @classmethod
    def from_db_dict(cls, row: dict) -> "CommuteMetrics":
        """Construct from a ``db["commutes"].rows`` row dict."""
        return cls(
            date=pd.Timestamp(row["date"]),
            description=row["description"],
            direction=row["direction"],
            distance=row["distance"],
            elapsed_time_s=row["elapsed_time_s"],
            moving_time_s=row["moving_time_s"],
            filename=row["filename"],
            segment=cache_db.segment_from_db(row["segment"]),
        )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def load_commutes_cache(
    cache_dir: str | PathLike[str],
) -> dict[str, list[CommuteMetrics]]:
    """Read the commutes cache from the database.

    Args:
        cache_dir: Cache directory containing the SQLite database.

    Returns:
        Dict mapping each activity filename to its list of cached split metrics.
    """
    result = {}
    with cache_db.open_db(cache_dir) as db:
        for row in db["commutes"].rows:
            m = CommuteMetrics.from_db_dict(row)
            result.setdefault(m.filename, []).append(m)

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

    with cache_db.open_db(cache_dir) as db:
        if files is None:
            rows = list(
                db["commutes"].rows_where(
                    "segment != -1", select="filename, segment"
                )
            )
        else:
            files_list = list(files)
            marks = ",".join("?" * len(files_list))
            rows = list(
                db["commutes"].rows_where(
                    f"filename IN ({marks}) AND segment != -1",
                    files_list,
                    select="filename, segment",
                )
            )

        if rows:
            records.invalidate_records_cache(
                (r["filename"] for r in rows),
                (r["segment"] for r in rows),
                cache_dir,
            )

        # Single transaction: row delete and fingerprint delete are atomic;
        # delete_where() uses db.execute() directly and participates correctly.
        with db.conn:
            if files is None:
                db["commutes"].delete_where()
            else:
                db["commutes"].delete_where(
                    f"filename IN ({marks})", files_list
                )
            db["cluster_fingerprints"].delete_where(
                "table_name = ?", ["commutes"]
            )


# ---------------------------------------------------------------------------
# Activity processing
# ---------------------------------------------------------------------------


def segment_metrics(
    activity: pd.Series,
    group: pd.DataFrame,
    seg_idx: int | None,
    config: CommuteConfig,
) -> CommuteMetrics:
    """Compute summary metrics for one commute segment.

    Args:
        activity: Activity row from the Strava CSV.
        group: Activity records for this segment.
        seg_idx: 1-based segment index, or ``None`` for single-segment
            activities.
        config: Configuration parameters.

    Returns:
        Computed metrics for the segment.
    """
    timezone = utils.infer_timezone(group)
    ts = group.index[0]
    date = (ts.tz_convert(timezone) if timezone else ts).tz_localize(None)
    direction = (
        "Morning" if date.hour < config.morning_cutoff_hour else "Afternoon"
    )
    elapsed_time_s = (group.index[-1] - group.index[0]).total_seconds()

    if "distance" in group.columns:
        dist = group["distance"]
        distance = utils.KM_TO_MI * (dist.max() - dist.min())
        inactive_periods = utils.identify_inactive_periods(
            group["distance"],
            config.stopped_speed / 3600.0,
            config.min_stop_duration,
        )
        moving_time_s = float((~inactive_periods).sum())
    else:
        distance = None
        moving_time_s = None

    return CommuteMetrics(
        date=date,
        description=activity["Activity Name"],
        direction=direction,
        distance=distance,
        elapsed_time_s=elapsed_time_s,
        moving_time_s=moving_time_s,
        filename=activity["Filename"],
        segment=seg_idx,
    )


def parse_commute_file(
    activity: pd.Series,
    path: str | PathLike[str],
    config: CommuteConfig,
    cache_dir: str | PathLike[str] | None = None,
    db: sqlite_utils.Database | None = None,
) -> list[CommuteMetrics]:
    """Calculate summary metrics for one commute activity.

    Loads the activity file, filters out long inactive periods, splits on gaps
    greater than ``config.delta``, and returns metrics for each split.

    When ``cache_dir`` is set and the file produces multiple splits, each
    split's records are written to the parquet cache. When ``db`` is provided,
    the computed splits are inserted into the cache as each file is parsed.

    Args:
        activity: Activity row from the Strava CSV.
        path: Strava export directory.
        config: Configuration parameters.
        cache_dir: Optional cache directory for parsed activity records.
        db: Optional open database.

    Returns:
        List of ``CommuteMetrics``, one per commute split.
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

    if db is not None:
        db["commutes"].upsert_all(
            [split.to_db_dict() for split in results],
            pk=("filename", "segment"),
        )

    return results


def process_commute_csv(
    activity: pd.Series,
    utc_date: pd.Timestamp,
    tz: str,
    config: CommuteConfig,
) -> CommuteMetrics:
    """Derive commute metrics for an activity with no file.

    Uses Strava CSV fields directly in place of parsed activity records.

    Args:
        activity: Activity row from the Strava CSV.
        utc_date: UTC start time of the activity (from the CSV index).
        tz: Timezone used to convert the UTC date to local time.
        config: Configuration parameters.

    Returns:
        Computed metrics for the commute.
    """
    date = utc_date.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    direction = (
        "Morning" if date.hour < config.morning_cutoff_hour else "Afternoon"
    )

    return CommuteMetrics(
        date=date,
        description=activity["Activity Name"],
        direction=direction,
        distance=activity["Distance"] * utils.KM_TO_MI,
        elapsed_time_s=float(activity["Elapsed Time"]),
        moving_time_s=float(activity["Moving Time"]),
        filename=activity["Filename"],
    )


def load_commute_splits(
    file_commutes: pd.DataFrame,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
    cache: dict[str, list[CommuteMetrics]] | None,
) -> dict[str, list[CommuteMetrics]]:
    """Compute per-commute metrics, using a pre-loaded cache when provided.

    Cache misses are computed and written to the DB as each file is parsed.

    Args:
        file_commutes: Commute activities with a non-NaN ``Filename`` column.
        path: Strava export directory.
        cache_dir: Optional directory for the records parquet cache.
        config: Configuration parameters.
        cache: Commutes cache keyed by filename, or ``None`` to skip caching.

    Returns:
        Dict mapping each filename to its list of commute split metrics.
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

    with cache_db.open_db(cache_dir) as db:
        for _, activity in miss_rows.iterrows():
            splits[activity["Filename"]] = parse_commute_file(
                activity, path, config, cache_dir, db
            )

    return splits


def commute_spans_signal(
    direction: pd.Series,
    clusters: pd.DataFrame,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Build a scaled 4D home/work position signal for span detection.

    Args:
        direction: Per-commute direction (``"Morning"`` or ``"Afternoon"``),
            with a DatetimeIndex.
        clusters: Cluster DataFrame with ``start_lat``, ``start_lon``,
            ``end_lat``, ``end_lon`` columns, aligned to ``direction``.

    Returns:
        Tuple of:
        - Scaled ``(N, 4)`` signal array.
        - Aligned DatetimeIndex after dropping rows without GPS data.
    """
    is_morning = (direction == "Morning").to_numpy()
    positions = pd.DataFrame(
        {
            "home_lat": np.where(
                is_morning, clusters["start_lat"], clusters["end_lat"]
            ),
            "home_lon": np.where(
                is_morning, clusters["start_lon"], clusters["end_lon"]
            ),
            "work_lat": np.where(
                is_morning, clusters["end_lat"], clusters["start_lat"]
            ),
            "work_lon": np.where(
                is_morning, clusters["end_lon"], clusters["start_lon"]
            ),
        },
        index=direction.index,
    ).dropna()

    signal = StandardScaler().fit_transform(positions.values)
    return signal, positions.index


def build_commute_columns(
    commutes: pd.DataFrame,
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
) -> list[dict]:
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
        Tuple of:
        - ``results``: list of result dicts, one per commute split.
        - ``clusters``: full cluster DataFrame (including position columns)
          when clustering is enabled, or ``None`` otherwise.
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

    metrics = []
    for utc_date, activity in commutes.iterrows():
        fn = activity["Filename"]
        if pd.isna(fn):
            metrics.append(csv_splits[utc_date])
        else:
            metrics.extend(file_splits[fn])

    results = [dataclasses.asdict(m) for m in metrics]

    if config.clustering is None:
        return results, None

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

    return results, clusters


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def load_commute_activities(
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None = None,
    config: CommuteConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Calculate summary metrics for a set of commute activities.

    Loads bicycling activities from a Strava export, filters to commutes,
    and returns per-commute summary metrics. Activities recorded as a single
    file are split on any gap greater than ``config.delta``, producing one
    entry per segment.

    Activities without a FIT/GPX file are treated as simple one-way commutes,
    deriving metrics from the Strava CSV fields and using ``home_tz`` to
    determine the local date.

    When ``config.span_detection`` is set, changepoint detection is run to
    identify periods of consistent commute endpoints. Each detected span
    represents a contiguous block of commutes between the same home location
    and workplace.

    Args:
        path: Strava export directory.
        home_tz: Timezone for commutes without GPS location data. Either a
            fixed timezone string or a callable that accepts a Series and
            returns per-activity timezone values.
        cache_dir: Optional directory for cached results. If omitted, activity
            files are parsed on every call.
        config: Optional config parameters. Defaults to ``CommuteConfig()``.

    Returns:
        Tuple of:
        - ``commutes_df``: Summary metrics indexed by local date.
        - ``spans_df``: Detected spans, or ``None`` when not configured.
    """
    if config is None:
        config = CommuteConfig()

    csv = strava.load_strava_activities_raw(path)
    commutes_csv = csv[csv["Commute"].fillna(False)]
    results, clusters = build_commute_columns(
        commutes_csv, path, home_tz, cache_dir, config
    )

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

    if not results:
        empty = pd.DataFrame(columns=columns, index=pd.Index([], name="date"))
        return empty, None

    commutes_df = (
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
    )[columns]
    if config.clustering is not None:
        commutes_df["cluster_id"] = commutes_df["cluster_id"].astype("Int64")

    spans_df = None
    if config.span_detection is not None and clusters is not None:
        signal, idx = commute_spans_signal(commutes_df["direction"], clusters)
        spans_df = changepoints.detect_spans(signal, idx, config.span_detection)

    return commutes_df, spans_df
