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
        clustering: Route clustering parameters. If None, ``cluster_id``,
            ``cluster_name``, and span columns are not added to the output.
        span_penalty: Changepoint detection penalty. Higher values produce
            fewer spans.
        span_min_size: Minimum number of commutes per span.
    """

    delta: pd.Timedelta = pd.Timedelta(90, "m")
    inactive_speed: float = 2.5
    stopped_speed: float = 1.0
    min_stop_duration: pd.Timedelta = pd.Timedelta(10, "s")
    morning_cutoff_hour: int = 12
    clustering: routes.RouteClusterConfig | None = dataclasses.field(
        default_factory=routes.RouteClusterConfig
    )
    span_penalty: float = 100.0
    span_min_size: int = 2


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
    def from_db_dict(cls, row: dict) -> CommuteMetrics:
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

        if files is None:
            with db.conn:
                db["commutes"].drop()
                db["cluster_fingerprints"].delete_where(
                    "table_name = ?", ["commutes"]
                )
        else:
            with db.conn:
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


def build_commute_columns(
    commutes: pd.DataFrame,
    path: str | PathLike[str],
    home_tz: Callable[[pd.Series], pd.Series | str] | str,
    cache_dir: str | PathLike[str] | None,
    config: CommuteConfig,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
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
        - ``calcs``: One row per commute split, indexed by local split start
          time, with cache-schema column names.
        - ``clusters``: Full cluster assignments including position columns,
          or ``None`` when clustering is disabled.
    """
    cache = load_commutes_cache(cache_dir) if cache_dir is not None else None

    file_mask = commutes["Filename"].notna()
    file_splits = load_commute_splits(
        commutes[file_mask], path, cache_dir, config, cache
    )

    csv_commutes = commutes[~file_mask]
    tz_series = pd.Series(np.nan, index=csv_commutes.index, dtype=object)
    tz_series = tz_series.mask(tz_series.isna(), home_tz)

    # Keyed by UTC timestamp (no filename for CSV-only activities).
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

    segments = [m.segment for m in metrics]
    calcs = (
        pd.DataFrame([dataclasses.asdict(m) for m in metrics])
        .set_index("date")
        .astype({"segment": "Int64"})
    )

    if config.clustering is None:
        return calcs, None

    clusters = routes.cluster_routes_cached(
        calcs,
        segments,
        path,
        cache_dir,
        "commutes",
        config.clustering,
    )

    return calcs, clusters


def compute_spans(
    commutes_df: pd.DataFrame,
    clusters: pd.DataFrame,
    config: CommuteConfig,
) -> pd.DataFrame | None:
    """Detect commute spans and enrich them with counts and addresses.

    Builds the home/work position signal from ``clusters``, runs PELT
    changepoint detection, then adds ``n_commutes`` and (when geocoding is
    enabled) ``home_address`` / ``work_address`` to each span row.

    Args:
        commutes_df: Commutes indexed by local date, must include ``direction``.
        clusters: Cluster result DataFrame aligned to ``commutes_df``.
        config: Configuration parameters.

    Returns:
        Spans DataFrame with ``start_date``, ``end_date``, ``n_commutes``, and
        optionally ``home_address`` / ``work_address`` columns.
    """
    geocoding_enabled = (
        config.clustering is not None
        and config.clustering.geocoding is not None
    )

    is_morning = commutes_df["direction"] == "Morning"
    signal = pd.DataFrame(
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
        index=commutes_df.index,
    )

    spans_df = utils.pelt_segments(
        signal, config.span_penalty, config.span_min_size
    )

    rows = []
    for _, span in spans_df.iterrows():
        in_span = (commutes_df.index >= span["start"]) & (
            commutes_df.index <= span["end"]
        )

        row: dict = {
            "start_date": span["start"].normalize(),
            "end_date": span["end"].normalize(),
            "n_commutes": int(in_span.sum()),
        }

        if geocoding_enabled:
            morning_mask = in_span & is_morning
            afternoon_mask = in_span & ~is_morning

            home_addrs = pd.concat(
                [
                    clusters.loc[morning_mask, "start_address"],
                    clusters.loc[afternoon_mask, "end_address"],
                ]
            ).dropna()
            work_addrs = pd.concat(
                [
                    clusters.loc[morning_mask, "end_address"],
                    clusters.loc[afternoon_mask, "start_address"],
                ]
            ).dropna()

            row["home_address"] = (
                home_addrs.mode().iat[0] if len(home_addrs) else None
            )
            row["work_address"] = (
                work_addrs.mode().iat[0] if len(work_addrs) else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


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

    When clustering is enabled, changepoint detection is run to identify
    periods of consistent commute endpoints. Each detected span represents a
    contiguous block of commutes between the same home location and workplace.

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
    calcs, clusters = build_commute_columns(
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
        if config.clustering.geocoding is not None:
            columns += ["start_address", "end_address"]

    if calcs.empty:
        empty = pd.DataFrame(columns=columns, index=pd.Index([], name="date"))
        return empty, None

    base_df = calcs if clusters is None else calcs.join(clusters)
    commutes_df = base_df.assign(
        elapsed_time=lambda d: pd.to_timedelta(d["elapsed_time_s"], unit="s"),
        moving_time=lambda d: pd.to_timedelta(
            d["moving_time_s"].fillna(0), unit="s"
        ).where(d["moving_time_s"].notna()),
    )[columns]

    spans_df = (
        compute_spans(commutes_df, clusters, config)
        if config.clustering is not None
        else None
    )

    return commutes_df.sort_index(), spans_df
