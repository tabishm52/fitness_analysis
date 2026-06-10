"""GPS route clustering for bicycle activities."""

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import pandas as pd
import utm
from shapely import frechet_distance
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN

from . import cache_db, geocoding, records, utils


@dataclass
class RouteClusterConfig:
    """Configuration parameters for ``cluster_routes``.

    Attributes:
        points_per_km: Number of resampled points per km of route length.
        points_min: Minimum number of resampled points (short routes).
        points_max: Maximum number of resampled points (long routes).
        partition_eps_m: How close two routes' start (or end) points must
            be to be considered the same origin (or destination), in metres.
        similarity_eps: How similar two routes must be to be clustered
            together, as a fraction of their mean length (e.g. 0.02 = 2%).
        min_samples: Minimum rides on a route for it to form a cluster.
        length_ratio_max: Routes longer than this multiple of each other
            are skipped without computing a full shape comparison.
        geocoding: Geocoding parameters. If ``None``, ``start_address`` and
            ``end_address`` will be absent from cluster results.
    """

    points_per_km: float = 1.0
    points_min: int = 20
    points_max: int = 100
    partition_eps_m: float = 750.0
    similarity_eps: float = 0.02
    min_samples: int = 2
    length_ratio_max: float = 1.2
    geocoding: geocoding.GeocodingConfig | None = dataclasses.field(
        default_factory=geocoding.GeocodingConfig
    )

    @property
    def partition_eps_rad(self) -> float:
        """``partition_eps_m`` converted to radians."""
        return self.partition_eps_m / utils.EARTH_RADIUS_M

    # If True, cluster_routes() uses Strava CSV column names. If False,
    # cluster_routes() uses column names from load_strava_activities().
    raw_csv: bool = dataclasses.field(default=False, init=False, repr=False)

    @property
    def filename_col(self) -> str:
        return "Filename" if self.raw_csv else "filename"

    @property
    def name_col(self) -> str:
        return "Activity Name" if self.raw_csv else "description"


@dataclass
class ClusterResult:
    """Cached cluster assignment for one activity or commute segment.

    Attributes:
        cluster_id: Integer cluster ID (0 = most frequent route), or ``None``
            for unmatched activities.
        cluster_name: Modal activity name within the cluster, or ``None`` for
            unmatched activities.
        start_lat: Representative start latitude. For clustered GPS activities
            this is the cluster centroid (median); for GPS activities below
            ``min_samples`` it is the raw route start. ``None`` for activities
            without GPS data.
        start_lon: Representative start longitude (same semantics as
            ``start_lat``).
        end_lat: Representative end latitude (same semantics as ``start_lat``).
        end_lon: Representative end longitude (same semantics as ``start_lat``).
    """

    cluster_id: int | None = None
    cluster_name: str | None = None
    start_lat: float | None = None
    start_lon: float | None = None
    end_lat: float | None = None
    end_lon: float | None = None
    start_address: str | None = None
    end_address: str | None = None

    @classmethod
    def update_sql(cls, table: str) -> str:
        """UPDATE SQL for writing cluster assignments back to ``table``."""
        sets = ", ".join(f"{f.name}=?" for f in dataclasses.fields(cls))
        return f"UPDATE {table} SET {sets} WHERE filename=? AND segment=?"

    def to_update_params(self, filename: str, segment: int) -> tuple:
        """Positional params for ``update_sql`` (SET values then WHERE key)."""
        return (*dataclasses.asdict(self).values(), filename, segment)

    @classmethod
    def from_db_dict(cls, row: dict) -> ClusterResult:
        """Construct from a ``db[table].rows_where()`` row dict."""
        return cls(**{f.name: row[f.name] for f in dataclasses.fields(cls)})


def compute_cluster_fingerprint(
    keys: Iterable[tuple[str, int] | None],
    config: RouteClusterConfig,
) -> str:
    """Compute an MD5 fingerprint of the activity keys and clustering config.

    Used to detect whether the cached cluster assignments are still valid.
    Any change to the activity set or config parameters produces a different
    fingerprint, triggering a full recompute.

    Args:
        keys: ``cache_key`` results for all activities, including ``None``
            for fileless entries. Order and duplicates do not matter.
        config: Clustering configuration.

    Returns:
        MD5 hex digest string.
    """
    init_fields = {f.name for f in dataclasses.fields(config) if f.init}
    config_dict = {
        k: v for k, v in dataclasses.asdict(config).items() if k in init_fields
    }

    file_keys = sorted({k for k in keys if k is not None})
    payload = json.dumps(
        {"keys": file_keys, "config": config_dict}, sort_keys=True
    )

    return hashlib.md5(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# GPS preprocessing
# ---------------------------------------------------------------------------


def resample_route(
    lat: np.ndarray,
    lon: np.ndarray,
    zone_number: int,
    zone_letter: str,
    config: RouteClusterConfig,
) -> tuple[np.ndarray, float] | None:
    """Project a GPS route to UTM and arc-length resample it.

    Uses a caller-supplied UTM zone so all routes in a dataset share a
    consistent coordinate system.

    Args:
        lat: Latitude values (may contain NaN).
        lon: Longitude values (may contain NaN).
        zone_number: UTM zone number to force.
        zone_letter: UTM zone letter to force.
        config: Clustering configuration.

    Returns:
        ``(route_xy, route_length_m)`` where ``route_xy`` is shape (n, 2),
        or None if the route has zero length after NaN filtering.
    """
    mask = np.isfinite(lat)
    lat, lon = lat[mask], lon[mask]
    easting, northing, _, _ = utm.from_latlon(
        lat, lon, force_zone_number=zone_number, force_zone_letter=zone_letter
    )
    xy = np.column_stack([easting, northing])

    deltas = np.diff(xy, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    route_length_m = cum_len[-1]

    if route_length_m == 0.0:
        return None

    raw_points = int(route_length_m / 1000.0 * config.points_per_km)
    n_points = max(config.points_min, min(config.points_max, raw_points))
    sample_distances = np.linspace(0.0, route_length_m, n_points)
    route_xy = np.column_stack(
        [
            np.interp(sample_distances, cum_len, xy[:, 0]),
            np.interp(sample_distances, cum_len, xy[:, 1]),
        ]
    )

    return route_xy, route_length_m


def extract_route_features(
    activities: pd.DataFrame,
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    preloaded_coords: dict[tuple[str, int | None], pd.DataFrame | None] | None,
    config: RouteClusterConfig,
) -> tuple[list, list[pd.DataFrame]]:
    """Load GPS records and return trimmed lat/lon data for each route.

    Args:
        activities: Subset of the activity DataFrame with filenames present.
            Activities with no GPS data are excluded from the valid output.
        segments: Per-activity segment indices, or ``None`` to treat all
            activities as whole-file.
        path: Strava export directory.
        cache_dir: Optional records parquet cache directory.
        preloaded_coords: Optional mapping of ``(filename, segment)`` to
            already-extracted lat/lon DataFrames (or ``None`` for no-GPS
            files).
        config: Clustering configuration.

    Returns:
        Tuple of:
        - ``valid_idx``: the original activity index values.
        - ``valid_routes``: corresponding trimmed lat/lon DataFrames, one per
          valid route (``latitude`` and ``longitude`` columns, NaN-trimmed).
    """
    if preloaded_coords:
        fns = activities[config.filename_col]
        segs = (
            list(segments) if segments is not None else itertools.repeat(None)
        )

        misses = [
            pair for pair in zip(fns, segs) if pair not in preloaded_coords
        ]
        miss_coords = records.load_activity_coords(
            [fn for fn, _ in misses],
            [s for _, s in misses],
            path,
            cache_dir,
        )

        coords_map = dict(zip(misses, miss_coords)) | preloaded_coords
        coords_list = [coords_map[pair] for pair in zip(fns, segs)]

    else:
        coords_list = records.load_activity_coords(
            activities[config.filename_col], segments, path, cache_dir
        )

    if not any(c is not None for c in coords_list):
        return [], []

    gps_idx = list(activities.index)
    valid_idx = [idx for idx, c in zip(gps_idx, coords_list) if c is not None]
    valid_routes = [c for c in coords_list if c is not None]
    return valid_idx, valid_routes


# ---------------------------------------------------------------------------
# Pairwise Fréchet distance
# ---------------------------------------------------------------------------


def frechet_pair(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    len_a: float,
    len_b: float,
    length_ratio_max: float,
) -> float:
    """Compute normalised discrete Fréchet distance for one pair of routes.

    Args:
        xy_a: Resampled UTM coordinates for route A, shape (n, 2).
        xy_b: Resampled UTM coordinates for route B, shape (m, 2).
        len_a: Arc length of route A in metres.
        len_b: Arc length of route B in metres.
        length_ratio_max: Routes longer than this multiple of each other
            are skipped without computing a full shape comparison.

    Returns:
        Fréchet distance normalised by the mean route length of the pair,
        or ``inf`` if the length ratio pre-filter rejects the pair.
    """
    if len_a > len_b * length_ratio_max or len_b > len_a * length_ratio_max:
        return np.inf

    raw = frechet_distance(LineString(xy_a), LineString(xy_b))
    return raw / ((len_a + len_b) / 2.0)


def route_pairs(
    route_list: list[dict], config: RouteClusterConfig
) -> Iterator[tuple]:
    """Generate argument tuples for all pairs in route_list."""
    n = len(route_list)
    for i in range(n):
        for j in range(i + 1, n):
            yield (
                route_list[i][0],
                route_list[j][0],
                route_list[i][1],
                route_list[j][1],
                config.length_ratio_max,
            )


def symmetric_matrix(n: int, upper_tri: list[float]) -> np.ndarray:
    """Build a symmetric (N, N) matrix from upper-triangle values."""
    rows, cols = np.triu_indices(n, k=1)
    matrix = np.zeros((n, n))
    matrix[rows, cols] = upper_tri
    matrix[cols, rows] = upper_tri
    return matrix


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_partition(
    distance_matrix: np.ndarray,
    config: RouteClusterConfig,
) -> np.ndarray:
    """Run DBSCAN on a precomputed distance matrix.

    Args:
        distance_matrix: Symmetric (N, N) distance matrix.
        config: Clustering configuration.

    Returns:
        Label array of length N. -1 indicates noise (unmatched).
    """
    # sklearn rejects non-finite values; replace inf (pre-filtered pairs) with
    # a sentinel larger than any eps.
    safe = np.nan_to_num(distance_matrix, copy=True, posinf=1e9)

    return DBSCAN(
        metric="precomputed",
        eps=config.similarity_eps,
        min_samples=config.min_samples,
    ).fit_predict(safe)


def _centroid_pos(pos_dicts: list[dict]) -> dict:
    """Median of per-route position dicts."""
    return {
        key: float(np.median([d[key] for d in pos_dicts]))
        for key in ("start_lat", "start_lon", "end_lat", "end_lon")
    }


def partition_by_location(
    valid_routes: list[pd.DataFrame],
    config: RouteClusterConfig,
) -> tuple[list[list[int]], list[dict]]:
    """Group route indices by shared start/end location using haversine DBSCAN.

    Uses haversine distance on lat/lon so that routes anywhere in the world
    are compared in a globally consistent coordinate system.

    Args:
        valid_routes: Trimmed lat/lon DataFrames, one per route.
        config: Clustering configuration.

    Returns:
        Tuple of:
        - Route-index groups; only groups with at least ``config.min_samples``
          members are included.
        - Per-route position dicts, aligned to ``valid_routes``.
    """
    pos_dicts = [
        {
            "start_lat": float(df["latitude"].iloc[0]),
            "start_lon": float(df["longitude"].iloc[0]),
            "end_lat": float(df["latitude"].iloc[-1]),
            "end_lon": float(df["longitude"].iloc[-1]),
        }
        for df in valid_routes
    ]

    start_pts = np.radians(
        [(d["start_lat"], d["start_lon"]) for d in pos_dicts]
    )
    end_pts = np.radians([(d["end_lat"], d["end_lon"]) for d in pos_dicts])

    # min_samples=1 so every route gets a partition label.
    start_labels = DBSCAN(
        eps=config.partition_eps_rad, metric="haversine", min_samples=1
    ).fit_predict(start_pts)
    end_labels = DBSCAN(
        eps=config.partition_eps_rad, metric="haversine", min_samples=1
    ).fit_predict(end_pts)

    partitions = {}
    for i, (sl, el) in enumerate(zip(start_labels, end_labels)):
        partitions.setdefault((int(sl), int(el)), []).append(i)

    groups = [m for m in partitions.values() if len(m) >= config.min_samples]
    return groups, pos_dicts


def resample_partition(
    members: list[int],
    raw_routes: list[pd.DataFrame],
    route_pos_dicts: list[dict],
    config: RouteClusterConfig,
) -> tuple[list[int], list[tuple[np.ndarray, float]]]:
    """Project and resample all routes in a partition using a shared UTM zone.

    The UTM zone is derived from the median start lat/lon of the partition,
    ensuring all routes use a locally appropriate projection.

    Args:
        members: Indices into ``raw_routes`` for this partition.
        raw_routes: Trimmed lat/lon DataFrames for all routes.
        route_pos_dicts: Per-route position dicts, aligned to ``raw_routes``.
        config: Clustering configuration.

    Returns:
        Tuple of:
        - ``surviving_members``: subset of ``members`` whose routes have
          non-zero length after projection (preserves order).
        - ``resampled``: corresponding ``(route_xy, route_length_m)`` tuples.
    """
    centroid = _centroid_pos([route_pos_dicts[i] for i in members])
    _, _, zone_number, zone_letter = utm.from_latlon(
        centroid["start_lat"], centroid["start_lon"]
    )

    surviving_members = []
    resampled = []
    for i in members:
        df = raw_routes[i]
        result = resample_route(
            df["latitude"].to_numpy(dtype=float),
            df["longitude"].to_numpy(dtype=float),
            zone_number,
            zone_letter,
            config,
        )
        if result is not None:
            surviving_members.append(i)
            resampled.append(result)

    return surviving_members, resampled


def partition_and_cluster(
    valid_idx: list,
    valid_routes: list[pd.DataFrame],
    config: RouteClusterConfig,
) -> tuple[list[list], dict]:
    """Partition routes by start/end location and cluster within each partition.

    Args:
        valid_idx: Activity index values corresponding to each route.
        valid_routes: Trimmed lat/lon data from ``extract_route_features``.
        config: Clustering configuration.

    Returns:
        Tuple of:
        - Activity index values grouped into clusters, ordered by cluster size
          descending (largest first).
        - Per-activity position dict keyed by activity index.
    """
    raw_partitions, route_pos_dicts = partition_by_location(
        valid_routes, config
    )
    act_pos_dicts = dict(zip(valid_idx, route_pos_dicts))

    # Drop partitions that fall below min_samples after resampling (routes
    # with zero length after NaN filtering are silently excluded).
    resampled_partitions = []
    member_partitions = []
    for members in raw_partitions:
        surviving, resampled = resample_partition(
            members, valid_routes, route_pos_dicts, config
        )
        if len(resampled) >= config.min_samples:
            resampled_partitions.append(resampled)
            member_partitions.append(surviving)

    all_clusters = []
    for members, resampled in zip(member_partitions, resampled_partitions):
        results = [frechet_pair(*p) for p in route_pairs(resampled, config)]
        labels = cluster_partition(
            symmetric_matrix(len(resampled), results), config
        )

        clusters = {}
        for i, label in enumerate(labels):
            if label < 0:
                continue
            clusters.setdefault(int(label), []).append(valid_idx[members[i]])

        all_clusters.extend(clusters.values())

    all_clusters.sort(key=len, reverse=True)
    return all_clusters, act_pos_dicts


def compute_clusters(
    activities: pd.DataFrame,
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    preloaded_coords: dict[tuple[str, int | None], pd.DataFrame | None] | None,
    config: RouteClusterConfig,
) -> list[ClusterResult]:
    """Cluster bicycle activities by GPS route similarity or activity name.

    Activities with GPS data are clustered by discrete Fréchet distance. The
    GPS pipeline is computed as follows:
    1. Load latitude/longitude data for each activity file.
    2. Extract start/end positions per route and partition activities by shared
       start and end locations.
    3. Project each partition into its local UTM zone and resample routes to a
       point count proportional to route length.
    4. Compute pairwise discrete Fréchet distances within each partition.
    5. Cluster within each partition on the distance matrix.
    6. Assign each cluster a representative start/end position (median of member
       routes); unmatched GPS activities retain their raw start/end.

    File-based activities that yield no GPS data (e.g. trainer rides on a
    non-simulated course) are clustered by activity description instead.

    All clusters are merged into single ID space ordered by size (0 = most
    frequent). Activities with no file are not clustered.

    Args:
        activities: Activity data with filename and name columns per ``config``.
        segments: Per-activity segment indices, or ``None`` to treat all
            activities as whole-file.
        path: Strava export directory.
        cache_dir: Optional records parquet cache directory.
        preloaded_coords: Optional mapping of ``(filename, segment)`` to
            already-extracted lat/lon DataFrames. Passed to
            ``extract_route_features``; files not present are batch-loaded.
        config: Clustering configuration.

    Returns:
        List of ``ClusterResult`` aligned to ``activities``, one per row.
    """
    has_file = activities[config.filename_col].notna()
    file_segments = (
        (seg for seg, keep in zip(segments, has_file) if keep)
        if segments is not None
        else None
    )
    valid_idx, valid_routes = extract_route_features(
        activities[has_file],
        file_segments,
        path,
        cache_dir,
        preloaded_coords,
        config,
    )

    # Start every activity as unmatched
    results = [ClusterResult() for _ in activities.index]

    # GPS clustering, named by modal activity description
    gps_clusters, act_pos_dicts = (
        partition_and_cluster(valid_idx, valid_routes, config)
        if valid_routes
        else ([], {})
    )
    gps_clusters_named = [
        (activities.loc[idx_list, config.name_col].mode().iat[0], idx_list)
        for idx_list in gps_clusters
    ]

    # Name-based clustering of activities that yielded no GPS data
    no_gps_mask = has_file & ~activities.index.isin(valid_idx)
    desc = activities.loc[no_gps_mask, config.name_col].dropna()
    name_clusters = [
        (name, list(group.index))
        for name, group in desc.groupby(desc)
        if len(group) >= config.min_samples
    ]

    # Merge and renumber by size (0 = most frequent)
    pos_of = {idx: pos for pos, idx in enumerate(activities.index)}
    all_clusters = sorted(
        [(name, idx_list, True) for name, idx_list in gps_clusters_named]
        + [(name, idx_list, False) for name, idx_list in name_clusters],
        key=lambda x: len(x[1]),
        reverse=True,
    )
    for global_id, (cluster_name, idx_list, has_gps) in enumerate(all_clusters):
        pos = (
            _centroid_pos([act_pos_dicts[idx] for idx in idx_list])
            if has_gps
            else {}
        )
        for act_idx in idx_list:
            results[pos_of[act_idx]] = ClusterResult(
                cluster_id=global_id, cluster_name=cluster_name, **pos
            )

    # Populate raw start/end for GPS activities that didn't form a cluster
    clustered_idx = {idx for idx_list in gps_clusters for idx in idx_list}
    for act_idx in valid_idx:
        if act_idx not in clustered_idx:
            results[pos_of[act_idx]] = ClusterResult(**act_pos_dicts[act_idx])

    if config.geocoding is not None:
        addresses = geocoding.geocode_positions(
            itertools.chain(
                (
                    (r.start_lat, r.start_lon)
                    for r in results
                    if r.start_lat is not None
                ),
                (
                    (r.end_lat, r.end_lon)
                    for r in results
                    if r.end_lat is not None
                ),
            ),
            cache_dir,
            config.geocoding,
        )

        for result in results:
            if result.start_lat is not None:
                result.start_address = addresses.get(
                    (result.start_lat, result.start_lon)
                )
            if result.end_lat is not None:
                result.end_address = addresses.get(
                    (result.end_lat, result.end_lon)
                )

    return results


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def cluster_routes(
    activities: pd.DataFrame,
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
    config: RouteClusterConfig | None = None,
) -> pd.DataFrame:
    """Cluster bicycle activities by GPS route similarity or activity name.

    Activities with GPS data are clustered by route similarity. File-based
    activities that yield no GPS data are clustered by activity description
    instead. Activities with no file are not clustered. All clusters share a
    single ID space ordered by size (0 = most frequent).

    Args:
        activities: Activity data from ``load_strava_activities`` or
            ``load_strava_activities_raw``.
        segments: Per-activity segment indices, or ``None`` to treat all
            activities as whole-file.
        path: Strava export directory.
        cache_dir: Optional cache directory for the records parquet cache.
            If omitted, activity files are parsed on every call.
        config: Clustering parameters. Defaults to ``RouteClusterConfig()``.

    Returns:
        DataFrame with the same index as ``activities`` containing:
        - ``cluster_id``: Integer cluster ID (0 = most frequent route),
          ``pd.NA`` for unmatched or no-file activities.
        - ``cluster_name``: Mode of activity names within the cluster,
          ``None`` for unmatched activities.
        - ``start_lat``, ``start_lon``: Representative start position.
          Cluster centroid (median) for GPS-clustered activities; raw
          route start for GPS activities below ``min_samples``; ``None``
          for activities without GPS data.
        - ``end_lat``, ``end_lon``: Representative end position (same
          semantics as ``start_lat``/``start_lon``).
    """
    if config is None:
        config = RouteClusterConfig()

    return pd.DataFrame(
        compute_clusters(activities, segments, path, cache_dir, None, config),
        index=activities.index,
    ).astype({"cluster_id": "Int64"})


def cluster_routes_cached(
    activities: pd.DataFrame,
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    table: Literal["activities", "commutes"],
    preloaded_coords: dict[tuple[str, int | None], pd.DataFrame | None]
    | None = None,
    config: RouteClusterConfig | None = None,
) -> pd.DataFrame:
    """Cluster bicycle routes, reading from and writing to the cache DB.

    Uses a fingerprint of the (filename, segment) pairs and clustering config
    to detect staleness. Any change to the activity set or config parameters
    forces a full recompute; otherwise cluster columns are read directly from
    the DB.

    Cache persistence (SELECT, UPDATE, fingerprint upsert) is handled
    internally. The rows being clustered must already exist in the DB before
    this function is called so that the UPDATE has rows to write into.

    Args:
        activities: Activity DataFrame. Must have columns matching
            ``config.filename_col`` and ``config.name_col``.
        segments: Per-activity segment indices, or ``None`` to treat all
            activities as whole-file.
        path: Strava export directory (passed to ``cluster_routes``).
        cache_dir: Records cache directory. If ``None``, clustering is always
            computed and no DB I/O is performed.
        table: DB table to read from and write cluster columns into.
        preloaded_coords: Optional mapping of ``(filename, segment)`` to
            already-extracted lat/lon DataFrames. Passed to
            ``compute_clusters`` on a stale-fingerprint recompute; ignored
            on a cache hit.
        config: Clustering configuration.

    Returns:
        DataFrame identical to ``cluster_routes``.
    """
    if config is None:
        config = RouteClusterConfig()

    if cache_dir is None:
        return pd.DataFrame(
            compute_clusters(
                activities,
                segments,
                path,
                cache_dir,
                preloaded_coords,
                config,
            ),
            index=activities.index,
        ).astype({"cluster_id": "Int64"})

    filenames = activities[config.filename_col]
    segs = segments if segments is not None else itertools.repeat(None)
    keys = [cache_db.cache_key(fn, seg) for fn, seg in zip(filenames, segs)]
    expected_fp = compute_cluster_fingerprint(keys, config)

    with cache_db.open_db(cache_dir) as db:
        fp_row = next(
            db["cluster_fingerprints"].rows_where("table_name = ?", [table]),
            None,
        )
        stored_fp = fp_row["fingerprint"] if fp_row else None

        if stored_fp == expected_fp:
            fns = list({k[0] for k in keys if k is not None})
            marks = ",".join("?" * len(fns))
            rows = (
                db[table].rows_where(f"filename IN ({marks})", fns)
                if fns
                else []
            )
            lookup = {
                (r["filename"], r["segment"]): ClusterResult.from_db_dict(r)
                for r in rows
            }
            results = [lookup.get(k, ClusterResult()) for k in keys]

            return pd.DataFrame(results, index=activities.index).astype(
                {"cluster_id": "Int64"}
            )

    results = compute_clusters(
        activities,
        segments,
        path,
        cache_dir,
        preloaded_coords,
        config,
    )

    # Raw SQL so the batch UPDATE and fingerprint INSERT commit atomically
    with cache_db.open_db(cache_dir) as db:
        with db.conn:
            db.conn.executemany(
                ClusterResult.update_sql(table),
                [
                    cr.to_update_params(k[0], k[1])
                    for cr, k in zip(results, keys)
                    if k is not None
                ],
            )
            db.conn.execute(
                "INSERT OR REPLACE INTO cluster_fingerprints"
                " (table_name, fingerprint) VALUES (?, ?)",
                (table, expected_fp),
            )

    return pd.DataFrame(results, index=activities.index).astype(
        {"cluster_id": "Int64"}
    )
