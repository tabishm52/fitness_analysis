"""GPS route clustering for bicycle activities."""

import itertools
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from os import PathLike

import numpy as np
import pandas as pd
import similaritymeasures
import utm
from sklearn.cluster import DBSCAN

from . import records

# Parallel processing: min_pairs is where pool startup overhead is justified;
# chunk_size balances IPC overhead vs load balance. (tuned on Apple M1 Pro)
PARALLEL_MIN_PAIRS = 30000
PARALLEL_CHUNK_SIZE = 50

# WGS-84 semi-major axis in metres, consistent with the UTM projection
EARTH_RADIUS_M: float = 6_378_137.0


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
    """

    points_per_km: float = 1.0
    points_min: int = 20
    points_max: int = 100
    partition_eps_m: float = 750.0
    similarity_eps: float = 0.02
    min_samples: int = 2
    length_ratio_max: float = 1.2

    @property
    def partition_eps_rad(self) -> float:
        """``partition_eps_m`` converted to radians."""
        return self.partition_eps_m / EARTH_RADIUS_M

    # If True, cluster_routes() uses Strava CSV column names. If False,
    # cluster_routes() uses column names from load_strava_activities().
    raw_csv: bool = field(default=False, init=False, repr=False)

    @property
    def filename_col(self) -> str:
        return "Filename" if self.raw_csv else "filename"

    @property
    def name_col(self) -> str:
        return "Activity Name" if self.raw_csv else "description"


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
        config: Clustering configuration.

    Returns:
        Tuple of:
        - ``valid_idx``: the original activity index values.
        - ``valid_routes``: corresponding trimmed lat/lon DataFrames, one per
          valid route (``latitude`` and ``longitude`` columns, NaN-trimmed).
    """

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

    raw = similaritymeasures.frechet_dist(xy_a, xy_b)
    return raw / ((len_a + len_b) / 2.0)


def _frechet_pair_packed(args: tuple) -> float:
    """Picklable single-argument wrapper around ``frechet_pair``."""

    return frechet_pair(*args)


def route_pairs(
    route_list: list[dict], config: RouteClusterConfig
) -> Iterator[tuple]:
    """Generate argument tuples for all pairs in route_list.

    Suitable for unpacking into ``frechet_pair`` or passing to
    ``_frechet_pair_packed``.
    """

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


def partition_by_location(
    valid_routes: list[pd.DataFrame],
    config: RouteClusterConfig,
) -> list[list[int]]:
    """Group route indices by shared start/end location using haversine DBSCAN.

    Uses haversine distance on lat/lon so that routes anywhere in the world
    are compared in a globally consistent coordinate system.

    Args:
        valid_routes: Trimmed lat/lon DataFrames, one per route.
        config: Clustering configuration.

    Returns:
        List of route-index groups; only groups with at least
        ``config.min_samples`` members are included.
    """

    start_lats = np.radians([df["latitude"].iloc[0] for df in valid_routes])
    start_lons = np.radians([df["longitude"].iloc[0] for df in valid_routes])
    end_lats = np.radians([df["latitude"].iloc[-1] for df in valid_routes])
    end_lons = np.radians([df["longitude"].iloc[-1] for df in valid_routes])

    start_pts = np.column_stack([start_lats, start_lons])
    end_pts = np.column_stack([end_lats, end_lons])

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

    return [m for m in partitions.values() if len(m) >= config.min_samples]


def resample_partition(
    members: list[int],
    raw_routes: list[pd.DataFrame],
    config: RouteClusterConfig,
) -> tuple[list[int], list[tuple[np.ndarray, float]]]:
    """Project and resample all routes in a partition using a shared UTM zone.

    The UTM zone is derived from the median start lat/lon of the partition,
    ensuring all routes use a locally appropriate projection.

    Args:
        members: Indices into ``raw_routes`` for this partition.
        raw_routes: Trimmed lat/lon DataFrames for all routes.
        config: Clustering configuration.

    Returns:
        Tuple of:
        - ``surviving_members``: subset of ``members`` whose routes have
          non-zero length after projection (preserves order).
        - ``resampled``: corresponding ``(route_xy, route_length_m)`` tuples.
    """

    start_lats = [raw_routes[i]["latitude"].iloc[0] for i in members]
    start_lons = [raw_routes[i]["longitude"].iloc[0] for i in members]
    centroid_lat = float(np.median(start_lats))
    centroid_lon = float(np.median(start_lons))
    _, _, zone_number, zone_letter = utm.from_latlon(centroid_lat, centroid_lon)

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
) -> list[list]:
    """Partition routes by start/end location and cluster within each partition.

    Fréchet distances across all partitions are computed in a single parallel
    batch when the total pair count justifies it.

    Args:
        valid_idx: Activity index values corresponding to each route.
        valid_routes: Trimmed lat/lon data from ``extract_route_features``.
        config: Clustering configuration.

    Returns:
        Activity index values grouped into clusters, ordered by cluster size
        descending (largest first).
    """

    raw_partitions = partition_by_location(valid_routes, config)

    # Drop partitions that fall below min_samples after resampling (routes
    # with zero length after NaN filtering are silently excluded).
    resampled_partitions = []
    member_partitions = []
    for members in raw_partitions:
        surviving, resampled = resample_partition(members, valid_routes, config)
        if len(resampled) >= config.min_samples:
            resampled_partitions.append(resampled)
            member_partitions.append(surviving)

    pair_counts = [len(r) * (len(r) - 1) // 2 for r in resampled_partitions]
    n_pairs_total = sum(pair_counts)

    def all_pairs():
        for resampled in resampled_partitions:
            yield from route_pairs(resampled, config)

    if n_pairs_total >= PARALLEL_MIN_PAIRS:
        with ProcessPoolExecutor() as ex:
            all_results = list(
                ex.map(
                    _frechet_pair_packed,
                    all_pairs(),
                    chunksize=PARALLEL_CHUNK_SIZE,
                )
            )
    else:
        all_results = [frechet_pair(*p) for p in all_pairs()]

    all_clusters = []
    offset = 0
    for members, resampled, n_pairs in zip(
        member_partitions, resampled_partitions, pair_counts
    ):
        results = all_results[offset : offset + n_pairs]
        offset += n_pairs

        local_labels = cluster_partition(
            symmetric_matrix(len(resampled), results), config
        )
        local_clusters: dict[int, list] = {}
        for local_i, label in enumerate(local_labels):
            if label < 0:
                continue
            local_clusters.setdefault(int(label), []).append(
                valid_idx[members[local_i]]
            )
        all_clusters.extend(local_clusters.values())

    all_clusters.sort(key=len, reverse=True)
    return all_clusters


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

    Activities with GPS data are clustered by discrete Fréchet distance.
    File-based activities that yield no GPS data (e.g. trainer rides on a
    non-simulated course) are clustered by activity description instead.
    Activities with no file are not clustered.

    All clusters share a single ID space ordered by size (0 = most frequent).

    The GPS pipeline:
    1. Loads latitude/longitude data for each activity file.
    2. Partitions activities based on shared start and end locations.
    3. Projects each partition into its local UTM zone and resamples routes
       to a point count proportional to route length.
    4. Computes pairwise discrete Fréchet distances within each partition.
    5. Clusters within each partition on the distance matrix.

    Args:
        activities: Activity data from ``load_strava_activities`` or
            ``load_strava_activities_raw``.
        segments: Per-activity segment indices, or ``None`` to treat all
            activities as whole-file.
        path: Strava export directory (passed to record loading).
        cache_dir: Optional cache directory for the records parquet cache.
            If omitted, activity files are parsed on every call.
        config: Clustering parameters. Defaults to ``RouteClusterConfig()``.

    Returns:
        DataFrame with the same index as ``activities`` containing:
        - ``cluster_id``: Integer cluster ID (0 = most frequent route),
          ``pd.NA`` for unmatched or no-file activities.
        - ``cluster_name``: Mode of activity names within the cluster,
          ``None`` for unmatched activities.
    """

    if config is None:
        config = RouteClusterConfig()

    has_file = activities[config.filename_col].notna()
    valid_idx, valid_routes = extract_route_features(
        activities[has_file], segments, path, cache_dir, config
    )

    cluster_id_arr = pd.array([pd.NA] * len(activities), dtype=pd.Int64Dtype())
    cluster_name_arr = np.full(len(activities), None, dtype=object)

    # GPS clustering, named by modal activity description
    gps_clusters = (
        partition_and_cluster(valid_idx, valid_routes, config)
        if valid_routes
        else []
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
        gps_clusters_named + name_clusters,
        key=lambda x: len(x[1]),
        reverse=True,
    )
    for global_id, (cluster_name, idx_list) in enumerate(all_clusters):
        for act_idx in idx_list:
            cluster_id_arr[pos_of[act_idx]] = global_id
            cluster_name_arr[pos_of[act_idx]] = cluster_name

    return pd.DataFrame(
        {"cluster_id": cluster_id_arr, "cluster_name": cluster_name_arr},
        index=activities.index,
    )


def cluster_cache_key(fn, seg) -> tuple | None:
    """Return ``(fn, int(seg))`` or ``None`` if fn is NaN.

    Uses -1 as a sentinel for whole-file (None) segments so keys are hashable.
    """

    if pd.isna(fn):
        return None
    return (fn, -1 if seg is None or pd.isna(seg) else int(seg))


def cluster_routes_cached(  # noqa: PLR0913
    activities: pd.DataFrame,
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
    cache_df: pd.DataFrame | None = None,
    config: RouteClusterConfig = RouteClusterConfig(),
) -> tuple[pd.DataFrame, bool]:
    """Cluster bicycle routes, using a pre-loaded cache when provided.

    Cache hit condition: ``cache_df`` contains ``cluster_id`` and
    ``cluster_name`` columns with entries for every file-based activity in
    ``activities``. Any missing entry triggers a full recompute, as does
    passing ``cache_df=None``. Cache persistence is the caller's
    responsibility.

    Cache keys are always composite ``(filename, segment)`` pairs. When
    ``segments`` is ``None``, a sentinel of ``-1`` is used for the segment
    component. ``cache_df`` rows without a ``"segment"`` column also resolve
    to ``-1``, so the strava path (no segments) works without a segment column.

    Args:
        activities: Activity DataFrame. Must have columns matching
            ``config.filename_col`` and ``config.name_col``.
        segments: Per-activity segment indices, or ``None`` to treat all
            activities as whole-file.
        path: Strava export directory (passed to ``cluster_routes``).
        cache_dir: Records cache directory (passed to ``cluster_routes``).
        cache_df: Already-loaded activities cache indexed by filename.
        config: Clustering configuration.

    Returns:
        Tuple of:
        - ``clusters``: ``cluster_id`` and ``cluster_name`` columns with the
          same index as ``activities``.
        - ``cache_miss``: True when clustering was recomputed.
    """

    cache_cols = {"cluster_id", "cluster_name"}
    if cache_df is None or not cache_cols.issubset(cache_df.columns):
        return (
            cluster_routes(activities, segments, path, cache_dir, config),
            True,
        )

    filenames = activities[config.filename_col]
    segs = segments if segments is not None else itertools.repeat(None)
    result_keys = [
        cluster_cache_key(fn, seg) for fn, seg in zip(filenames, segs)
    ]
    cache_lookup = {
        cluster_cache_key(fn, row.get("segment")): (
            row["cluster_id"],
            row["cluster_name"],
        )
        for fn, row in cache_df.iterrows()
    }

    file_keys = [k for k in result_keys if k is not None]
    if not all(k in cache_lookup for k in file_keys):
        return (
            cluster_routes(activities, segments, path, cache_dir, config),
            True,
        )

    clusters = [
        cache_lookup[k] if k is not None else (pd.NA, None) for k in result_keys
    ]
    return pd.DataFrame(
        {
            "cluster_id": pd.array([c[0] for c in clusters], dtype="Int64"),
            "cluster_name": [c[1] for c in clusters],
        },
        index=activities.index,
    ), False
