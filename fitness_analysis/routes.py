"""GPS route clustering for bicycle activities."""

import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from os import PathLike

import numpy as np
import pandas as pd
import similaritymeasures
import utm
from sklearn.cluster import DBSCAN

from . import records

# Pairs/core below which serial beats ProcessPoolExecutor. Tuned on M1 Pro.
PARALLEL_PAIRS_PER_CPU = 2500


@dataclass
class RouteClusterConfig:
    """Configuration parameters for ``cluster_routes``.

    Attributes:
        points_per_km: Number of resampled points per km of route length.
        points_min: Minimum number of resampled points (short routes).
        points_max: Maximum number of resampled points (long routes).
        partition_eps_m: DBSCAN eps (metres) for grouping activities by
            start/end location into independent partitions.
        similarity_eps: DBSCAN eps for route similarity - fraction of mean
            route length (0.05 = 5% deviation allowed).
        min_samples: DBSCAN ``min_samples`` for route clustering.
        length_ratio_max: Pre-filter threshold - skip Fréchet for pairs whose
            route lengths differ by more than this factor.
    """

    points_per_km: float = 1.0
    points_min: int = 20
    points_max: int = 100
    partition_eps_m: float = 750.0
    similarity_eps: float = 0.02
    min_samples: int = 2
    length_ratio_max: float = 1.2


# ---------------------------------------------------------------------------
# GPS preprocessing
# ---------------------------------------------------------------------------


def resample_route(
    lat: np.ndarray,
    lon: np.ndarray,
    config: RouteClusterConfig,
    zone_number: int,
    zone_letter: str,
) -> tuple[np.ndarray, float] | None:
    """Project a GPS route to UTM and arc-length resample it.

    Uses a caller-supplied UTM zone so all routes in a dataset share a
    consistent coordinate system.

    Args:
        lat: Latitude values (may contain NaN).
        lon: Longitude values (may contain NaN).
        config: Clustering configuration.
        zone_number: UTM zone number to force.
        zone_letter: UTM zone letter to force.

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

    n_points = max(
        config.points_min,
        min(
            config.points_max,
            int(route_length_m / 1000.0 * config.points_per_km),
        ),
    )
    sample_distances = np.linspace(0.0, route_length_m, n_points)
    route_xy = np.column_stack(
        [
            np.interp(sample_distances, cum_len, xy[:, 0]),
            np.interp(sample_distances, cum_len, xy[:, 1]),
        ]
    )

    return route_xy, route_length_m


def load_raw_coords(
    record_list: list[pd.DataFrame],
) -> tuple[list[tuple[np.ndarray, np.ndarray] | None], tuple[int, str] | None]:
    """Extract lat/lon arrays from records and determine a shared UTM zone.

    Args:
        record_list: Parsed activity record DataFrames.

    Returns:
        Tuple of:

        - ``raw_coords``: One entry per record - either ``(lat, lon)`` numpy
          arrays or ``None`` if the record has no GPS columns.
        - ``utm_zone``: ``(zone_number, zone_letter)`` derived from the median
          first-point coordinates across all GPS records, or ``None`` if no
          records have GPS data.
    """

    raw_coords = []
    first_lats = []
    first_lons = []

    for rec in record_list:
        if "latitude" not in rec.columns or "longitude" not in rec.columns:
            raw_coords.append(None)
            continue

        lat = rec["latitude"].to_numpy(dtype=float)
        lon = rec["longitude"].to_numpy(dtype=float)

        valid = np.isfinite(lat)
        if not valid.any():
            raw_coords.append(None)
            continue

        first_lats.append(float(lat[valid][0]))
        first_lons.append(float(lon[valid][0]))
        raw_coords.append((lat, lon))

    if not first_lats:
        return raw_coords, None

    _, _, zone_number, zone_letter = utm.from_latlon(
        float(np.median(first_lats)), float(np.median(first_lons))
    )
    return raw_coords, (zone_number, zone_letter)


def extract_route_features(
    gps_activities: pd.DataFrame,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
    config: RouteClusterConfig,
) -> tuple[list, list[tuple[np.ndarray, float]]]:
    """Load GPS records and resample all routes to UTM feature dicts.

    Args:
        gps_activities: Subset of the activity DataFrame with ``filename``
            present and ``trainer`` False.
        path: Strava export directory.
        cache_dir: Optional records parquet cache directory.
        config: Clustering configuration.

    Returns:
        Tuple of ``(valid_idx, valid_routes)`` where ``valid_idx`` are the
        original activity index values and ``valid_routes`` are the
        corresponding ``(route_xy, route_length_m)`` tuples from
        ``resample_route``.
    """

    record_list = records.load_activity_records(
        gps_activities["filename"], path, cache_dir
    )
    raw_coords, utm_zone = load_raw_coords(record_list)

    if utm_zone is None:
        return [], []

    zone_number, zone_letter = utm_zone
    routes = [
        resample_route(*entry, config, zone_number, zone_letter)
        if entry is not None
        else None
        for entry in raw_coords
    ]

    gps_idx = list(gps_activities.index)
    valid_idx = [idx for idx, r in zip(gps_idx, routes) if r is not None]
    valid_routes = [r for r in routes if r is not None]
    return valid_idx, valid_routes


# ---------------------------------------------------------------------------
# Pairwise Fréchet distance
# ---------------------------------------------------------------------------


def frechet_pair(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    len_a: float,
    len_b: float,
    config: RouteClusterConfig,
) -> float:
    """Compute normalised discrete Fréchet distance for one pair of routes.

    Args:
        xy_a: Resampled UTM coordinates for route A, shape (n, 2).
        xy_b: Resampled UTM coordinates for route B, shape (m, 2).
        len_a: Arc length of route A in metres.
        len_b: Arc length of route B in metres.
        config: Clustering configuration.

    Returns:
        Fréchet distance normalised by the mean route length of the pair,
        or ``inf`` if the length ratio pre-filter rejects the pair.
    """

    ratio = config.length_ratio_max
    if len_a > len_b * ratio or len_b > len_a * ratio:
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
                config,
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


def partition_and_cluster(
    valid_idx: list,
    valid_routes: list[tuple[np.ndarray, float]],
    config: RouteClusterConfig,
) -> list[list]:
    """Partition routes by start/end location and cluster within each partition.

    Fréchet distances across all partitions are computed in a single parallel
    batch when the total pair count justifies it.

    Args:
        valid_idx: Activity index values corresponding to each feature dict.
        valid_routes: ``(route_xy, route_length_m)`` tuples from
            ``resample_route``.
        config: Clustering configuration.

    Returns:
        List of clusters, each cluster being a list of activity index values.
        Ordered by cluster size descending (largest first).
    """

    start_pts = np.array([r[0][0] for r in valid_routes])
    end_pts = np.array([r[0][-1] for r in valid_routes])

    # min_samples=1 so every start and end point gets a partition label.
    start_labels = DBSCAN(
        eps=config.partition_eps_m, metric="euclidean", min_samples=1
    ).fit_predict(start_pts)
    end_labels = DBSCAN(
        eps=config.partition_eps_m, metric="euclidean", min_samples=1
    ).fit_predict(end_pts)

    partitions = {}
    for i, (sl, el) in enumerate(zip(start_labels, end_labels)):
        partitions.setdefault((int(sl), int(el)), []).append(i)

    valid_partitions = [
        m for m in partitions.values() if len(m) >= config.min_samples
    ]
    pair_counts = [len(m) * (len(m) - 1) // 2 for m in valid_partitions]
    n_pairs_total = sum(pair_counts)

    def all_pairs():
        for members in valid_partitions:
            yield from route_pairs([valid_routes[i] for i in members], config)

    if n_pairs_total > os.cpu_count() * PARALLEL_PAIRS_PER_CPU:
        with ProcessPoolExecutor() as ex:
            all_results = list(ex.map(_frechet_pair_packed, all_pairs()))
    else:
        all_results = [frechet_pair(*p) for p in all_pairs()]

    all_clusters = []
    offset = 0
    for members, n_pairs in zip(valid_partitions, pair_counts):
        results = all_results[offset : offset + n_pairs]
        offset += n_pairs

        local_labels = cluster_partition(
            symmetric_matrix(len(members), results), config
        )
        local_clusters = {}
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
# Top-level entry point
# ---------------------------------------------------------------------------


def cluster_routes(
    activities: pd.DataFrame,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
    config: RouteClusterConfig | None = None,
) -> pd.DataFrame:
    """Cluster bicycle activities by GPS route similarity.

    Identifies groups of rides that follow the same route. Activities without
    GPS data (trainer rides, activities with no location) receive NaN cluster
    IDs and are not clustered.

    The pipeline:
    1. Resamples each GPS route to a fixed point count (proportional to
       route length) in UTM coordinates.
    2. Partitions activities into independent groups based on shared
       start and end locations (DBSCAN with ``partition_eps_m``).
    3. Computes pairwise discrete Fréchet distances within each partition.
    4. Clusters within each partition via DBSCAN on the distance matrix.
    5. Assigns global ``cluster_id`` values ordered by cluster size
       (0 = most frequent route).

    Args:
        activities: Activity DataFrame from ``load_strava_activities``. Must
            have ``filename``, ``trainer``, and ``description`` columns.
        path: Strava export directory (passed to record loading).
        cache_dir: Optional cache directory for the records parquet cache.
            If omitted, activity files are parsed on every call.
        config: Clustering parameters. Defaults to ``RouteClusterConfig()``.

    Returns:
        DataFrame with the same index as ``activities`` containing:

        - ``cluster_id``: Integer cluster ID (0 = most frequent route),
          ``pd.NA`` for unmatched or GPS-less activities.
        - ``cluster_name``: Mode of activity names within the cluster,
          ``None`` for unmatched activities.
    """

    if config is None:
        config = RouteClusterConfig()

    gps_mask = activities["filename"].notna() & ~activities["trainer"]
    valid_idx, valid_routes = extract_route_features(
        activities[gps_mask], path, cache_dir, config
    )

    cluster_id_arr = pd.array([pd.NA] * len(activities), dtype=pd.Int64Dtype())
    cluster_name_arr = np.full(len(activities), None, dtype=object)

    if not valid_routes:
        return pd.DataFrame(
            {"cluster_id": cluster_id_arr, "cluster_name": cluster_name_arr},
            index=activities.index,
        )

    all_clusters = partition_and_cluster(valid_idx, valid_routes, config)

    pos_of = {idx: pos for pos, idx in enumerate(activities.index)}
    for global_id, idx_list in enumerate(all_clusters):
        mode = activities.loc[idx_list, "description"].mode()
        cluster_name = mode.iloc[0] if len(mode) > 0 else None
        for act_idx in idx_list:
            cluster_id_arr[pos_of[act_idx]] = global_id
            cluster_name_arr[pos_of[act_idx]] = cluster_name

    return pd.DataFrame(
        {"cluster_id": cluster_id_arr, "cluster_name": cluster_name_arr},
        index=activities.index,
    )
