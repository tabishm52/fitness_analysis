"""Tests for GPS route clustering."""

import dataclasses

import gpx_fixtures as gf
import numpy as np
import pandas as pd
import pytest
import utm
from test_geocoding import FakeProvider

from fitness_analysis import cache_db, geocoding, routes

# ~5 km straight line at SF's latitude, well above the default points_min=20 clamp.
_ZONE_NUMBER, _ZONE_LETTER = utm.from_latlon(*gf.SF)[2:]

_PreloadedCoords = dict[tuple[str, int | None], pd.DataFrame | None]


def _activities(rows: list[dict], dates: list[str]) -> pd.DataFrame:
    """Builds a non-raw_csv-style activities DataFrame (filename/description columns)."""
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


# --------------------------------------------------------------------------------------
# RouteClusterConfig
# --------------------------------------------------------------------------------------


def test_route_cluster_config_partition_eps_rad_conversion():
    config = routes.RouteClusterConfig(partition_eps_m=6_378_137.0)  # one Earth radius
    assert config.partition_eps_rad == pytest.approx(1.0)


def test_route_cluster_config_raw_csv_toggles_column_names():
    config = routes.RouteClusterConfig()
    assert (config.filename_col, config.name_col) == ("filename", "description")

    config.raw_csv = True
    assert (config.filename_col, config.name_col) == ("Filename", "Activity Name")


# --------------------------------------------------------------------------------------
# ClusterResult
# --------------------------------------------------------------------------------------


def test_cluster_result_update_sql_shape():
    sql = routes.ClusterResult.update_sql("activities")
    assert sql.startswith("UPDATE activities SET ")
    assert sql.endswith("WHERE filename=? AND segment=?")
    for f in dataclasses.fields(routes.ClusterResult):
        assert f"{f.name}=?" in sql


def test_cluster_result_to_update_params_appends_key():
    cr = routes.ClusterResult(cluster_id=1, cluster_name="Loop")
    params = cr.to_update_params("ride.gpx", 2)
    assert params[0] == 1
    assert params[-2:] == ("ride.gpx", 2)


def test_cluster_result_from_db_dict_roundtrip():
    cr = routes.ClusterResult(
        cluster_id=1, cluster_name="Loop", start_lat=1.0, start_lon=2.0, end_lat=3.0, end_lon=4.0
    )
    row = {f.name: getattr(cr, f.name) for f in dataclasses.fields(cr)}
    assert routes.ClusterResult.from_db_dict(row) == cr


# --------------------------------------------------------------------------------------
# compute_cluster_fingerprint
# --------------------------------------------------------------------------------------


def test_compute_cluster_fingerprint_deterministic():
    config = routes.RouteClusterConfig(geocoding=None)
    keys = [("a.gpx", -1), ("b.gpx", -1)]
    assert routes.compute_cluster_fingerprint(keys, config) == routes.compute_cluster_fingerprint(
        keys, config
    )


def test_compute_cluster_fingerprint_ignores_order_duplicates_and_none():
    config = routes.RouteClusterConfig(geocoding=None)
    keys1 = [("a.gpx", -1), ("b.gpx", -1)]
    keys2 = [None, ("b.gpx", -1), ("a.gpx", -1), ("a.gpx", -1), None]
    assert routes.compute_cluster_fingerprint(keys1, config) == routes.compute_cluster_fingerprint(
        keys2, config
    )


def test_compute_cluster_fingerprint_changes_with_config():
    keys = [("a.gpx", -1)]
    fp1 = routes.compute_cluster_fingerprint(
        keys, routes.RouteClusterConfig(geocoding=None, min_samples=2)
    )
    fp2 = routes.compute_cluster_fingerprint(
        keys, routes.RouteClusterConfig(geocoding=None, min_samples=3)
    )
    assert fp1 != fp2


def test_compute_cluster_fingerprint_identical_with_and_without_provider():
    """Companion to the provider seam: a live provider must not affect the fingerprint,
    and asdict-ing it (which deep-copies non-dataclass fields) must not raise."""
    keys = [("a.gpx", -1)]
    fp_no_provider = routes.compute_cluster_fingerprint(keys, routes.RouteClusterConfig())

    # A real (non-fake) provider: its RateLimiter wraps a thread lock, which
    # copy.deepcopy cannot handle, so this exercises the actual failure mode.
    provider = geocoding.GeocodingProvider.from_env("FITNESS_ANALYSIS_TEST_UNSET_VAR")
    config_with_provider = routes.RouteClusterConfig(
        geocoding=geocoding.GeocodingConfig(provider=provider)
    )
    fp_with_provider = routes.compute_cluster_fingerprint(keys, config_with_provider)

    assert fp_no_provider == fp_with_provider


# --------------------------------------------------------------------------------------
# resample_route
# --------------------------------------------------------------------------------------


def test_resample_route_zero_length_returns_none():
    lat = np.array([gf.SF[0], gf.SF[0]])
    lon = np.array([gf.SF[1], gf.SF[1]])
    result = routes.resample_route(
        lat, lon, _ZONE_NUMBER, _ZONE_LETTER, routes.RouteClusterConfig()
    )
    assert result is None


def test_resample_route_clamps_to_points_min():
    lat = np.array([gf.SF[0], gf.FAR[0]])
    lon = np.array([gf.SF[1], gf.FAR[1]])
    config = routes.RouteClusterConfig()  # ~5 km route, points_per_km=1.0 -> raw ~5
    result = routes.resample_route(lat, lon, _ZONE_NUMBER, _ZONE_LETTER, config)

    assert result is not None
    xy, length_m = result
    assert xy.shape == (config.points_min, 2)
    assert length_m == pytest.approx(4_990, rel=0.05)


def test_resample_route_filters_nan():
    lat = np.array([np.nan, gf.SF[0], gf.FAR[0]])
    lon = np.array([np.nan, gf.SF[1], gf.FAR[1]])
    result = routes.resample_route(
        lat, lon, _ZONE_NUMBER, _ZONE_LETTER, routes.RouteClusterConfig()
    )
    # If the NaN leaked into the UTM projection instead of being filtered, this would be
    # NaN (or the call would raise) rather than the same ~5.2 km SF->FAR distance the
    # points_min test above gets from the same two real points with no leading NaN.
    assert result is not None
    _, length_m = result
    assert length_m == pytest.approx(5_231, rel=0.05)


# --------------------------------------------------------------------------------------
# frechet_pair / route_pairs / symmetric_matrix / cluster_partition
# --------------------------------------------------------------------------------------


def test_frechet_pair_identical_routes_within_tolerance():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    ratio = routes.frechet_pair(xy, xy.copy(), 200.0, 200.0, routes.RouteClusterConfig())
    assert ratio < 1.0


def test_frechet_pair_far_apart_routes_above_tolerance():
    xy_a = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    xy_b = xy_a + np.array([0.0, 1000.0])
    ratio = routes.frechet_pair(xy_a, xy_b, 200.0, 200.0, routes.RouteClusterConfig())
    assert ratio > 1.0


def test_frechet_pair_length_ratio_prefilter_skips_shape_comparison():
    xy_a = np.array([[0.0, 0.0], [200.0, 0.0]])
    xy_b = np.array([[0.0, 0.0], [500.0, 0.0]])
    ratio = routes.frechet_pair(xy_a, xy_b, 200.0, 500.0, routes.RouteClusterConfig())
    assert ratio == float("inf")


def test_route_pairs_generates_all_combinations():
    route_list = [(np.zeros((2, 2)), 100.0), (np.zeros((2, 2)), 200.0), (np.zeros((2, 2)), 300.0)]
    pairs = list(routes.route_pairs(route_list))
    assert len(pairs) == 3
    assert [(p[2], p[3]) for p in pairs] == [(100.0, 200.0), (100.0, 300.0), (200.0, 300.0)]


def test_symmetric_matrix_mirrors_upper_triangle():
    m = routes.symmetric_matrix(3, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(m, [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])


def test_cluster_partition_groups_within_eps_and_flags_noise():
    dm = routes.symmetric_matrix(3, [0.5, 5.0, 5.0])  # 0&1 close, 2 far from both
    labels = routes.cluster_partition(dm, routes.RouteClusterConfig(min_samples=2))
    assert labels[0] == labels[1]
    assert labels[2] == -1


# --------------------------------------------------------------------------------------
# extract_route_features
# --------------------------------------------------------------------------------------


def test_extract_route_features_merges_preloaded_and_freshly_parsed(tmp_path):
    # Only "b.gpx" exists on disk; "a.gpx" must come entirely from preloaded_coords, or
    # this raises FileNotFoundError rather than silently returning a wrong result.
    pts = gf.line_route(gf.SF, gf.FAR, n=30)
    (tmp_path / "b.gpx").write_bytes(gf.gpx_bytes(pts))

    preloaded_a = pd.DataFrame(
        {"latitude": [37.7749, 37.7752], "longitude": [-122.4194, -122.4190]}
    )
    activities = pd.DataFrame({"filename": ["a.gpx", "b.gpx"]})
    preloaded: _PreloadedCoords = {("a.gpx", None): preloaded_a}

    valid_idx, valid_routes = routes.extract_route_features(
        activities, None, tmp_path, None, preloaded, routes.RouteClusterConfig()
    )

    assert valid_idx == [0, 1]
    pd.testing.assert_frame_equal(valid_routes[0], preloaded_a)
    assert valid_routes[1]["latitude"].iloc[0] == pytest.approx(gf.SF[0])


# --------------------------------------------------------------------------------------
# compute_clusters
# --------------------------------------------------------------------------------------


def _close_route_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two near-duplicate 3-point routes, well within the default similarity floor."""
    route_a = pd.DataFrame(
        {"latitude": [37.7749, 37.7752, 37.7755], "longitude": [-122.4194, -122.4190, -122.4186]}
    )
    route_b = pd.DataFrame(
        {
            "latitude": [37.77495, 37.77525, 37.77555],
            "longitude": [-122.41935, -122.41895, -122.41855],
        }
    )
    return route_a, route_b


def test_compute_clusters_gps_activities_cluster_together():
    route_a, route_b = _close_route_pair()
    activities = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00"],
    )
    config = routes.RouteClusterConfig(geocoding=None, min_samples=2)
    preloaded: _PreloadedCoords = {("a.gpx", None): route_a, ("b.gpx", None): route_b}

    results = routes.compute_clusters(activities, None, "unused", None, preloaded, config)

    assert [r.cluster_id for r in results] == [0, 0]
    assert results[0].start_lat is not None


def test_compute_clusters_duplicate_timestamps_both_get_results():
    """Activities sharing an index timestamp must not collapse onto one another."""
    route_a, route_b = _close_route_pair()
    activities = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
        ],
        ["2026-01-05 08:00:00", "2026-01-05 08:00:00"],
    )
    config = routes.RouteClusterConfig(geocoding=None, min_samples=2)
    preloaded: _PreloadedCoords = {("a.gpx", None): route_a, ("b.gpx", None): route_b}

    results = routes.compute_clusters(activities, None, "unused", None, preloaded, config)

    assert results[0].start_lat is not None
    assert results[1].start_lat is not None
    assert results[0].cluster_id == results[1].cluster_id == 0


def test_compute_clusters_all_nan_names_does_not_raise():
    """A GPS cluster whose members all have a NaN name must not raise IndexError."""
    route_a, route_b = _close_route_pair()
    activities = _activities(
        [
            {"filename": "a.gpx", "description": np.nan},
            {"filename": "b.gpx", "description": np.nan},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00"],
    )
    config = routes.RouteClusterConfig(geocoding=None, min_samples=2)
    preloaded: _PreloadedCoords = {("a.gpx", None): route_a, ("b.gpx", None): route_b}

    results = routes.compute_clusters(activities, None, "unused", None, preloaded, config)

    assert [r.cluster_id for r in results] == [0, 0]
    assert results[0].cluster_name is None


def test_compute_clusters_no_gps_files_cluster_by_name():
    activities = _activities(
        [
            {"filename": "a.fit", "description": "Trainer Ride"},
            {"filename": "b.fit", "description": "Trainer Ride"},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00"],
    )
    config = routes.RouteClusterConfig(geocoding=None, min_samples=2)
    preloaded: _PreloadedCoords = {("a.fit", None): None, ("b.fit", None): None}

    results = routes.compute_clusters(activities, None, "unused", None, preloaded, config)

    assert [r.cluster_id for r in results] == [0, 0]
    assert [r.cluster_name for r in results] == ["Trainer Ride", "Trainer Ride"]
    assert results[0].start_lat is None


def test_compute_clusters_fileless_activities_stay_unclustered():
    activities = _activities(
        [{"filename": np.nan, "description": "Manual entry"}], ["2026-01-05 08:00:00"]
    )
    config = routes.RouteClusterConfig(geocoding=None)

    results = routes.compute_clusters(activities, None, "unused", None, {}, config)

    assert results[0].cluster_id is None
    assert results[0].start_lat is None


def test_compute_clusters_geocoding_populates_addresses():
    route_a, route_b = _close_route_pair()
    activities = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00"],
    )
    provider = FakeProvider(
        addresses={geocoding.round_pos((37.774925, -122.419425)): "123 Main St"}
    )
    config = routes.RouteClusterConfig(
        geocoding=geocoding.GeocodingConfig(provider=provider), min_samples=2
    )
    preloaded: _PreloadedCoords = {("a.gpx", None): route_a, ("b.gpx", None): route_b}

    results = routes.compute_clusters(activities, None, "unused", None, preloaded, config)

    assert results[0].start_address == "123 Main St"
    assert provider.reverse_calls  # confirms no real network provider was reached


def test_compute_clusters_shared_endpoints_different_shapes_stay_separate():
    """Two pairs sharing start/end land in one partition; only the near-duplicate pairs
    within each shape should join a cluster, not the whole partition."""
    s_lat, s_lon = 37.7749, -122.4194
    e_lat, e_lon = s_lat - 0.01, s_lon  # ~1.1 km south
    mid_lat, mid_lon = (s_lat + e_lat) / 2, s_lon

    def straight(jitter_lon: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "latitude": [s_lat, mid_lat, e_lat],
                "longitude": [s_lon + jitter_lon, mid_lon + jitter_lon, e_lon + jitter_lon],
            }
        )

    def bulge(jitter_lon: float = 0.0) -> pd.DataFrame:
        # Same start/end as `straight`; the midpoint bulges ~200 m east, well past the
        # default similarity tolerance but within length_ratio_max of the straight route.
        return pd.DataFrame(
            {
                "latitude": [s_lat, mid_lat, e_lat],
                "longitude": [
                    s_lon + jitter_lon,
                    mid_lon + 0.002274 + jitter_lon,
                    e_lon + jitter_lon,
                ],
            }
        )

    activities = _activities(
        [
            {"filename": "a1.gpx", "description": "A1"},
            {"filename": "a2.gpx", "description": "A2"},
            {"filename": "b1.gpx", "description": "B1"},
            {"filename": "b2.gpx", "description": "B2"},
        ],
        [
            "2026-01-05 08:00:00",
            "2026-01-06 08:00:00",
            "2026-01-07 08:00:00",
            "2026-01-08 08:00:00",
        ],
    )
    config = routes.RouteClusterConfig(geocoding=None, min_samples=2)
    preloaded: _PreloadedCoords = {
        ("a1.gpx", None): straight(),
        ("a2.gpx", None): straight(jitter_lon=0.0003),
        ("b1.gpx", None): bulge(),
        ("b2.gpx", None): bulge(jitter_lon=0.0003),
    }

    results = routes.compute_clusters(activities, None, "unused", None, preloaded, config)

    assert results[0].cluster_id == results[1].cluster_id
    assert results[2].cluster_id == results[3].cluster_id
    assert results[0].cluster_id != results[2].cluster_id


def test_compute_clusters_larger_name_cluster_outranks_smaller_gps_cluster():
    route_a, route_b = _close_route_pair()
    activities = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
            {"filename": "c.fit", "description": "Trainer Ride"},
            {"filename": "d.fit", "description": "Trainer Ride"},
            {"filename": "e.fit", "description": "Trainer Ride"},
        ],
        [
            "2026-01-05 08:00:00",
            "2026-01-06 08:00:00",
            "2026-01-07 08:00:00",
            "2026-01-08 08:00:00",
            "2026-01-09 08:00:00",
        ],
    )
    config = routes.RouteClusterConfig(geocoding=None, min_samples=2)
    preloaded: _PreloadedCoords = {
        ("a.gpx", None): route_a,
        ("b.gpx", None): route_b,
        ("c.fit", None): None,
        ("d.fit", None): None,
        ("e.fit", None): None,
    }

    results = routes.compute_clusters(activities, None, "unused", None, preloaded, config)

    # The 3-member name cluster outranks the 2-member GPS cluster for id 0.
    assert [r.cluster_id for r in results] == [1, 1, 0, 0, 0]


# --------------------------------------------------------------------------------------
# cluster_routes
# --------------------------------------------------------------------------------------


def test_cluster_routes_clusters_similar_gps_routes(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=30)
    (tmp_path / "a.gpx").write_bytes(gf.gpx_bytes(pts))
    (tmp_path / "b.gpx").write_bytes(gf.gpx_bytes(gf.offset_route(pts)))
    activities = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00"],
    )
    config = routes.RouteClusterConfig(geocoding=None)

    result = routes.cluster_routes(activities, None, tmp_path, None, config)

    assert result["cluster_id"].tolist() == [0, 0]


def test_cluster_routes_activities_without_file_are_unclustered(tmp_path):
    activities = _activities(
        [{"filename": np.nan, "description": "Manual"}], ["2026-01-05 08:00:00"]
    )
    config = routes.RouteClusterConfig(geocoding=None)

    result = routes.cluster_routes(activities, None, tmp_path, None, config)

    assert pd.isna(result["cluster_id"].iloc[0])


# --------------------------------------------------------------------------------------
# cluster_routes_cached
# --------------------------------------------------------------------------------------


def _insert_activity_rows(cache_dir, keys: list[tuple[str, int]]) -> None:
    with cache_db.open_db(cache_dir) as db:
        with db.conn:
            db["activities"].insert_all(
                [{"filename": fn, "segment": seg} for fn, seg in keys],
                pk=("filename", "segment"),
            )


def test_cluster_routes_cached_hit_skips_reparsing(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    pts = gf.line_route(gf.SF, gf.FAR, n=30)
    (export_dir / "a.gpx").write_bytes(gf.gpx_bytes(pts))
    (export_dir / "b.gpx").write_bytes(gf.gpx_bytes(gf.offset_route(pts)))
    cache_dir = tmp_path / "cache"
    _insert_activity_rows(cache_dir, [("a.gpx", -1), ("b.gpx", -1)])

    activities = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00"],
    )
    config = routes.RouteClusterConfig(geocoding=None)

    first = routes.cluster_routes_cached(
        activities, None, export_dir, cache_dir, "activities", None, config
    )

    # Corrupt the source files; a fingerprint-matched cache hit must not touch them.
    (export_dir / "a.gpx").write_bytes(b"not gpx")
    (export_dir / "b.gpx").write_bytes(b"not gpx")

    second = routes.cluster_routes_cached(
        activities, None, export_dir, cache_dir, "activities", None, config
    )

    pd.testing.assert_frame_equal(first, second)
    assert second["cluster_id"].tolist() == [0, 0]


def test_cluster_routes_cached_no_cache_dir_recomputes_every_call(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=30)
    (tmp_path / "a.gpx").write_bytes(gf.gpx_bytes(pts))
    (tmp_path / "b.gpx").write_bytes(gf.gpx_bytes(gf.offset_route(pts)))
    activities = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00"],
    )
    config = routes.RouteClusterConfig(geocoding=None)

    result = routes.cluster_routes_cached(
        activities, None, tmp_path, None, "activities", None, config
    )

    assert result["cluster_id"].tolist() == [0, 0]
    assert not (tmp_path / cache_db.DB_FILE).exists()


def test_cluster_routes_cached_stale_fingerprint_on_activity_set_change(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    pts = gf.line_route(gf.SF, gf.FAR, n=30)
    (export_dir / "a.gpx").write_bytes(gf.gpx_bytes(pts))
    (export_dir / "b.gpx").write_bytes(gf.gpx_bytes(gf.offset_route(pts)))
    (export_dir / "c.gpx").write_bytes(gf.gpx_bytes(gf.offset_route(pts, dlat=-0.0003)))
    cache_dir = tmp_path / "cache"
    _insert_activity_rows(cache_dir, [("a.gpx", -1), ("b.gpx", -1), ("c.gpx", -1)])
    config = routes.RouteClusterConfig(geocoding=None)

    two_files = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00"],
    )
    first = routes.cluster_routes_cached(
        two_files, None, export_dir, cache_dir, "activities", None, config
    )
    assert first["cluster_id"].tolist() == [0, 0]

    three_files = _activities(
        [
            {"filename": "a.gpx", "description": "Ride A"},
            {"filename": "b.gpx", "description": "Ride B"},
            {"filename": "c.gpx", "description": "Ride C"},
        ],
        ["2026-01-05 08:00:00", "2026-01-06 08:00:00", "2026-01-07 08:00:00"],
    )
    second = routes.cluster_routes_cached(
        three_files, None, export_dir, cache_dir, "activities", None, config
    )
    assert second["cluster_id"].tolist() == [0, 0, 0]


def test_cluster_routes_cached_raises_when_rows_missing_from_db(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    pts = gf.line_route(gf.SF, gf.FAR, n=30)
    (export_dir / "a.gpx").write_bytes(gf.gpx_bytes(pts))
    cache_dir = tmp_path / "cache"
    # Deliberately don't insert the activities row cluster_routes_cached expects.
    activities = _activities(
        [{"filename": "a.gpx", "description": "Ride A"}], ["2026-01-05 08:00:00"]
    )
    config = routes.RouteClusterConfig(geocoding=None)

    with pytest.raises(RuntimeError, match="Cluster cache UPDATE"):
        routes.cluster_routes_cached(
            activities, None, export_dir, cache_dir, "activities", None, config
        )
