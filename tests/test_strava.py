"""Tests for Strava activity loading, per-file metrics, and caching."""

import io
from datetime import datetime

import fit_fixtures
import gpx_fixtures as gf
import numpy as np
import pandas as pd
import pytest
import strava_fixtures as sfx
from activity_parser import ActivityParser

from fitness_analysis import cache_db, records, routes, strava, utils

HOME_TZ = "America/New_York"  # UTC-5 in January, distinct from the GPX fixtures' SF tz


def _stationary_points(n: int) -> list[tuple[int, float, float]]:
    """``n`` points 1 s apart at a fixed location.

    compute_power_curve resamples to a 1-second grid and treats gaps as 0 W, so a
    "constant power" fixture needs samples at every second — line_route's 10 s spacing
    would otherwise coast to 0 W between points.
    """
    return [(i, *gf.SF) for i in range(n)]


# --------------------------------------------------------------------------------------
# ActivitiesConfig
# --------------------------------------------------------------------------------------


def test_activities_config_does_not_mutate_shared_clustering_config():
    shared = routes.RouteClusterConfig()
    assert shared.raw_csv is False

    config = strava.ActivitiesConfig(clustering=shared)

    assert config.clustering is not None
    assert config.clustering.raw_csv is True
    assert shared.raw_csv is False


# --------------------------------------------------------------------------------------
# ActivityMetrics
# --------------------------------------------------------------------------------------


def test_activity_metrics_db_roundtrip_with_power():
    metrics = strava.ActivityMetrics(
        filename="ride.gpx",
        timezone="America/Los_Angeles",
        has_location=True,
        max_heart_rate=180.0,
        power_windows=np.array([1, 5, 60]),
        power_curve=np.array([300.0, 280.0, 250.0]),
        estimated_ftp=237.5,
    )
    restored = strava.ActivityMetrics.from_db_dict(metrics.to_db_dict())

    assert restored.filename == "ride.gpx"
    assert restored.timezone == "America/Los_Angeles"
    assert restored.has_location is True
    assert restored.max_heart_rate == 180.0
    assert restored.power_windows is not None
    assert restored.power_curve is not None
    assert restored.power_windows.tolist() == [1, 5, 60]
    assert restored.power_curve.tolist() == [300.0, 280.0, 250.0]
    assert restored.estimated_ftp == 237.5


def test_activity_metrics_db_roundtrip_without_power():
    metrics = strava.ActivityMetrics(filename="manual.gpx")
    restored = strava.ActivityMetrics.from_db_dict(metrics.to_db_dict())

    assert restored.power_windows is None
    assert restored.power_curve is None
    assert restored.has_location is False


def test_activity_metrics_coerces_nan_to_none():
    metrics = strava.ActivityMetrics(filename="x.gpx", max_heart_rate=float("nan"))
    assert metrics.max_heart_rate is None


# --------------------------------------------------------------------------------------
# Cache management
# --------------------------------------------------------------------------------------


def test_load_activities_cache_roundtrips_parsed_metrics(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=5)
    records_df_path = tmp_path / "ride.gpx"
    records_df_path.write_bytes(gf.gpx_bytes(pts, hr=[100, 101, 102, 103, 104], power=[200] * 5))

    activity_records = records.parse_record_cached("ride.gpx", None, tmp_path)
    config = strava.ActivitiesConfig(clustering=None)
    with cache_db.open_db(tmp_path) as db:
        strava.parse_activity_file("ride.gpx", activity_records, config, db)

    cache = strava.load_activities_cache(tmp_path)
    assert set(cache) == {"ride.gpx"}
    assert cache["ride.gpx"].max_heart_rate == 104
    assert cache["ride.gpx"].timezone == "America/Los_Angeles"


def test_invalidate_activities_cache_missing_db_is_noop(tmp_path):
    strava.invalidate_activities_cache(None, tmp_path)  # should not raise


def test_invalidate_activities_cache_none_clears_everything(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        strava.parse_activity_file("a.gpx", pd.DataFrame(), strava.ActivitiesConfig(), db)
        strava.parse_activity_file("b.gpx", pd.DataFrame(), strava.ActivitiesConfig(), db)

    strava.invalidate_activities_cache(None, tmp_path)

    assert strava.load_activities_cache(tmp_path) == {}


def test_invalidate_activities_cache_removes_only_listed_files(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        strava.parse_activity_file("a.gpx", pd.DataFrame(), strava.ActivitiesConfig(), db)
        strava.parse_activity_file("b.gpx", pd.DataFrame(), strava.ActivitiesConfig(), db)

    strava.invalidate_activities_cache(["a.gpx"], tmp_path)

    assert set(strava.load_activities_cache(tmp_path)) == {"b.gpx"}


def test_invalidate_activities_cache_empty_list_is_noop(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        strava.parse_activity_file("a.gpx", pd.DataFrame(), strava.ActivitiesConfig(), db)

    strava.invalidate_activities_cache([], tmp_path)

    assert set(strava.load_activities_cache(tmp_path)) == {"a.gpx"}


# --------------------------------------------------------------------------------------
# parse_activity_file
# --------------------------------------------------------------------------------------


def test_parse_activity_file_computes_timezone_hr_and_ftp():
    pts = _stationary_points(5)
    hr = [100, 101, 102, 103, 104]
    power = [200] * 5
    activity_records, _, _ = _parse_bytes(gf.gpx_bytes(pts, hr=hr, power=power), "gpx")
    config = strava.ActivitiesConfig()

    metrics = strava.parse_activity_file("ride.gpx", activity_records, config)

    assert metrics.timezone == "America/Los_Angeles"
    assert metrics.has_location is True
    assert metrics.max_heart_rate == 104
    assert metrics.estimated_ftp == pytest.approx(200 * config.ftp_factor)


def test_parse_activity_file_no_location_or_power():
    activity_records, _, _ = _parse_bytes(fit_fixtures.simple_ride(n=3), "fit")
    metrics = strava.parse_activity_file("ride.fit", activity_records, strava.ActivitiesConfig())

    assert metrics.timezone is None
    assert metrics.has_location is False
    assert metrics.estimated_ftp is None


def _parse_bytes(data: bytes, ext: str):
    return ActivityParser().parse(io.BytesIO(data), ext=ext)


# --------------------------------------------------------------------------------------
# load_file_metrics
# --------------------------------------------------------------------------------------


def test_load_file_metrics_handles_fileless_activities(tmp_path):
    files = pd.Series([float("nan"), float("nan")])
    calcs, preloaded = strava.load_file_metrics(files, tmp_path, None, strava.ActivitiesConfig())

    assert calcs["timezone"].isna().all()
    assert preloaded == {}


def test_load_file_metrics_handles_repeated_filename(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=3)
    (tmp_path / "ride.gpx").write_bytes(gf.gpx_bytes(pts, hr=[100, 101, 102]))

    files = pd.Series(["ride.gpx", "ride.gpx"])
    calcs, preloaded = strava.load_file_metrics(files, tmp_path, None, strava.ActivitiesConfig())

    assert calcs["max_heart_rate"].tolist() == [102, 102]
    assert set(preloaded) == {("ride.gpx", None)}


def test_load_file_metrics_preloaded_coords_only_covers_misses(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=3)
    (tmp_path / "ride.gpx").write_bytes(gf.gpx_bytes(pts, hr=[100, 101, 102]))
    cache_dir = tmp_path / "cache"

    strava.load_file_metrics(
        pd.Series(["ride.gpx"]), tmp_path, cache_dir, strava.ActivitiesConfig()
    )
    cache = strava.load_activities_cache(cache_dir)

    _, preloaded = strava.load_file_metrics(
        pd.Series(["ride.gpx"]), tmp_path, cache_dir, strava.ActivitiesConfig(), cache
    )
    assert preloaded == {}


# --------------------------------------------------------------------------------------
# build_activity_columns
# --------------------------------------------------------------------------------------


def _write_basic_export(export_dir, gpx_hr=(100, 101, 102)):
    pts = gf.line_route(gf.SF, gf.FAR, n=len(gpx_hr))
    gpx_bytes = gf.gpx_bytes(pts, hr=list(gpx_hr))
    fit_bytes = fit_fixtures.simple_ride(n=3)

    rows = [
        sfx.activity_row(
            datetime(2026, 1, 5, 8, 0), activity_type="Ride", filename="activities/gps.gpx"
        ),
        sfx.activity_row(
            datetime(2026, 1, 5, 9, 0), activity_type="Ride", filename="activities/no_gps.fit"
        ),
        sfx.activity_row(datetime(2026, 1, 5, 10, 0), activity_type="Ride", filename=None),
        sfx.activity_row(
            datetime(2026, 1, 5, 11, 0),
            activity_type="Virtual Ride",
            filename="activities/gps.gpx",
        ),
    ]
    sfx.write_export(
        export_dir,
        rows,
        {"activities/gps.gpx": gpx_bytes, "activities/no_gps.fit": fit_bytes},
    )


def test_build_activity_columns_trainer_classification(tmp_path):
    _write_basic_export(tmp_path)
    csv = strava.load_strava_activities_raw(tmp_path)

    calcs, _ = strava.build_activity_columns(
        csv, tmp_path, HOME_TZ, None, strava.ActivitiesConfig(clustering=None)
    )

    # gps ride, file without GPS, fileless ride, virtual ride (with a GPS file)
    assert calcs["trainer"].tolist() == [False, True, False, True]


def test_build_activity_columns_local_date_uses_gps_or_home_tz(tmp_path):
    _write_basic_export(tmp_path)
    csv = strava.load_strava_activities_raw(tmp_path)

    calcs, _ = strava.build_activity_columns(
        csv, tmp_path, HOME_TZ, None, strava.ActivitiesConfig(clustering=None)
    )

    # 08:00 UTC in America/Los_Angeles (PST, UTC-8) on this January date -> 00:00 local
    assert calcs["local_date"].iloc[0] == pd.Timestamp("2026-01-05 00:00:00")
    # No-GPS activities fall back to HOME_TZ (America/New_York, UTC-5) -> 04:00 local
    assert calcs["local_date"].iloc[1] == pd.Timestamp("2026-01-05 04:00:00")
    assert calcs["local_date"].iloc[2] == pd.Timestamp("2026-01-05 05:00:00")


def test_build_activity_columns_clustering_disabled_returns_none(tmp_path):
    _write_basic_export(tmp_path)
    csv = strava.load_strava_activities_raw(tmp_path)

    _, clusters = strava.build_activity_columns(
        csv, tmp_path, HOME_TZ, None, strava.ActivitiesConfig(clustering=None)
    )
    assert clusters is None


def test_build_activity_columns_clustering_enabled_no_network(tmp_path):
    _write_basic_export(tmp_path)
    csv = strava.load_strava_activities_raw(tmp_path)
    config = strava.ActivitiesConfig(clustering=routes.RouteClusterConfig(geocoding=None))

    _, clusters = strava.build_activity_columns(csv, tmp_path, HOME_TZ, None, config)

    assert clusters is not None
    assert len(clusters) == len(csv)
    assert "cluster_id" in clusters.columns
    assert clusters["start_address"].isna().all()


# --------------------------------------------------------------------------------------
# load_strava_activities_raw
# --------------------------------------------------------------------------------------


def test_load_strava_activities_raw_filters_and_indexes(tmp_path):
    rows = [
        sfx.activity_row(datetime(2026, 1, 5, 8, 0), activity_type="Ride"),
        sfx.activity_row(datetime(2026, 1, 6, 8, 0), activity_type="Run"),
        sfx.activity_row(datetime(2026, 1, 7, 8, 0), activity_type="Virtual Ride"),
    ]
    sfx.write_export(tmp_path, rows, {})

    csv = strava.load_strava_activities_raw(tmp_path)

    assert csv.index.name == "Activity Date"
    assert csv["Activity Type"].tolist() == ["Ride", "Virtual Ride"]
    assert csv.index.tolist() == [
        pd.Timestamp("2026-01-05 08:00:00"),
        pd.Timestamp("2026-01-07 08:00:00"),
    ]


# --------------------------------------------------------------------------------------
# load_strava_activities
# --------------------------------------------------------------------------------------


def test_load_strava_activities_output_columns(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=3)
    rows = [
        sfx.activity_row(
            datetime(2026, 1, 5, 8, 0),
            filename="activities/ride.gpx",
            distance_km=10.0,
            elevation_m=100.0,
            elapsed_s=1800,
            moving_s=1700,
            commute=False,
            gear="Road Bike",
            name="Morning Ride",
        )
    ]
    sfx.write_export(tmp_path, rows, {"activities/ride.gpx": gf.gpx_bytes(pts, hr=[100, 101, 102])})

    activities, weekly_sums = strava.load_strava_activities(
        tmp_path, HOME_TZ, None, strava.ActivitiesConfig(clustering=None)
    )

    row = activities.iloc[0]
    assert activities.index[0] == pd.Timestamp("2026-01-05 00:00:00")
    assert row["description"] == "Morning Ride"
    assert row["bicycle"] == "Road Bike"
    assert bool(row["trainer"]) is False
    assert bool(row["commute"]) is False
    assert row["distance"] == pytest.approx(10.0 * utils.KM_TO_MI)
    assert row["elevation"] == pytest.approx(100.0 * utils.M_TO_FT)
    assert row["elapsed_time"] == pd.Timedelta(seconds=1800)
    assert row["moving_time"] == pd.Timedelta(seconds=1700)
    assert row["max_heart_rate"] == 102
    assert row["filename"] == "activities/ride.gpx"

    assert weekly_sums["distance"].iloc[0] == pytest.approx(10.0 * utils.KM_TO_MI)


def test_load_strava_activities_omits_addresses_when_geocoding_disabled(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=3)
    rows = [sfx.activity_row(datetime(2026, 1, 5, 8, 0), filename="activities/ride.gpx")]
    sfx.write_export(tmp_path, rows, {"activities/ride.gpx": gf.gpx_bytes(pts, hr=[100, 101, 102])})
    config = strava.ActivitiesConfig(clustering=routes.RouteClusterConfig(geocoding=None))

    activities, _ = strava.load_strava_activities(tmp_path, HOME_TZ, None, config)

    assert "cluster_id" in activities.columns
    assert "start_address" not in activities.columns


def test_load_strava_activities_cache_hit_skips_reparsing(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=10)
    rows = [
        sfx.activity_row(datetime(2026, 1, 5, 8, 0), filename="activities/a.gpx"),
        sfx.activity_row(datetime(2026, 1, 6, 8, 0), filename="activities/b.gpx"),
    ]
    sfx.write_export(
        tmp_path,
        rows,
        {
            "activities/a.gpx": gf.gpx_bytes(pts),
            "activities/b.gpx": gf.gpx_bytes(gf.offset_route(pts)),
        },
    )
    cache_dir = tmp_path / "cache"
    config = strava.ActivitiesConfig(clustering=routes.RouteClusterConfig(geocoding=None))

    first_activities, first_weekly = strava.load_strava_activities(
        tmp_path, HOME_TZ, cache_dir, config
    )

    # Corrupt the source files; a cache hit (by filename, assumed immutable) must not
    # touch them.
    (tmp_path / "activities" / "a.gpx").write_bytes(b"not gpx")
    (tmp_path / "activities" / "b.gpx").write_bytes(b"not gpx")

    second_activities, second_weekly = strava.load_strava_activities(
        tmp_path, HOME_TZ, cache_dir, config
    )

    pd.testing.assert_frame_equal(first_activities, second_activities)
    pd.testing.assert_frame_equal(first_weekly, second_weekly)
    assert second_activities["cluster_id"].tolist() == [0, 0]


# --------------------------------------------------------------------------------------
# load_power_curves
# --------------------------------------------------------------------------------------


def test_load_power_curves_skips_clustering(tmp_path, monkeypatch):
    """Power curves never use clusters, so clustering (and any geocoding it would
    trigger) must not run even when the caller's config has it enabled."""

    def _fail(*_args, **_kwargs):
        raise AssertionError("cluster_routes_cached should not be called by load_power_curves")

    monkeypatch.setattr(routes, "cluster_routes_cached", _fail)
    pts = _stationary_points(5)
    rows = [sfx.activity_row(datetime(2026, 1, 5, 8, 0), filename="activities/ride.gpx")]
    sfx.write_export(tmp_path, rows, {"activities/ride.gpx": gf.gpx_bytes(pts, power=[200] * 5)})

    curves = strava.load_power_curves(tmp_path, HOME_TZ, None, strava.ActivitiesConfig())

    assert len(curves) == 1


def test_load_power_curves_constant_power(tmp_path):
    pts = _stationary_points(5)
    rows = [sfx.activity_row(datetime(2026, 1, 5, 8, 0), filename="activities/ride.gpx")]
    sfx.write_export(tmp_path, rows, {"activities/ride.gpx": gf.gpx_bytes(pts, power=[200] * 5)})

    curves = strava.load_power_curves(
        tmp_path, HOME_TZ, None, strava.ActivitiesConfig(clustering=None)
    )

    assert len(curves) == 1
    assert curves.iloc[0].dropna().unique().tolist() == pytest.approx([200.0])


def test_load_power_curves_omits_activities_without_power(tmp_path):
    rows = [sfx.activity_row(datetime(2026, 1, 5, 8, 0), filename="activities/ride.fit")]
    sfx.write_export(tmp_path, rows, {"activities/ride.fit": fit_fixtures.simple_ride()})

    curves = strava.load_power_curves(
        tmp_path, HOME_TZ, None, strava.ActivitiesConfig(clustering=None)
    )

    assert curves.empty


def test_load_power_curves_keeps_activities_with_same_local_date(tmp_path):
    # Both trainer rides (forced by "Virtual Ride") fall back to home_tz regardless of
    # GPS, so an identical CSV date collapses to an identical local_date.
    same_date = datetime(2026, 1, 5, 8, 0)
    rows = [
        sfx.activity_row(same_date, activity_type="Virtual Ride", filename="activities/a.gpx"),
        sfx.activity_row(same_date, activity_type="Virtual Ride", filename="activities/b.gpx"),
    ]
    sfx.write_export(
        tmp_path,
        rows,
        {
            "activities/a.gpx": gf.gpx_bytes(_stationary_points(5), power=[200] * 5),
            "activities/b.gpx": gf.gpx_bytes(_stationary_points(5), power=[250] * 5),
        },
    )

    curves = strava.load_power_curves(
        tmp_path, HOME_TZ, None, strava.ActivitiesConfig(clustering=None)
    )

    assert len(curves) == 2
    assert sorted(curves.iloc[i].dropna().unique().item() for i in range(2)) == [200.0, 250.0]
