"""Tests for Strava commute processing, splitting, and caching."""

from datetime import datetime, timedelta
from typing import Any, cast

import gpx_fixtures as gf
import numpy as np
import pandas as pd
import pytest
import strava_fixtures as sfx

from fitness_analysis import cache_db, commute, records, routes, utils

HOME_TZ = "America/New_York"


def _leg(n: int, start_offset_s: int, distance_start: float = 0.0, distance_end: float = 1.0):
    """A leg of ``n`` GPS records, 1 s apart, starting ``start_offset_s`` after T0."""
    idx = pd.DatetimeIndex(
        [gf.T0 + timedelta(seconds=start_offset_s + i) for i in range(n)], tz="UTC"
    )
    lat = np.linspace(gf.SF[0], gf.FAR[0], n)
    lon = np.linspace(gf.SF[1], gf.FAR[1], n)
    dist = np.linspace(distance_start, distance_end, n)
    return pd.DataFrame({"latitude": lat, "longitude": lon, "distance": dist}, index=idx)


def _write_commute_gpx(path, name, n=10):
    pts = gf.line_route(gf.SF, gf.FAR, n=n)
    (path / name).write_bytes(gf.gpx_bytes(pts))


# --------------------------------------------------------------------------------------
# CommuteMetrics
# --------------------------------------------------------------------------------------


def test_commute_metrics_db_roundtrip():
    m = commute.CommuteMetrics(
        date=pd.Timestamp("2026-01-05 08:00:00"),
        description="Ride to work",
        direction="Morning",
        distance=5.0,
        elapsed_time_s=1200.0,
        moving_time_s=1100.0,
        filename="ride.gpx",
        segment=1,
    )
    restored = commute.CommuteMetrics.from_db_dict(m.to_db_dict())
    assert restored == m


def test_commute_metrics_coerces_nan_to_none():
    m = commute.CommuteMetrics(
        date=pd.Timestamp("2026-01-05"),
        description="x",
        direction="Morning",
        distance=float("nan"),
        elapsed_time_s=100.0,
        filename="a.gpx",
    )
    assert m.distance is None


# --------------------------------------------------------------------------------------
# Cache management
# --------------------------------------------------------------------------------------


def test_load_commutes_cache_groups_by_filename(tmp_path):
    m1 = commute.CommuteMetrics(
        date=pd.Timestamp("2026-01-05 08:00"),
        description="A",
        direction="Morning",
        elapsed_time_s=100.0,
        filename="a.gpx",
        segment=1,
    )
    m2 = commute.CommuteMetrics(
        date=pd.Timestamp("2026-01-05 10:00"),
        description="A",
        direction="Afternoon",
        elapsed_time_s=100.0,
        filename="a.gpx",
        segment=2,
    )
    with cache_db.open_db(tmp_path) as db:
        db["commutes"].upsert_all(
            [m1.to_db_dict(), m2.to_db_dict()], pk=cast(Any, ("filename", "segment"))
        )

    cache = commute.load_commutes_cache(tmp_path)
    assert set(cache) == {"a.gpx"}
    assert len(cache["a.gpx"]) == 2


def test_invalidate_commutes_cache_missing_db_is_noop(tmp_path):
    commute.invalidate_commutes_cache(None, tmp_path)  # should not raise


def test_invalidate_commutes_cache_none_clears_everything(tmp_path):
    m = commute.CommuteMetrics(
        date=pd.Timestamp("2026-01-05"),
        description="A",
        direction="Morning",
        elapsed_time_s=100.0,
        filename="a.gpx",
    )
    with cache_db.open_db(tmp_path) as db:
        db["commutes"].upsert(m.to_db_dict(), pk=cast(Any, ("filename", "segment")))

    commute.invalidate_commutes_cache(None, tmp_path)

    assert commute.load_commutes_cache(tmp_path) == {}


def test_invalidate_commutes_cache_removes_only_listed_files(tmp_path):
    for fn in ("a.gpx", "b.gpx"):
        m = commute.CommuteMetrics(
            date=pd.Timestamp("2026-01-05"),
            description="A",
            direction="Morning",
            elapsed_time_s=100.0,
            filename=fn,
        )
        with cache_db.open_db(tmp_path) as db:
            db["commutes"].upsert(m.to_db_dict(), pk=cast(Any, ("filename", "segment")))

    commute.invalidate_commutes_cache(["a.gpx"], tmp_path)

    assert set(commute.load_commutes_cache(tmp_path)) == {"b.gpx"}


def test_invalidate_commutes_cache_removes_only_listed_files_segment_parquets(tmp_path):
    records.cache_record(pd.DataFrame({"latitude": [1.0]}), "a.gpx", 1, tmp_path)
    records.cache_record(pd.DataFrame({"latitude": [2.0]}), "b.gpx", 1, tmp_path)
    for fn in ("a.gpx", "b.gpx"):
        m = commute.CommuteMetrics(
            date=pd.Timestamp("2026-01-05"),
            description="A",
            direction="Morning",
            elapsed_time_s=100.0,
            filename=fn,
            segment=1,
        )
        with cache_db.open_db(tmp_path) as db:
            db["commutes"].upsert(m.to_db_dict(), pk=cast(Any, ("filename", "segment")))

    commute.invalidate_commutes_cache(["a.gpx"], tmp_path)

    assert not records.parquet_path("a.gpx", 1, tmp_path).exists()
    assert records.parquet_path("b.gpx", 1, tmp_path).exists()


def test_invalidate_commutes_cache_empty_list_is_noop(tmp_path):
    """Modern SQLite treats an empty `IN ()` as always-false, so an empty ``files`` list
    already no-ops correctly with no source change needed."""
    m = commute.CommuteMetrics(
        date=pd.Timestamp("2026-01-05"),
        description="A",
        direction="Morning",
        elapsed_time_s=100.0,
        filename="a.gpx",
    )
    with cache_db.open_db(tmp_path) as db:
        db["commutes"].upsert(m.to_db_dict(), pk=cast(Any, ("filename", "segment")))

    commute.invalidate_commutes_cache([], tmp_path)

    assert set(commute.load_commutes_cache(tmp_path)) == {"a.gpx"}


def test_invalidate_commutes_cache_none_deletes_segment_parquets(tmp_path):
    records.cache_record(pd.DataFrame({"latitude": [1.0]}), "a.gpx", 1, tmp_path)
    m = commute.CommuteMetrics(
        date=pd.Timestamp("2026-01-05"),
        description="A",
        direction="Morning",
        elapsed_time_s=100.0,
        filename="a.gpx",
        segment=1,
    )
    with cache_db.open_db(tmp_path) as db:
        db["commutes"].upsert(m.to_db_dict(), pk=cast(Any, ("filename", "segment")))

    commute.invalidate_commutes_cache(None, tmp_path)

    assert not records.parquet_path("a.gpx", 1, tmp_path).exists()


# --------------------------------------------------------------------------------------
# segment_metrics
# --------------------------------------------------------------------------------------


def test_segment_metrics_computes_distance_time_and_direction():
    group = _leg(10, 0, distance_start=0.0, distance_end=1.0)
    activity = pd.Series({"Activity Name": "Bike to Work", "Filename": "ride.gpx"})

    metrics = commute.segment_metrics(activity, group, None, commute.CommuteConfig())

    assert metrics.description == "Bike to Work"
    # T0 = 08:00 UTC -> America/Los_Angeles (PST, UTC-8 in Jan) -> 00:00 local -> Morning
    assert metrics.direction == "Morning"
    assert metrics.distance == pytest.approx(1.0 * utils.KM_TO_MI)
    assert metrics.elapsed_time_s == pytest.approx(9.0)
    assert metrics.filename == "ride.gpx"
    assert metrics.segment is None


def test_segment_metrics_excludes_stopped_time_from_moving_time():
    move1 = _leg(5, 0, 0.0, 1.0)
    stop = _leg(20, 5, 1.0, 1.0)
    move2 = _leg(5, 25, 1.0, 2.0)
    group = pd.concat([move1, stop, move2])
    activity = pd.Series({"Activity Name": "Ride", "Filename": "ride.gpx"})
    config = commute.CommuteConfig(stopped_speed=1.0, min_stop_duration=pd.Timedelta(10, "s"))

    metrics = commute.segment_metrics(activity, group, None, config)

    assert metrics.moving_time_s is not None
    assert metrics.moving_time_s < metrics.elapsed_time_s


def test_segment_metrics_no_distance_column_returns_none_distance():
    idx = pd.DatetimeIndex([gf.T0, gf.T0 + timedelta(seconds=10)])
    group = pd.DataFrame({"latitude": [gf.SF[0]] * 2, "longitude": [gf.SF[1]] * 2}, index=idx)
    activity = pd.Series({"Activity Name": "Ride", "Filename": "ride.gpx"})

    metrics = commute.segment_metrics(activity, group, None, commute.CommuteConfig())

    assert metrics.distance is None
    assert metrics.moving_time_s is None


# --------------------------------------------------------------------------------------
# parse_commute_file
# --------------------------------------------------------------------------------------


def test_parse_commute_file_splits_on_delta_gap():
    activity_records = pd.concat([_leg(10, 0), _leg(10, 3 * 3600)])
    activity = pd.Series({"Activity Name": "Round trip", "Filename": "ride.gpx"})

    results, split_coords = commute.parse_commute_file(
        activity, activity_records, commute.CommuteConfig()
    )

    assert [m.segment for m in results] == [1, 2]
    assert set(split_coords) == {("ride.gpx", 1), ("ride.gpx", 2)}


def test_parse_commute_file_single_segment_has_none_segment():
    activity = pd.Series({"Activity Name": "One way", "Filename": "ride.gpx"})

    results, split_coords = commute.parse_commute_file(
        activity, _leg(10, 0), commute.CommuteConfig()
    )

    assert len(results) == 1
    assert results[0].segment is None
    assert set(split_coords) == {("ride.gpx", None)}


def test_parse_commute_file_writes_multi_segment_parquets(tmp_path):
    activity_records = pd.concat([_leg(10, 0), _leg(10, 3 * 3600)])
    activity = pd.Series({"Activity Name": "Round trip", "Filename": "ride.gpx"})

    commute.parse_commute_file(
        activity, activity_records, commute.CommuteConfig(), cache_dir=tmp_path
    )

    assert records.parquet_path("ride.gpx", 1, tmp_path).exists()
    assert records.parquet_path("ride.gpx", 2, tmp_path).exists()


def test_parse_commute_file_inserts_into_db_when_provided(tmp_path):
    activity = pd.Series({"Activity Name": "One way", "Filename": "ride.gpx"})

    with cache_db.open_db(tmp_path) as db:
        commute.parse_commute_file(activity, _leg(10, 0), commute.CommuteConfig(), db=db)

    assert set(commute.load_commutes_cache(tmp_path)) == {"ride.gpx"}


def test_parse_commute_file_drops_long_inactive_period_before_splitting():
    """GPS left on all day: a long stationary run, densely sampled (no single gap above
    ``delta``), must still be dropped so the surrounding legs split into two segments."""
    config = commute.CommuteConfig(delta=pd.Timedelta(10, "m"), inactive_speed=2.5)

    def _points(start_s, n, step_s, dist_start, dist_end):
        times = [gf.T0 + timedelta(seconds=start_s + step_s * i) for i in range(n)]
        dist = np.linspace(dist_start, dist_end, n)
        return times, dist

    leg1_times, leg1_dist = _points(0, 11, 30, 0.0, 1.0)  # 5 min moving
    stat_times, stat_dist = _points(330, 41, 30, 1.0, 1.0)  # 20 min stationary, densely sampled
    leg2_times, leg2_dist = _points(1560, 11, 30, 1.0, 2.0)  # 5 min moving

    all_times = leg1_times + stat_times + leg2_times
    all_dist = np.concatenate([leg1_dist, stat_dist, leg2_dist])
    lat = np.linspace(gf.SF[0], gf.FAR[0], len(all_times))
    lon = np.linspace(gf.SF[1], gf.FAR[1], len(all_times))
    activity_records = pd.DataFrame(
        {"latitude": lat, "longitude": lon, "distance": all_dist},
        index=pd.DatetimeIndex(all_times, tz="UTC"),
    )
    activity = pd.Series({"Activity Name": "Commute", "Filename": "ride.gpx"})

    results, _ = commute.parse_commute_file(activity, activity_records, config)

    assert [m.segment for m in results] == [1, 2]


# --------------------------------------------------------------------------------------
# process_commute_csv
# --------------------------------------------------------------------------------------


def test_process_commute_csv_computes_metrics_from_csv_fields():
    activity = pd.Series(
        {
            "Activity Name": "Bike commute",
            "Distance": 8.0,
            "Elapsed Time": 1500,
            "Moving Time": 1400,
            "Filename": float("nan"),
        }
    )
    utc_date = pd.Timestamp("2026-01-05 16:00:00")

    metrics = commute.process_commute_csv(
        activity, utc_date, "America/New_York", commute.CommuteConfig()
    )

    assert metrics.distance == pytest.approx(8.0 * utils.KM_TO_MI)
    assert metrics.elapsed_time_s == 1500.0
    assert metrics.moving_time_s == 1400.0
    # 16:00 UTC -> 11:00 America/New_York (EST, UTC-5 in Jan) -> Morning
    assert metrics.date == pd.Timestamp("2026-01-05 11:00:00")
    assert metrics.direction == "Morning"


# --------------------------------------------------------------------------------------
# load_commute_splits
# --------------------------------------------------------------------------------------


def test_load_commute_splits_parses_misses(tmp_path):
    _write_commute_gpx(tmp_path, "ride.gpx")
    file_commutes = pd.DataFrame({"Filename": ["ride.gpx"], "Activity Name": ["Commute"]})

    splits, preloaded = commute.load_commute_splits(
        file_commutes, tmp_path, None, commute.CommuteConfig(), None
    )

    assert set(splits) == {"ride.gpx"}
    assert set(preloaded) == {("ride.gpx", None)}


def test_load_commute_splits_cache_hit_skips_reparsing(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_commute_gpx(export_dir, "ride.gpx")
    file_commutes = pd.DataFrame({"Filename": ["ride.gpx"], "Activity Name": ["Commute"]})
    cache_dir = tmp_path / "cache"

    commute.load_commute_splits(file_commutes, export_dir, cache_dir, commute.CommuteConfig(), None)
    cache = commute.load_commutes_cache(cache_dir)

    (export_dir / "ride.gpx").write_bytes(b"not gpx")
    splits, preloaded = commute.load_commute_splits(
        file_commutes, export_dir, cache_dir, commute.CommuteConfig(), cache
    )

    assert set(splits) == {"ride.gpx"}
    assert preloaded == {}


# --------------------------------------------------------------------------------------
# build_commute_columns
# --------------------------------------------------------------------------------------


def test_build_commute_columns_fileless_same_instant_both_kept():
    """Fileless commutes sharing a UTC timestamp must not collapse."""
    same_date = datetime(2026, 1, 5, 16, 0)
    commutes = pd.DataFrame(
        [
            {
                "Activity Name": "Commute A",
                "Distance": 5.0,
                "Elapsed Time": 1000,
                "Moving Time": 900,
                "Filename": float("nan"),
            },
            {
                "Activity Name": "Commute B",
                "Distance": 6.0,
                "Elapsed Time": 1100,
                "Moving Time": 1000,
                "Filename": float("nan"),
            },
        ],
        index=pd.DatetimeIndex([same_date, same_date]),
    )

    calcs, _ = commute.build_commute_columns(
        commutes, "unused", HOME_TZ, None, commute.CommuteConfig(clustering=None)
    )

    assert len(calcs) == 2
    assert set(calcs["description"]) == {"Commute A", "Commute B"}


def test_build_commute_columns_clustering_disabled_returns_none():
    same_date = datetime(2026, 1, 5, 16, 0)
    commutes = pd.DataFrame(
        [
            {
                "Activity Name": "Commute A",
                "Distance": 5.0,
                "Elapsed Time": 1000,
                "Moving Time": 900,
                "Filename": float("nan"),
            }
        ],
        index=pd.DatetimeIndex([same_date]),
    )

    _, clusters = commute.build_commute_columns(
        commutes, "unused", HOME_TZ, None, commute.CommuteConfig(clustering=None)
    )

    assert clusters is None


# --------------------------------------------------------------------------------------
# compute_spans
# --------------------------------------------------------------------------------------


def _span_inputs(addresses=None):
    dates = pd.DatetimeIndex(
        [
            datetime(2026, 1, 5, 8, 0),
            datetime(2026, 1, 5, 17, 0),
            datetime(2026, 1, 6, 8, 0),
            datetime(2026, 1, 6, 17, 0),
        ]
    )
    commutes_df = pd.DataFrame(
        {"direction": ["Morning", "Afternoon", "Morning", "Afternoon"]}, index=dates
    )
    addr = addresses or [None] * 4
    clusters = pd.DataFrame(
        {
            "start_lat": [37.0, 38.0, 37.0, 38.0],
            "start_lon": [-122.0, -121.0, -122.0, -121.0],
            "end_lat": [38.0, 37.0, 38.0, 37.0],
            "end_lon": [-121.0, -122.0, -121.0, -122.0],
            "start_address": addr,
            "end_address": list(reversed(addr)),
        },
        index=dates,
    )
    return commutes_df, clusters


def test_compute_spans_detects_one_span_and_counts_commutes():
    commutes_df, clusters = _span_inputs()
    config = commute.CommuteConfig(
        clustering=routes.RouteClusterConfig(geocoding=None), span_min_size=2
    )

    spans = commute.compute_spans(commutes_df, clusters, config)

    assert len(spans) == 1
    assert spans["n_commutes"].iloc[0] == 4
    assert "home_address" not in spans.columns


def test_compute_spans_includes_addresses_when_geocoding_enabled():
    commutes_df, clusters = _span_inputs(
        ["123 Home St", "456 Work Ave", "123 Home St", "456 Work Ave"]
    )
    config = commute.CommuteConfig(clustering=routes.RouteClusterConfig(), span_min_size=2)

    spans = commute.compute_spans(commutes_df, clusters, config)

    assert spans["home_address"].iloc[0] == "123 Home St"
    assert spans["work_address"].iloc[0] == "456 Work Ave"


def test_compute_spans_all_nan_addresses_returns_none_not_indexerror():
    commutes_df, clusters = _span_inputs()
    config = commute.CommuteConfig(clustering=routes.RouteClusterConfig(), span_min_size=2)

    spans = commute.compute_spans(commutes_df, clusters, config)

    assert spans["home_address"].iloc[0] is None
    assert spans["work_address"].iloc[0] is None


def test_compute_spans_detects_changepoint_when_location_shifts():
    """A genuine move partway through: PELT must split into two spans, not one."""
    dates = pd.DatetimeIndex(
        [
            datetime(2026, 1, 5, 8, 0),
            datetime(2026, 1, 5, 17, 0),
            datetime(2026, 1, 6, 8, 0),
            datetime(2026, 1, 6, 17, 0),
            datetime(2026, 2, 5, 8, 0),
            datetime(2026, 2, 5, 17, 0),
            datetime(2026, 2, 6, 8, 0),
            datetime(2026, 2, 6, 17, 0),
        ]
    )
    commutes_df = pd.DataFrame({"direction": ["Morning", "Afternoon"] * 4}, index=dates)
    # First 4 commutes: home (37, -122) <-> work (38, -121). Last 4: a different home/work
    # pair entirely (10, 20) <-> (11, 21) -- a real relocation, not GPS noise.
    clusters = pd.DataFrame(
        {
            "start_lat": [37.0, 38.0] * 2 + [10.0, 11.0] * 2,
            "start_lon": [-122.0, -121.0] * 2 + [20.0, 21.0] * 2,
            "end_lat": [38.0, 37.0] * 2 + [11.0, 10.0] * 2,
            "end_lon": [-121.0, -122.0] * 2 + [21.0, 20.0] * 2,
            "start_address": [None] * 8,
            "end_address": [None] * 8,
        },
        index=dates,
    )
    # Default span_penalty (100.0) is tuned for real GPS jitter and doesn't reliably
    # split on just 8 points even with a huge, clean jump; lower it for this test.
    config = commute.CommuteConfig(
        clustering=routes.RouteClusterConfig(geocoding=None), span_min_size=2, span_penalty=10.0
    )

    spans = commute.compute_spans(commutes_df, clusters, config)

    assert len(spans) == 2
    assert spans["n_commutes"].tolist() == [4, 4]


# --------------------------------------------------------------------------------------
# load_commute_activities
# --------------------------------------------------------------------------------------


def test_load_commute_activities_filters_to_commutes(tmp_path):
    rows = [
        sfx.activity_row(datetime(2026, 1, 5, 8, 0), commute=True, name="Commute"),
        sfx.activity_row(datetime(2026, 1, 5, 9, 0), commute=False, name="Not a commute"),
    ]
    sfx.write_export(tmp_path, rows, {})

    commutes_df, _ = commute.load_commute_activities(
        tmp_path, HOME_TZ, None, commute.CommuteConfig(clustering=None)
    )

    assert len(commutes_df) == 1
    assert commutes_df["description"].iloc[0] == "Commute"


def test_load_commute_activities_gps_commute_with_clustering(tmp_path):
    pts = gf.line_route(gf.SF, gf.FAR, n=30)
    rows = [
        sfx.activity_row(datetime(2026, 1, 5, 8, 0), commute=True, filename="activities/ride.gpx")
    ]
    sfx.write_export(tmp_path, rows, {"activities/ride.gpx": gf.gpx_bytes(pts)})
    config = commute.CommuteConfig(
        clustering=routes.RouteClusterConfig(geocoding=None), span_min_size=1
    )

    commutes_df, spans_df = commute.load_commute_activities(tmp_path, HOME_TZ, None, config)

    assert "cluster_id" in commutes_df.columns
    assert spans_df is not None
