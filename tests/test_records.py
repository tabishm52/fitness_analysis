"""Tests for activity record parsing and its parquet cache."""

import gzip
from pathlib import Path

import fit_fixtures
import gpx_fixtures as gf
import pandas as pd
import pytest

from fitness_analysis import records

# --------------------------------------------------------------------------------------
# parquet_path
# --------------------------------------------------------------------------------------


def test_parquet_path_whole_file():
    p = records.parquet_path("ride.gpx", None, "/cache")
    assert p == Path("/cache/activity_records/ride.gpx.parquet")


def test_parquet_path_strips_gz():
    p = records.parquet_path("ride.fit.gz", None, "/cache")
    assert p == Path("/cache/activity_records/ride.fit.parquet")


def test_parquet_path_with_segment():
    p = records.parquet_path("ride.gpx", 2, "/cache")
    assert p == Path("/cache/activity_records/ride-2.gpx.parquet")


def test_parquet_path_segment_with_gz():
    p = records.parquet_path("ride.fit.gz", 1, "/cache")
    assert p == Path("/cache/activity_records/ride-1.fit.parquet")


def test_parquet_path_drops_parent_directories():
    p = records.parquet_path("activities/ride.gpx", None, "/cache")
    assert p == Path("/cache/activity_records/ride.gpx.parquet")


# --------------------------------------------------------------------------------------
# cache_record / parse_record_cached
# --------------------------------------------------------------------------------------


def _write_gpx(path: Path, hr=(100, 101, 102)) -> Path:
    pts = gf.line_route(gf.SF, gf.FAR, n=len(hr))
    path.write_bytes(gf.gpx_bytes(pts, hr=list(hr)))
    return path


def test_parse_record_cached_no_cache_dir_reparses_every_call(tmp_path):
    _write_gpx(tmp_path / "ride.gpx")

    first = records.parse_record_cached("ride.gpx", None, tmp_path)
    second = records.parse_record_cached("ride.gpx", None, tmp_path)

    pd.testing.assert_frame_equal(first, second)
    assert first["heart_rate"].tolist() == [100, 101, 102]


def test_parse_record_cached_writes_and_reads_cache(tmp_path):
    _write_gpx(tmp_path / "ride.gpx")
    cache_dir = tmp_path / "cache"

    parsed = records.parse_record_cached("ride.gpx", None, tmp_path, cache_dir)
    cache_path = records.parquet_path("ride.gpx", None, cache_dir)
    assert cache_path.exists()

    cached = records.parse_record_cached("ride.gpx", None, tmp_path, cache_dir)
    pd.testing.assert_frame_equal(parsed, cached)


def test_parse_record_cached_hit_skips_reparsing(tmp_path):
    _write_gpx(tmp_path / "ride.gpx")
    cache_dir = tmp_path / "cache"
    records.parse_record_cached("ride.gpx", None, tmp_path, cache_dir)

    # Corrupt the source file; a cache hit must not touch it.
    (tmp_path / "ride.gpx").write_bytes(b"not gpx")

    cached = records.parse_record_cached("ride.gpx", None, tmp_path, cache_dir)
    assert cached["heart_rate"].tolist() == [100, 101, 102]


def test_parse_record_cached_segment_requires_cache_dir(tmp_path):
    _write_gpx(tmp_path / "ride.gpx")
    with pytest.raises(ValueError, match="cache_dir is required"):
        records.parse_record_cached("ride.gpx", 1, tmp_path, None)


def test_parse_record_cached_segment_miss_raises(tmp_path):
    _write_gpx(tmp_path / "ride.gpx")
    with pytest.raises(FileNotFoundError, match="No cached records for segment"):
        records.parse_record_cached("ride.gpx", 1, tmp_path, tmp_path / "cache")


def test_parse_record_cached_segment_hit(tmp_path):
    _write_gpx(tmp_path / "ride.gpx")
    cache_dir = tmp_path / "cache"
    df = records.parse_record_cached("ride.gpx", None, tmp_path, cache_dir)
    records.cache_record(df, "ride.gpx", 1, cache_dir)

    segment = records.parse_record_cached("ride.gpx", 1, tmp_path, cache_dir)
    pd.testing.assert_frame_equal(df, segment)


def test_parse_record_cached_fit_file(tmp_path):
    (tmp_path / "ride.fit").write_bytes(fit_fixtures.simple_ride(n=4))
    parsed = records.parse_record_cached("ride.fit", None, tmp_path)
    assert parsed["heart_rate"].tolist() == [100, 101, 102, 103]


def test_parse_record_cached_fit_gz_file(tmp_path):
    (tmp_path / "ride.fit.gz").write_bytes(gzip.compress(fit_fixtures.simple_ride(n=4)))
    cache_dir = tmp_path / "cache"

    parsed = records.parse_record_cached("ride.fit.gz", None, tmp_path, cache_dir)
    assert parsed["heart_rate"].tolist() == [100, 101, 102, 103]
    assert records.parquet_path("ride.fit.gz", None, cache_dir).exists()


# --------------------------------------------------------------------------------------
# coords_from_records
# --------------------------------------------------------------------------------------


def test_coords_from_records_no_location_columns():
    df = pd.DataFrame({"heart_rate": [100, 101]})
    assert records.coords_from_records(df) is None


def test_coords_from_records_all_nan_latitude():
    df = pd.DataFrame({"latitude": [None, None], "longitude": [None, None]})
    assert records.coords_from_records(df) is None


def test_coords_from_records_trims_to_valid_range():
    df = pd.DataFrame(
        {
            "latitude": [None, 37.0, 37.1, None],
            "longitude": [None, -122.0, -122.1, None],
        }
    )
    coords = records.coords_from_records(df)
    assert coords is not None
    assert coords["latitude"].tolist() == [37.0, 37.1]
    assert coords["longitude"].tolist() == [-122.0, -122.1]


# --------------------------------------------------------------------------------------
# warm_records_cache
# --------------------------------------------------------------------------------------


def test_warm_records_cache_empty_is_noop(tmp_path):
    records.warm_records_cache([], tmp_path / "cache")  # should not raise


def test_warm_records_cache_writes_cold_files(tmp_path):
    _write_gpx(tmp_path / "a.gpx")
    _write_gpx(tmp_path / "b.gpx")
    cache_dir = tmp_path / "cache"

    args = [("a.gpx", None, tmp_path, cache_dir), ("b.gpx", None, tmp_path, cache_dir)]
    records.warm_records_cache(args, cache_dir)

    assert records.parquet_path("a.gpx", None, cache_dir).exists()
    assert records.parquet_path("b.gpx", None, cache_dir).exists()


def test_warm_records_cache_skips_segment_entries(tmp_path):
    cache_dir = tmp_path / "cache"
    args = [("a.gpx", 1, tmp_path, cache_dir)]
    records.warm_records_cache(args, cache_dir)
    assert not records.parquet_path("a.gpx", 1, cache_dir).exists()


def test_warm_records_cache_skips_already_cached(tmp_path):
    _write_gpx(tmp_path / "a.gpx", hr=(100, 101))
    cache_dir = tmp_path / "cache"
    records.cache_record(
        records.parse_record_cached("a.gpx", None, tmp_path), "a.gpx", None, cache_dir
    )
    mtime_before = records.parquet_path("a.gpx", None, cache_dir).stat().st_mtime

    # Change the source; if warm_records_cache treated this file as cold, the cache
    # would be overwritten with new content.
    _write_gpx(tmp_path / "a.gpx", hr=(999, 998))
    records.warm_records_cache([("a.gpx", None, tmp_path, cache_dir)], cache_dir)

    assert records.parquet_path("a.gpx", None, cache_dir).stat().st_mtime == mtime_before


# --------------------------------------------------------------------------------------
# invalidate_records_cache
# --------------------------------------------------------------------------------------


def test_invalidate_records_cache_missing_dir_is_noop(tmp_path):
    records.invalidate_records_cache(None, None, tmp_path / "cache")  # should not raise


def test_invalidate_records_cache_none_clears_everything(tmp_path):
    _write_gpx(tmp_path / "a.gpx")
    cache_dir = tmp_path / "cache"
    records.parse_record_cached("a.gpx", None, tmp_path, cache_dir)

    records.invalidate_records_cache(None, None, cache_dir)
    assert not (cache_dir / records.RECORDS_CACHE_DIR).exists()


def test_invalidate_records_cache_removes_only_listed_files(tmp_path):
    _write_gpx(tmp_path / "a.gpx")
    _write_gpx(tmp_path / "b.gpx")
    cache_dir = tmp_path / "cache"
    records.parse_record_cached("a.gpx", None, tmp_path, cache_dir)
    records.parse_record_cached("b.gpx", None, tmp_path, cache_dir)

    records.invalidate_records_cache(["a.gpx"], None, cache_dir)

    assert not records.parquet_path("a.gpx", None, cache_dir).exists()
    assert records.parquet_path("b.gpx", None, cache_dir).exists()


# --------------------------------------------------------------------------------------
# load_activity_records / load_activity_coords
# --------------------------------------------------------------------------------------


def test_load_activity_records_no_cache_dir(tmp_path):
    _write_gpx(tmp_path / "a.gpx", hr=(100, 101))
    _write_gpx(tmp_path / "b.gpx", hr=(200, 201))

    out = records.load_activity_records(["a.gpx", "b.gpx"], None, tmp_path)

    assert [df["heart_rate"].tolist() for df in out] == [[100, 101], [200, 201]]


def test_load_activity_records_uses_and_populates_cache(tmp_path):
    _write_gpx(tmp_path / "a.gpx", hr=(100, 101))
    cache_dir = tmp_path / "cache"

    first = records.load_activity_records(["a.gpx"], None, tmp_path, cache_dir)
    assert records.parquet_path("a.gpx", None, cache_dir).exists()

    second = records.load_activity_records(["a.gpx"], None, tmp_path, cache_dir)
    pd.testing.assert_frame_equal(first[0], second[0])


def test_load_activity_coords_preserves_order_and_nones(tmp_path):
    _write_gpx(tmp_path / "a.gpx")
    (tmp_path / "b.fit").write_bytes(fit_fixtures.simple_ride())  # no GPS data

    out = records.load_activity_coords(["a.gpx", "b.fit"], None, tmp_path)

    assert out[0] is not None
    assert out[1] is None
