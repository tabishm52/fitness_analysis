"""Tests for the SQLite cache database shared by the activities and commutes caches."""

import numpy as np
import sqlite_utils

from fitness_analysis.cache_db import (
    cache_key,
    db_path,
    delete_fingerprint,
    ensure_tables,
    open_db,
    segment_from_db,
    segment_to_db,
    to_sql,
)


def test_to_sql_converts_nan_to_none():
    assert to_sql(float("nan")) is None


def test_to_sql_converts_numpy_scalar_to_python():
    out = to_sql(np.float64(1.5))
    assert out == 1.5
    assert type(out) is float


def test_to_sql_passes_through_plain_values():
    assert to_sql("filename.gpx") == "filename.gpx"
    assert to_sql(3) == 3


def test_segment_to_db_uses_sentinel_for_none():
    assert segment_to_db(None) == -1


def test_segment_to_db_uses_sentinel_for_nan():
    assert segment_to_db(float("nan")) == -1


def test_segment_to_db_passes_through_int():
    assert segment_to_db(2) == 2


def test_segment_from_db_uses_sentinel_for_none():
    assert segment_from_db(-1) is None


def test_segment_from_db_passes_through_int():
    assert segment_from_db(2) == 2


def test_cache_key_returns_none_for_nan_filename():
    assert cache_key(float("nan"), 0) is None


def test_cache_key_builds_tuple_with_sentinel_segment():
    assert cache_key("ride.gpx", None) == ("ride.gpx", -1)
    assert cache_key("ride.gpx", 1) == ("ride.gpx", 1)


def test_db_path_joins_cache_dir(tmp_path):
    assert db_path(tmp_path) == tmp_path / "fitness_cache.db"


def test_open_db_creates_cache_dir_if_missing(tmp_path):
    cache_dir = tmp_path / "nested" / "cache"
    with open_db(cache_dir) as db:
        assert db_path(cache_dir).exists()
        assert isinstance(db, sqlite_utils.Database)


def test_open_db_creates_expected_tables(tmp_path):
    with open_db(tmp_path) as db:
        assert set(db.table_names()) == {
            "activities",
            "commutes",
            "cluster_fingerprints",
            "geocode_cache",
        }


def test_ensure_tables_is_idempotent(tmp_path):
    with open_db(tmp_path) as db:
        ensure_tables(db)
        ensure_tables(db)
        assert set(db.table_names()) == {
            "activities",
            "commutes",
            "cluster_fingerprints",
            "geocode_cache",
        }


def test_delete_fingerprint_single_table(tmp_path):
    with open_db(tmp_path) as db:
        with db.conn:
            db["cluster_fingerprints"].insert_all(
                [
                    {"table_name": "activities", "fingerprint": "abc"},
                    {"table_name": "commutes", "fingerprint": "def"},
                ],
                pk="table_name",
                replace=True,
            )
            delete_fingerprint(db, "activities")
        rows = list(db["cluster_fingerprints"].rows)
        assert [r["table_name"] for r in rows] == ["commutes"]


def test_delete_fingerprint_all_tables(tmp_path):
    with open_db(tmp_path) as db:
        with db.conn:
            db["cluster_fingerprints"].insert_all(
                [
                    {"table_name": "activities", "fingerprint": "abc"},
                    {"table_name": "commutes", "fingerprint": "def"},
                ],
                pk="table_name",
                replace=True,
            )
            delete_fingerprint(db)
        assert list(db["cluster_fingerprints"].rows) == []
