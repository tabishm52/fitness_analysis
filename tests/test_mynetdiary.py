"""Tests for MyNetDiary export loading, caching, and EER calculations."""

import os
import time
from datetime import date
from pathlib import Path

import mnd_fixtures as mf
import pandas as pd
import pytest

from fitness_analysis import mynetdiary

FILES = Path(__file__).parent / "files"
XLS_EXPORT_DIR = FILES

# --------------------------------------------------------------------------------------
# merge_excel_files
# --------------------------------------------------------------------------------------


def test_merge_excel_files_reads_xls_export():
    data = mynetdiary.merge_excel_files(XLS_EXPORT_DIR)

    assert data["Measurements"]["Value"].tolist() == [150.0, 149.5, 149.0]
    assert data["Food"]["Calories, cals"].tolist() == [500, 600]
    assert data["Exercise"]["Calories"].tolist() == [200]


def test_merge_excel_files_concatenates_multiple_exports_alphabetically(tmp_path):
    measurements_1 = pd.DataFrame(
        {"Measurement": "Body Weight", "Date": [date(2026, 1, 1)], "Value": [150.0], "Unit": "lbs"}
    )
    measurements_2 = pd.DataFrame(
        {"Measurement": "Body Weight", "Date": [date(2026, 1, 2)], "Value": [149.0], "Unit": "lbs"}
    )
    mf.write_mnd_export(tmp_path, measurements=measurements_1, filename="export-2.xlsx")
    mf.write_mnd_export(tmp_path, measurements=measurements_2, filename="export-1.xlsx")

    data = mynetdiary.merge_excel_files(tmp_path)

    # "export-1.xlsx" sorts before "export-2.xlsx", so its rows come first regardless
    # of the order write_mnd_export was called in.
    assert data["Measurements"]["Value"].tolist() == [149.0, 150.0]


def test_merge_excel_files_skips_empty_sheets(tmp_path):
    mf.write_mnd_export(tmp_path, exercise=pd.DataFrame(columns=["Date & Time", "Calories"]))
    data = mynetdiary.merge_excel_files(tmp_path)
    assert "Exercise" not in data


# --------------------------------------------------------------------------------------
# merge_excel_files_cached
# --------------------------------------------------------------------------------------


def test_merge_excel_files_cached_writes_fingerprint(tmp_path):
    export_dir = tmp_path / "export"
    cache_dir = tmp_path / "cache"
    mf.write_mnd_export(export_dir, n_days=3)

    mynetdiary.merge_excel_files_cached(export_dir, cache_dir)

    cache_path = cache_dir / mynetdiary.MND_CACHE_DIR
    assert (cache_path / mynetdiary.MND_FINGERPRINT_FNAME).exists()
    assert (cache_path / "Measurements.parquet").exists()


def test_merge_excel_files_cached_hit_serves_stale_data_when_mtime_matches(tmp_path):
    export_dir = tmp_path / "export"
    cache_dir = tmp_path / "cache"
    xlsx_path = mf.write_mnd_export(export_dir, n_days=3, start_weight=150.0)
    original_mtime = xlsx_path.stat().st_mtime
    expected = pytest.approx([150.0, 150 - 1 / 7, 150 - 2 / 7])

    first = mynetdiary.merge_excel_files_cached(export_dir, cache_dir)
    assert first["Measurements"]["Value"].tolist() == expected

    # Overwrite with different content but restore the original mtime: the fingerprint
    # is (filename, mtime) only, so this must still read as a cache hit.
    mf.write_mnd_export(export_dir, n_days=3, start_weight=999.0, filename=xlsx_path.name)
    os.utime(xlsx_path, (original_mtime, original_mtime))

    second = mynetdiary.merge_excel_files_cached(export_dir, cache_dir)
    assert second["Measurements"]["Value"].tolist() == expected


def test_merge_excel_files_cached_mtime_change_invalidates(tmp_path):
    export_dir = tmp_path / "export"
    cache_dir = tmp_path / "cache"
    xlsx_path = mf.write_mnd_export(export_dir, n_days=3, start_weight=150.0)

    mynetdiary.merge_excel_files_cached(export_dir, cache_dir)

    time.sleep(0.01)
    mf.write_mnd_export(export_dir, n_days=3, start_weight=999.0, filename=xlsx_path.name)

    refreshed = mynetdiary.merge_excel_files_cached(export_dir, cache_dir)
    assert refreshed["Measurements"]["Value"].iloc[0] == pytest.approx(999.0)


# --------------------------------------------------------------------------------------
# invalidate_mnd_cache
# --------------------------------------------------------------------------------------


def test_invalidate_mnd_cache_missing_dir_is_noop(tmp_path):
    mynetdiary.invalidate_mnd_cache(tmp_path / "cache")  # should not raise


def test_invalidate_mnd_cache_deletes_cache_dir(tmp_path):
    export_dir = tmp_path / "export"
    cache_dir = tmp_path / "cache"
    mf.write_mnd_export(export_dir, n_days=3)
    mynetdiary.merge_excel_files_cached(export_dir, cache_dir)

    mynetdiary.invalidate_mnd_cache(cache_dir)

    assert not (cache_dir / mynetdiary.MND_CACHE_DIR).exists()


# --------------------------------------------------------------------------------------
# eer_male / eer_female
# --------------------------------------------------------------------------------------


def test_eer_male_pinned_output():
    # Pinned regression value (MyNetDiary formula effective January 2025)
    dob = "2000-01-05"
    weight = pd.Series([150.0], index=pd.DatetimeIndex(["2010-01-05"]))
    height = 70.0

    result = mynetdiary.eer_male(weight, height, dob)
    assert result.iloc[0] == pytest.approx(2759.803037)


def test_eer_female_pinned_output():
    # Pinned regression value (MyNetDiary formula effective January 2025)
    dob = "2000-01-05"
    weight = pd.Series([130.0], index=pd.DatetimeIndex(["2010-01-05"]))
    height = 65.0

    result = mynetdiary.eer_female(weight, height, dob)
    assert result.iloc[0] == pytest.approx(2149.666069)


# --------------------------------------------------------------------------------------
# load_mnd_data
# --------------------------------------------------------------------------------------


def test_load_mnd_data_rate_matches_known_slope(tmp_path):
    export_dir = tmp_path / "export"
    mf.write_mnd_export(export_dir, n_days=30, start_weight=180.0)

    weight, calories = mynetdiary.load_mnd_data(export_dir, eer_func=lambda w: w * 0 + 2000)

    # The fixture's default measurements are an exact -1 lb/week line.
    assert weight["rate"].dropna().mean() == pytest.approx(-1.0, abs=1e-6)
    assert (calories["exercise"] >= 0).all()


def test_load_mnd_data_rejects_unexpected_weight_units(tmp_path):
    export_dir = tmp_path / "export"
    bad_measurements = pd.DataFrame(
        {
            "Measurement": ["Body Weight"],
            "Date": [date(2026, 1, 1)],
            "Value": [70.0],
            "Unit": ["kg"],
        }
    )
    mf.write_mnd_export(export_dir, measurements=bad_measurements, n_days=1)

    with pytest.raises(ValueError, match="Unexpected body weight units"):
        mynetdiary.load_mnd_data(export_dir, eer_func=lambda w: w)


def test_load_mnd_data_tuning_changes_smoothing(tmp_path):
    export_dir = tmp_path / "export"
    mf.write_mnd_export(export_dir, n_days=30)

    default_weight, _ = mynetdiary.load_mnd_data(export_dir, eer_func=lambda w: w * 0 + 2000)

    # A much longer half-life than the "3D" default should visibly change smoothing,
    # confirming `tuning` actually threads through rather than being silently ignored.
    tuning = mynetdiary.MndTuning(weight_halflife="10D", calorie_halflife="10D", rate_window_days=6)
    tuned_weight, _ = mynetdiary.load_mnd_data(
        export_dir, eer_func=lambda w: w * 0 + 2000, tuning=tuning
    )

    assert not tuned_weight["smoothed"].equals(default_weight["smoothed"])
