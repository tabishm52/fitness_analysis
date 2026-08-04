"""Tests for piecewise linear regression and its disk cache."""

import numpy as np
import pandas as pd
import pytest

from fitness_analysis.piecewise_fit import (
    PIECEWISE_CACHE_DIR,
    invalidate_piecewise_fit_cache,
    piecewise_fit,
    piecewise_fit_auto,
    piecewise_fit_cached,
    piecewise_fit_with_breaks,
)

# 30 daily points, exact two-segment piecewise line: slope -1/day to day 15, then +2/day.
_IDX = pd.date_range("2026-01-01", periods=30, freq="D")
_DAYS = np.arange(30)
_TWO_SEGMENT_VALUES = np.where(_DAYS <= 15, 100.0 - 1.0 * _DAYS, 85.0 + 2.0 * (_DAYS - 15))
TWO_SEGMENT_SERIES = pd.Series(_TWO_SEGMENT_VALUES, index=_IDX)

# DE searches continuous time over a daily-sampled signal, so the fitted breakpoint
# lands near but not exactly on the true day-15 break.
_BREAK_TOL = pd.Timedelta(hours=1)


def _assert_two_segment_fit(out: pd.DataFrame):
    assert len(out) == 3
    assert out.index[0] == _IDX[0]
    assert out.index[-1] == _IDX[-1]
    assert abs(out.index[1] - _IDX[15]) < _BREAK_TOL

    np.testing.assert_allclose(out["value"].to_numpy(), [100.0, 85.0, 113.0], atol=1e-2)
    rates = out["rate"].to_numpy()
    np.testing.assert_allclose(rates[:2], [-1.0, 2.0], atol=1e-2)
    assert np.isnan(rates[2])


# --------------------------------------------------------------------------------------
# piecewise_fit
# --------------------------------------------------------------------------------------


def test_piecewise_fit_recovers_known_breakpoint_and_slopes():
    out = piecewise_fit(TWO_SEGMENT_SERIES, n_segments=2, units="D")

    _assert_two_segment_fit(out)


def test_piecewise_fit_drops_nan_before_fitting():
    series = TWO_SEGMENT_SERIES.copy()
    series.iloc[5] = np.nan

    out = piecewise_fit(series, n_segments=2, units="D")

    _assert_two_segment_fit(out)


def test_piecewise_fit_raises_value_error_when_infeasible():
    with pytest.raises(ValueError, match="infeasible"):
        piecewise_fit(
            TWO_SEGMENT_SERIES,
            n_segments=2,
            units="D",
            min_segment_duration=pd.Timedelta(days=20),
        )


# --------------------------------------------------------------------------------------
# piecewise_fit_with_breaks
# --------------------------------------------------------------------------------------


def test_piecewise_fit_with_breaks_uses_given_breakpoints():
    breaks = [_IDX[0], _IDX[15], _IDX[-1]]

    out = piecewise_fit_with_breaks(TWO_SEGMENT_SERIES, breaks=breaks, units="D")

    assert list(out.index) == breaks
    np.testing.assert_allclose(out["value"].to_numpy(), [100.0, 85.0, 113.0])
    rates = out["rate"].to_numpy()
    np.testing.assert_allclose(rates[:2], [-1.0, 2.0])
    assert np.isnan(rates[2])


# --------------------------------------------------------------------------------------
# piecewise_fit_auto
# --------------------------------------------------------------------------------------


def test_piecewise_fit_auto_selects_two_segments_by_bic():
    out = piecewise_fit_auto(TWO_SEGMENT_SERIES, units="D", max_segments=2)

    _assert_two_segment_fit(out)


def test_piecewise_fit_auto_rejects_max_segments_below_one():
    with pytest.raises(ValueError, match="max_segments"):
        piecewise_fit_auto(TWO_SEGMENT_SERIES, units="D", max_segments=0)


def test_piecewise_fit_auto_raises_when_min_segment_duration_exceeds_span():
    with pytest.raises(ValueError, match="min_segment_duration"):
        piecewise_fit_auto(
            TWO_SEGMENT_SERIES,
            units="D",
            max_segments=2,
            min_segment_duration=pd.Timedelta(days=40),
        )


# --------------------------------------------------------------------------------------
# piecewise_fit_cached
# --------------------------------------------------------------------------------------


def test_piecewise_fit_cached_accepts_str_cache_dir(tmp_path):
    out = piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=str(tmp_path))

    _assert_two_segment_fit(out)
    assert len(list((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet"))) == 1


def test_invalidate_piecewise_fit_cache_accepts_str_cache_dir(tmp_path):
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)

    invalidate_piecewise_fit_cache(str(tmp_path))

    assert not (tmp_path / PIECEWISE_CACHE_DIR).exists()


def test_piecewise_fit_cached_writes_and_reads_back(tmp_path):
    out = piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)

    _assert_two_segment_fit(out)
    cached_files = list((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet"))
    assert len(cached_files) == 1


def test_piecewise_fit_cached_none_dir_skips_caching():
    out = piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=None)

    _assert_two_segment_fit(out)


def test_piecewise_fit_cached_different_params_get_different_cache_entries(tmp_path):
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "W", max_segments=2, cache_dir=tmp_path)

    assert len(list((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet"))) == 2


# --------------------------------------------------------------------------------------
# invalidate_piecewise_fit_cache
# --------------------------------------------------------------------------------------


def test_invalidate_piecewise_fit_cache_none_dir_is_noop():
    invalidate_piecewise_fit_cache(None, max_cached=0)


def test_invalidate_piecewise_fit_cache_missing_dir_is_noop(tmp_path):
    invalidate_piecewise_fit_cache(tmp_path / "does-not-exist", max_cached=0)


def test_invalidate_piecewise_fit_cache_deletes_all_by_default(tmp_path):
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)

    invalidate_piecewise_fit_cache(tmp_path)

    assert not (tmp_path / PIECEWISE_CACHE_DIR).exists()


def test_invalidate_piecewise_fit_cache_ignores_stray_non_parquet_entries(tmp_path):
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)
    stray_dir = tmp_path / PIECEWISE_CACHE_DIR / "stray_subdir"
    stray_dir.mkdir()

    invalidate_piecewise_fit_cache(tmp_path)

    assert stray_dir.exists()
    assert list((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet")) == []


def test_invalidate_piecewise_fit_cache_prunes_to_max_cached(tmp_path):
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "W", max_segments=2, cache_dir=tmp_path)

    invalidate_piecewise_fit_cache(tmp_path, max_cached=1)

    assert len(list((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet"))) == 1
