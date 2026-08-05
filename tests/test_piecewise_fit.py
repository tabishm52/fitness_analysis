"""Tests for piecewise linear regression and its disk cache."""

import os
import time

import numpy as np
import pandas as pd
import pytest

import fitness_analysis.piecewise_fit as piecewise_fit_module
from fitness_analysis.piecewise_fit import (
    PIECEWISE_CACHE_DIR,
    _min_gap_breaks,
    invalidate_piecewise_fit_cache,
    piecewise_fit_auto,
    piecewise_fit_cached,
    piecewise_fit_fixed,
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
# piecewise_fit_fixed
# --------------------------------------------------------------------------------------


def test_piecewise_fit_fixed_recovers_known_breakpoint_and_slopes():
    out = piecewise_fit_fixed(TWO_SEGMENT_SERIES, n_segments=2, units="D")

    _assert_two_segment_fit(out)


def test_piecewise_fit_fixed_drops_nan_before_fitting():
    series = TWO_SEGMENT_SERIES.copy()
    series.iloc[5] = np.nan

    out = piecewise_fit_fixed(series, n_segments=2, units="D")

    _assert_two_segment_fit(out)


def test_piecewise_fit_fixed_raises_value_error_when_infeasible():
    with pytest.raises(ValueError, match="infeasible"):
        piecewise_fit_fixed(
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


# _fit_n_segments is mocked with controlled ssr values so this test verifies the BIC
# comparison loop itself (picks the lower-BIC candidate), isolated from differential
# evolution's own convergence behavior.


def test_piecewise_fit_auto_bic_rejects_marginal_extra_segment(monkeypatch):
    ssr_by_segs = {1: 1_000.0, 2: 1.0, 3: 0.95}

    def fake_fit_n_segments(model, x_lo, x_hi, n_segs, min_segment_s):
        return np.linspace(x_lo, x_hi, n_segs + 1), ssr_by_segs[n_segs]

    monkeypatch.setattr(piecewise_fit_module, "_fit_n_segments", fake_fit_n_segments)

    out = piecewise_fit_auto(TWO_SEGMENT_SERIES, units="D", max_segments=3)

    assert len(out) == 3  # 2 segments (3 breakpoints) wins on BIC over 3 segments


# --------------------------------------------------------------------------------------
# _min_gap_breaks
# --------------------------------------------------------------------------------------


def test_min_gap_breaks_output_is_ordered_and_respects_min_gap():
    breaks = _min_gap_breaks(np.array([5.0, 1.0, 3.0]), x_lo=0.0, min_gap=2.0)

    assert np.all(np.diff(breaks) >= 2.0)
    assert breaks[0] >= 2.0  # first breakpoint is also >= min_gap from x_lo


def test_min_gap_breaks_zero_slack_gives_exact_floor_spacing():
    # No slack to distribute: breakpoints land exactly min_gap apart from x_lo.
    breaks = _min_gap_breaks(np.zeros(3), x_lo=0.0, min_gap=4.0)

    np.testing.assert_allclose(breaks, [4.0, 8.0, 12.0])


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


def test_piecewise_fit_cached_key_is_sensitive_to_series_values(tmp_path):
    series_a = TWO_SEGMENT_SERIES
    series_b = TWO_SEGMENT_SERIES + 1.0

    out_a = piecewise_fit_cached(series_a, "D", max_segments=2, cache_dir=tmp_path)
    out_b = piecewise_fit_cached(series_b, "D", max_segments=2, cache_dir=tmp_path)

    assert len(list((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet"))) == 2
    assert out_a["value"].iloc[0] != out_b["value"].iloc[0]

    # Re-fetching series_a must return its own cached entry, not series_b's.
    out_a_again = piecewise_fit_cached(series_a, "D", max_segments=2, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(out_a, out_a_again)


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


def test_invalidate_piecewise_fit_cache_prunes_by_mtime_not_atime(tmp_path):
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "W", max_segments=2, cache_dir=tmp_path)
    daily, weekly = sorted((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet"))

    # daily: newest atime, oldest mtime. weekly: oldest atime, newest mtime. LRU by
    # mtime must keep weekly; atime-based LRU would keep daily instead.
    now = time.time()
    os.utime(daily, (now + 1000, now - 1000))
    os.utime(weekly, (now - 1000, now + 1000))

    invalidate_piecewise_fit_cache(tmp_path, max_cached=1)

    remaining = list((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet"))
    assert remaining == [weekly]


def test_piecewise_fit_cached_hit_touches_cache_file_mtime(tmp_path):
    piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)
    cache_file = next((tmp_path / PIECEWISE_CACHE_DIR).glob("*.parquet"))
    old_time = time.time() - 10_000
    os.utime(cache_file, (old_time, old_time))

    piecewise_fit_cached(TWO_SEGMENT_SERIES, "D", max_segments=2, cache_dir=tmp_path)

    assert cache_file.stat().st_mtime > old_time
