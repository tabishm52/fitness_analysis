"""Tests for shared constants and utility functions."""

import numpy as np
import pandas as pd
import pytest

from fitness_analysis.utils import (
    compute_power_curve,
    ewm_min_periods_from_halflife,
    identify_inactive_periods,
    infer_timezone,
    pelt_segments,
    power_curve_windows,
    rolling_linear_rate,
)

# --------------------------------------------------------------------------------------
# ewm_min_periods_from_halflife
# --------------------------------------------------------------------------------------


def test_ewm_min_periods_from_halflife_achieves_target_coverage():
    # Per half-life decay, the k most recent daily weights carry mass 1 - d**k of the
    # total, where d = 0.5**(1/halflife_days). min_periods is the smallest k clearing
    # `coverage` (floor doesn't bind here: 10 > the default floor of 2).
    halflife_days = 3.0
    coverage = 0.9

    min_periods = ewm_min_periods_from_halflife(f"{halflife_days:g}D", coverage)

    daily_decay = 0.5 ** (1 / halflife_days)
    mass_covered = 1 - daily_decay**min_periods
    mass_covered_one_less = 1 - daily_decay ** (min_periods - 1)
    assert mass_covered >= coverage
    assert mass_covered_one_less < coverage


def test_ewm_min_periods_from_halflife_floor_wins_for_short_halflife():
    assert ewm_min_periods_from_halflife("10D", 0.1) == 2


def test_ewm_min_periods_from_halflife_respects_custom_floor():
    assert ewm_min_periods_from_halflife("10D", 0.1, floor=5) == 5


def test_ewm_min_periods_from_halflife_rejects_coverage_out_of_range():
    with pytest.raises(ValueError, match="coverage"):
        ewm_min_periods_from_halflife("3D", 1.0)


def test_ewm_min_periods_from_halflife_rejects_nonpositive_halflife():
    with pytest.raises(ValueError, match="halflife"):
        ewm_min_periods_from_halflife("0D", 0.9)


# --------------------------------------------------------------------------------------
# rolling_linear_rate
# --------------------------------------------------------------------------------------


def test_rolling_linear_rate_constant_slope():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    series = pd.Series(np.arange(10) * 2.0, index=idx)

    out = rolling_linear_rate(series, window=5, min_periods=3, units="D", center=True)

    np.testing.assert_allclose(out.dropna().to_numpy(), 2.0)


def test_rolling_linear_rate_ignores_nan_gap_within_window():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    series = pd.Series(np.arange(10) * 2.0, index=idx)
    series.iloc[4] = np.nan

    out = rolling_linear_rate(series, window=5, min_periods=3, units="D", center=True)

    np.testing.assert_allclose(out.dropna().to_numpy(), 2.0)


def test_rolling_linear_rate_trailing_window_is_causal():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    series = pd.Series(np.arange(10) * 2.0, index=idx)

    out = rolling_linear_rate(series, window=3, min_periods=2, units="D", center=False)

    assert pd.isna(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:].to_numpy(), 2.0)


def test_rolling_linear_rate_single_valid_point_in_window_is_nan():
    # A window with only one non-NaN observation has an undefined slope: the OLS
    # denominator (sum of squared centered positions) is zero.
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    series = pd.Series([np.nan] * 4 + [5.0] + [np.nan] * 5, index=idx)

    out = rolling_linear_rate(series, window=5, min_periods=1, units="D", center=True)

    assert out.isna().all()


# --------------------------------------------------------------------------------------
# infer_timezone
# --------------------------------------------------------------------------------------


def test_infer_timezone_uses_first_valid_position():
    records = pd.DataFrame({"latitude": [np.nan, 37.7749], "longitude": [np.nan, -122.4194]})

    assert infer_timezone(records) == "America/Los_Angeles"


def test_infer_timezone_returns_none_without_gps_columns():
    records = pd.DataFrame({"heart_rate": [120, 130]})

    assert infer_timezone(records) is None


def test_infer_timezone_returns_none_when_all_positions_missing():
    records = pd.DataFrame({"latitude": [np.nan], "longitude": [np.nan]})

    assert infer_timezone(records) is None


# --------------------------------------------------------------------------------------
# identify_inactive_periods
# --------------------------------------------------------------------------------------


def test_identify_inactive_periods_flags_low_velocity_runs():
    idx = pd.date_range("2026-01-01", periods=10, freq="min")
    series = pd.Series([0, 0, 0, 0, 5, 10, 15, 15, 15, 15], index=idx, dtype=float)

    out = identify_inactive_periods(
        series, activity_threshold=0.01, min_duration=pd.Timedelta(minutes=2)
    )

    expected = pd.Series([True, True, True, True, False, False, False, True, True, True], index=idx)
    pd.testing.assert_series_equal(out, expected)


def test_identify_inactive_periods_short_run_not_flagged():
    idx = pd.date_range("2026-01-01", periods=6, freq="min")
    series = pd.Series([0, 0, 5, 5, 10, 10], index=idx, dtype=float)

    out = identify_inactive_periods(
        series, activity_threshold=0.01, min_duration=pd.Timedelta(minutes=3)
    )

    assert not out.any()


def test_identify_inactive_periods_duplicate_timestamp_same_value_is_inactive():
    # Zero elapsed time and zero value change (0/0 = NaN): the value genuinely didn't
    # change, so this counts as inactive.
    idx = pd.DatetimeIndex(["2026-01-01T00:00:00", "2026-01-01T00:00:00", "2026-01-01T00:01:00"])
    series = pd.Series([1.0, 1.0, 2.0], index=idx)

    out = identify_inactive_periods(series, activity_threshold=0.1, min_duration=pd.Timedelta(0))

    assert out.iloc[1]


def test_identify_inactive_periods_duplicate_timestamp_changed_value_is_active():
    # Zero elapsed time but a real value change (nonzero/0 = inf): genuine evidence of
    # activity.
    idx = pd.DatetimeIndex(["2026-01-01T00:00:00", "2026-01-01T00:00:00", "2026-01-01T00:01:00"])
    series = pd.Series([1.0, 5.0, 2.0], index=idx)

    out = identify_inactive_periods(series, activity_threshold=0.1, min_duration=pd.Timedelta(0))

    assert not out.iloc[1]


# --------------------------------------------------------------------------------------
# power_curve_windows / compute_power_curve
# --------------------------------------------------------------------------------------


def test_power_curve_windows_spans_1_to_max_s():
    windows = power_curve_windows(60, density=10)

    assert windows[0] == 1
    assert windows[-1] == 60
    assert list(windows) == sorted(set(windows))


def test_compute_power_curve_matches_hand_computed_maxima():
    idx = pd.date_range("2026-01-01", periods=20, freq="s")
    power = pd.Series([100.0] * 10 + [200.0] * 10, index=idx)

    out = compute_power_curve(power, np.array([1, 5, 10, 20]))

    assert out == pytest.approx([200.0, 200.0, 200.0, 150.0])


def test_compute_power_curve_returns_none_for_empty_series():
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))

    assert compute_power_curve(empty, np.array([1, 5])) is None


def test_compute_power_curve_nan_beyond_series_length():
    idx = pd.date_range("2026-01-01", periods=5, freq="s")
    power = pd.Series([100.0] * 5, index=idx)

    out = compute_power_curve(power, np.array([3, 10]))

    assert out is not None
    assert out[0] == pytest.approx(100.0)
    assert np.isnan(out[1])


# --------------------------------------------------------------------------------------
# pelt_segments
# --------------------------------------------------------------------------------------


def test_pelt_segments_detects_single_changepoint():
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    signal = pd.DataFrame({"x": [0.0] * 10 + [10.0] * 10}, index=idx)

    out = pelt_segments(signal, penalty=1.0, min_size=2)

    assert list(out["start"]) == [idx[0], idx[10]]
    assert list(out["end"]) == [idx[9], idx[19]]


def test_pelt_segments_empty_signal_returns_empty_frame():
    empty = pd.DataFrame({"x": pd.Series([], dtype=float)})

    out = pelt_segments(empty, penalty=1.0, min_size=2)

    assert out.empty
    assert list(out.columns) == ["start", "end"]


def test_pelt_segments_drops_nan_rows_before_fitting():
    idx = pd.date_range("2026-01-01", periods=11, freq="D")
    signal = pd.DataFrame({"x": [0.0] * 5 + [np.nan] + [10.0] * 5}, index=idx)

    out = pelt_segments(signal, penalty=1.0, min_size=2)

    assert idx[5] not in set(out["start"]) | set(out["end"])
