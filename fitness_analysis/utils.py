"""Shared constants and utility functions for fitness_analysis module."""

import math

import numpy as np
import pandas as pd
import pint
import ruptures as rpt
import timezonefinder
from sklearn.preprocessing import StandardScaler

# Unit conversion factors derived from pint
_ureg = pint.UnitRegistry()

KM_TO_MI = _ureg.Quantity(1, "km").to("mile").magnitude
M_TO_FT = _ureg.Quantity(1, "m").to("ft").magnitude
LBS_TO_KG = _ureg.Quantity(1, "lb").to("kg").magnitude

# Physiological approximation: 1 lb of body fat ≈ 3500 kcal
# Divided by 7 days/week → kcal/day per lb/week of weight change
_FAT_KCAL_PER_LB = 3500
CAL_PER_LB_WEEK = _FAT_KCAL_PER_LB / 7

# Geographical constants
EARTH_RADIUS_M = 6_378_137.0  # WGS-84 semi-major axis in metres
EARTH_M_PER_DEG = 2 * math.pi * EARTH_RADIUS_M / 360

# Global timezone finder shared across the fitness_analysis module
tz_finder = timezonefinder.TimezoneFinder()


# ---------------------------------------------------------------------------
# Averaging and rate estimation helpers
# ---------------------------------------------------------------------------


def ewm_min_periods_from_halflife(
    halflife: str | pd.Timedelta,
    coverage: float,
    floor: int = 2,
) -> int:
    """Derive EWM ``min_periods`` from half-life and target weight coverage.

    Args:
        halflife: EWM half-life (for example, ``'3D'``).
        coverage: Target cumulative EWM weight mass in ``(0, 1)``.
        floor: Minimum returned value.

    Returns:
        Integer ``min_periods`` aligned to the specified half-life.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")

    halflife_days = pd.to_timedelta(halflife) / pd.Timedelta("1D")
    if halflife_days <= 0:
        raise ValueError("halflife must be positive")

    daily_decay = 0.5 ** (1 / halflife_days)
    periods = int(np.ceil(np.log(1 - coverage) / np.log(daily_decay)))

    return max(floor, periods)


def rolling_linear_rate(
    series: pd.Series,
    window: int,
    min_periods: int,
    units: str,
    *,
    center: bool = True,
) -> pd.Series:
    """Compute the rolling OLS slope of a regularly-sampled time series.

    The series must be on a regular time grid but may contain NaN for missing
    observations. The grid step is inferred from the median index spacing.
    Each window's slope is computed using only the non-NaN observations, with
    their actual time offsets within the window used as x-values. When no
    values are missing the computation reduces to a fixed-kernel dot product.

    Args:
        series: Regularly-gridded time-indexed values. NaN entries are treated
            as missing observations and excluded from each window's regression.
        window: Rolling window size in grid steps.
        min_periods: Minimum number of non-NaN observations required to
            produce a result; windows with fewer yield NaN.
        units: Output rate units (e.g. ``"W"`` for per week, ``"D"`` for
            per day).
        center: If True, the window is centered on each observation. If False,
            the window is trailing (causal).

    Returns:
        Rolling slope aligned to the input index.
    """
    # Infer grid step from median index spacing to support any regular grid
    step_us = np.median(
        np.diff(
            np.array(series.index).astype("datetime64[us]").astype(np.float64)
        )
    )
    units_per_step = (
        np.timedelta64(1, units) / np.timedelta64(1, "us")
    ) / step_us

    def _slope(arr: np.ndarray) -> float:
        valid = ~np.isnan(arr)
        pos = np.where(valid)[0].astype(np.float64)
        pm = pos - pos.mean()
        return units_per_step * (pm @ arr[valid]) / (pm @ pm)

    roller = series.rolling(window, min_periods=min_periods, center=center)
    return roller.apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# GPS and activity helpers
# ---------------------------------------------------------------------------


def infer_timezone(records: pd.DataFrame) -> str | None:
    """Determine the timezone of an activity from its GPS location data.

    Finds the first valid latitude and longitude in the data and calculates the
    timezone for that GPS location.

    Args:
        records: Activity data to process.

    Returns:
        Timezone name, or None if no valid latitude/longitude is available
        or no timezone match is found.
    """
    if "latitude" not in records.columns or "longitude" not in records.columns:
        return None

    lat_col = records["latitude"]
    lng_col = records["longitude"]

    idx = lat_col.first_valid_index()
    if idx is None:
        return None

    return tz_finder.timezone_at(lng=lng_col.loc[idx], lat=lat_col.loc[idx])


def identify_inactive_periods(
    series: pd.Series,
    activity_threshold: float,
    min_duration: pd.Timedelta,
) -> pd.Series:
    """Identify periods where values in a time series are not changing.

    Returns a boolean mask identifying periods where the velocity of
    ``series`` (in units per second) remains below ``activity_threshold`` for
    at least ``min_duration``.

    Args:
        series: Time-indexed values.
        activity_threshold: Velocity, in units per second, below which values
            of ``series`` are considered inactive.
        min_duration: Minimum inactivity span to flag.

    Returns:
        Time-indexed boolean mask.
    """
    # Velocity in units/second between consecutive samples; first sample has
    # no predecessor so we assign zero.
    t_s = (series.index - series.index[0]).total_seconds().to_numpy()
    vals = series.to_numpy(dtype=float)
    velocity = np.concatenate([[0.0], np.diff(vals) / np.diff(t_s)])

    # Identify starts and ends of contiguous runs below velocity threshold;
    # padding ensures every run has a start and end
    below = velocity < activity_threshold
    below_pad = np.concatenate([[False], below, [False]])
    change_at = np.diff(below_pad).nonzero()[0]
    starts, ends = change_at[0::2], change_at[1::2]

    # Mark contiguous runs of sufficient duration
    result = np.zeros(len(series), dtype=bool)
    run_durations = t_s[ends - 1] - t_s[starts]
    long_enough = run_durations >= min_duration.total_seconds()
    for s, e in zip(starts[long_enough], ends[long_enough]):
        result[s:e] = True

    return pd.Series(result, index=series.index)


# ---------------------------------------------------------------------------
# Power curve computation
# ---------------------------------------------------------------------------


def power_curve_windows(max_s: int, density: int) -> np.ndarray:
    """Logarithmically-spaced integer window sizes.

    Args:
        max_s: Upper bound window size in seconds.
        density: Logspace samples per decade before deduplication.

    Returns:
        Approximately logarithmically-spaced 1-D array of window sizes in
        seconds, from 1 to ``max_s``, rounded to integers.
    """
    n = round(density * np.log10(max_s))
    return pd.unique(np.round(np.logspace(0, np.log10(max_s), n)).astype(int))


def compute_power_curve(
    power_series: pd.Series, windows: np.ndarray
) -> np.ndarray | None:
    """Calculate max mean power (W) for each window size.

    Args:
        power_series: Power values with a DatetimeIndex. NaN entries are
            dropped before resampling.
        windows: Integer window sizes in seconds, as from
            ``power_curve_windows``.

    Returns:
        1-D array of max mean power (W) aligned to ``windows``, or ``None`` if
        there is no usable power data.
    """
    # Resample to a regular 1-second grid, treating gaps as coasting at 0 W
    power_arr = (
        power_series.dropna()
        .resample("s")
        .mean()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    if len(power_arr) == 0:
        return None

    # Re-use a single cumsum for each window computation:
    # cumsum[n+window] - cumsum[n] = window sum at n,
    # dividing max by window gives best mean power
    results = np.empty(len(windows))
    cumsum = np.concatenate(([0.0], np.cumsum(power_arr)))
    for i, window in enumerate(windows):
        if window > len(power_arr):
            results[i] = np.nan
        else:
            results[i] = (cumsum[window:] - cumsum[:-window]).max() / window

    return results


# ---------------------------------------------------------------------------
# Changepoint detection
# ---------------------------------------------------------------------------


def pelt_segments(
    signal: pd.DataFrame,
    penalty: float,
    min_size: int,
) -> pd.DataFrame:
    """Detect changepoints in a signal using PELT.

    Scales ``signal`` before fitting so that all features contribute equally
    regardless of their units or variance.

    Args:
        signal: ``(N, D)`` DataFrame of features. Rows with any NaN are
            dropped before fitting.
        penalty: PELT penalty. Higher values produce fewer segments.
        min_size: Minimum observations per segment.

    Returns:
        One row per detected span with ``start`` and ``end`` columns.
        Empty if the signal has zero rows.
    """
    signal = signal.dropna().sort_index()
    index = signal.index

    if len(signal) == 0:
        return pd.DataFrame(columns=["start", "end"])

    scaled = StandardScaler().fit_transform(signal.values)
    breakpoints = (
        rpt.Pelt(model="l2", min_size=min_size, jump=1)
        .fit(scaled)
        .predict(pen=penalty)
    )

    starts = [index[0]] + [index[bp] for bp in breakpoints[:-1]]
    ends = [index[bp - 1] for bp in breakpoints[:-1]] + [index[-1]]
    return pd.DataFrame(
        {"start": starts, "end": ends}, index=range(len(starts))
    )
