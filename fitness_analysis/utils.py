"""Common utility functions for fitness_analysis module."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pint
import pwlf
import timezonefinder

# Unit conversion factors derived from pint
_ureg = pint.UnitRegistry()

KM_TO_MI = _ureg.Quantity(1, "km").to("mile").magnitude
M_TO_FT = _ureg.Quantity(1, "m").to("ft").magnitude
LBS_TO_KG = _ureg.Quantity(1, "lb").to("kg").magnitude

# Physiological approximation: 1 lb of body fat ≈ 3500 kcal
# Divided by 7 days/week → kcal/day per lb/week of weight change
_FAT_KCAL_PER_LB = 3500
CAL_PER_LB_WEEK = _FAT_KCAL_PER_LB / 7

# Global timezone finder shared across the fitness_analysis module
tz_finder = timezonefinder.TimezoneFinder()


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


def _per_unit_us_factor(units: str) -> np.uint64:
    """Get conversion factor from 1 ``units`` to microseconds."""

    return np.uint64(np.timedelta64(np.timedelta64(1, units), "us"))


def _to_seconds(
    index: pd.Index | Iterable[pd.Timestamp], t0_us: float
) -> np.ndarray:
    """Convert timestamps to float seconds since a reference time."""

    us = np.array(index).astype("datetime64[us]").astype(np.float64)
    return (us - t0_us) / 1e6


def _pwlf_fit(
    series: pd.Series,
    units: str,
    breaks: Iterable[pd.Timestamp] | None = None,
    num_segments: int | None = None,
) -> pd.DataFrame:
    """Fit piecewise linear regression on a time-indexed series.

    Exactly one of ``breaks`` or ``num_segments`` must be provided.
    """

    if (breaks is None) == (num_segments is None):
        raise ValueError("Specify exactly one of breaks or num_segments")

    # x in seconds from t0
    t0_us = float(
        np.array(series.index[:1])
        .astype("datetime64[us]")
        .astype(np.float64)[0]
    )
    x_s = _to_seconds(series.index, t0_us)
    s_per_unit = float(_per_unit_us_factor(units)) / 1e6  # µs/unit → s/unit

    model = pwlf.PiecewiseLinFit(x_s, series.values)
    if breaks is not None:
        breaks_s = _to_seconds(breaks, t0_us)
        model.fit_with_breaks(breaks_s)
        bp_s = breaks_s
    else:
        bp_s = model.fit(num_segments)

    bp_us = (bp_s * 1e6 + t0_us).astype("datetime64[us]")
    result = pd.DataFrame(index=bp_us)
    result["value"] = model.predict(bp_s)
    result["rate"] = s_per_unit * np.append(model.calc_slopes(), [np.nan])

    return result


def piecewise_fit(
    series: pd.Series,
    n_segments: int,
    *,
    units: str,
) -> pd.DataFrame:
    """Fit a piecewise linear regression on a time series.

    Args:
        series: Time-indexed values. NaN entries are dropped before fitting.
        n_segments: Number of segments for piecewise regression.
        units: Time unit for slope at each breakpoint
            (e.g., 'D' for per day, 'W' for per week).

    Returns:
        Time-indexed breakpoints with ``value`` and ``rate`` columns.
    """

    return _pwlf_fit(series.dropna(), units, num_segments=n_segments)


def piecewise_fit_with_breaks(
    series: pd.Series,
    breaks: Iterable[pd.Timestamp],
    *,
    units: str,
) -> pd.DataFrame:
    """Fit a piecewise linear regression with specified breakpoints.

    Args:
        series: Time-indexed values. NaN entries are dropped before fitting.
        breaks: Breakpoint timestamps for piecewise regression.
        units: Time unit for slope at each breakpoint
            (e.g., 'D' for per day, 'W' for per week).

    Returns:
        Time-indexed breakpoints with ``value`` and ``rate`` columns.
    """

    return _pwlf_fit(series.dropna(), units, breaks=breaks)


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
    units_per_step = float(_per_unit_us_factor(units)) / step_us

    def _slope(arr: np.ndarray) -> float:
        valid = ~np.isnan(arr)
        pos = np.where(valid)[0].astype(np.float64)
        pm = pos - pos.mean()
        return units_per_step * (pm @ arr[valid]) / (pm @ pm)

    roller = series.rolling(window, min_periods=min_periods, center=center)
    return roller.apply(_slope, raw=True)


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

    # Calculate the derivative of series values, in units per second.
    # (this will be noisy but works well enough for our purposes)
    velocity = series.resample("s").interpolate().diff()
    below_threshold = velocity < activity_threshold

    # Split series into segments where velocity remains above or below
    # threshold, then calculate each segment duration.
    group_ids = below_threshold.ne(below_threshold.shift()).cumsum()
    durations = below_threshold.groupby(group_ids, sort=False).transform("size")

    # Return True where velocity stays below the threshold for min_duration.
    return (durations >= min_duration.total_seconds()) & below_threshold
