"""Piecewise linear regression on time-indexed series, with caching."""

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pwlf
from scipy.optimize import differential_evolution

PIECEWISE_CACHE_DIR = "piecewise_fit"


def _to_seconds(
    series: pd.Series, units: str
) -> tuple[pd.Series, pd.Timestamp, np.ndarray, float]:
    """Common setup shared by every piecewise fit entry point."""
    clean = series.dropna()
    t0 = clean.index[0]
    x_s = (clean.index - t0).total_seconds().to_numpy()
    s_per_unit = np.timedelta64(1, units) / np.timedelta64(1, "s")
    return clean, t0, x_s, s_per_unit


def _min_gap_breaks(u: np.ndarray, x_lo: float, min_gap: float) -> np.ndarray:
    """Map a slack budget to breakpoints spaced >= min_gap apart."""
    slack = np.sort(u)  # Nondecreasing; caller bounds each u_i to [0, slack]
    floor = min_gap * np.arange(1, len(u) + 1)  # Guaranteed per-breakpoint min
    return x_lo + floor + slack


def _fit_n_segments(
    model: pwlf.PiecewiseLinFit,
    x_lo: float,
    x_hi: float,
    n_segs: int,
    min_segment_s: float,
) -> tuple[np.ndarray, float] | None:
    """Fit `model` with `n_segs` segments respecting the minimum segment length.

    Returns ``(breaks_s, ssr)``, or ``None`` if `n_segs` segments can't fit in
    the series span given `min_segment_s`.
    """
    slack = (x_hi - x_lo) - n_segs * min_segment_s  # Room beyond n_segs floors
    if slack < 0:
        return None
    if n_segs == 1:
        breaks_s = np.array([x_lo, x_hi])
        return breaks_s, model.fit_with_breaks_opt(np.array([]))

    model.use_custom_opt(n_segs)

    def objective(u: np.ndarray) -> float:
        interior = _min_gap_breaks(u, x_lo, min_segment_s)
        return model.fit_with_breaks_opt(interior)

    # Each free var searches slack in [0, slack]; _min_gap_breaks turns that
    # into ordered breakpoints that always respect the floor
    result = differential_evolution(
        objective, bounds=[(0, slack)] * (n_segs - 1)
    )
    interior = _min_gap_breaks(result.x, x_lo, min_segment_s)
    breaks_s = np.concatenate(([x_lo], interior, [x_hi]))
    return breaks_s, result.fun


def _build_result(
    model: pwlf.PiecewiseLinFit,
    breaks_s: np.ndarray,
    t0: pd.Timestamp,
    s_per_unit: float,
) -> pd.DataFrame:
    """Fit `model` to `breaks_s` and package it as a value/rate DataFrame."""
    model.fit_with_breaks(breaks_s)
    bp_timestamps = t0 + pd.to_timedelta(breaks_s, unit="s")
    result = pd.DataFrame(index=bp_timestamps)
    result["value"] = model.predict(breaks_s)
    result["rate"] = s_per_unit * np.append(model.calc_slopes(), np.nan)
    return result


def piecewise_fit(
    series: pd.Series,
    n_segments: int,
    units: str,
    min_segment_duration: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Fit a piecewise linear regression on a time series.

    Args:
        series: Time-indexed values. NaN entries are dropped before fitting.
        n_segments: Number of segments for piecewise regression.
        units: Timedelta unit string for slope (e.g. 'D' per day, 'W' per week).
        min_segment_duration: Minimum duration enforced on every segment.
            ``None`` disables the floor.

    Returns:
        Time-indexed breakpoints with ``value`` and ``rate`` columns.
        The final row's ``rate`` is NaN (no outgoing segment).
    """
    clean, t0, x_s, s_per_unit = _to_seconds(series, units)
    x_lo, x_hi = x_s[0], x_s[-1]
    min_segment_s = (
        min_segment_duration.total_seconds()
        if min_segment_duration is not None
        else 0.0
    )

    model = pwlf.PiecewiseLinFit(x_s, clean.values)
    candidate = _fit_n_segments(model, x_lo, x_hi, n_segments, min_segment_s)
    assert candidate is not None, (
        f"min_segment_duration={min_segment_duration} makes "
        f"n_segments={n_segments} infeasible over the series span"
    )
    breaks_s, _ssr = candidate
    return _build_result(model, breaks_s, t0, s_per_unit)


def piecewise_fit_with_breaks(
    series: pd.Series,
    breaks: Iterable[pd.Timestamp],
    units: str,
) -> pd.DataFrame:
    """Fit a piecewise linear regression with specified breakpoints.

    Args:
        series: Time-indexed values. NaN entries are dropped before fitting.
        breaks: Breakpoint timestamps for piecewise regression.
        units: Timedelta unit string for slope (e.g. 'D' per day, 'W' per week).

    Returns:
        Time-indexed breakpoints with ``value`` and ``rate`` columns.
        The final row's ``rate`` is NaN (no outgoing segment).
    """
    clean, t0, x_s, s_per_unit = _to_seconds(series, units)
    breaks_s = (pd.DatetimeIndex(breaks) - t0).total_seconds().to_numpy()
    model = pwlf.PiecewiseLinFit(x_s, clean.values)
    return _build_result(model, breaks_s, t0, s_per_unit)


def piecewise_fit_auto(
    series: pd.Series,
    units: str,
    max_segments: int = 6,
    min_segment_duration: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Fit a piecewise linear regression with BIC-selected segment count.

    Args:
        series: Time-indexed values. NaN entries are dropped before fitting.
        units: Timedelta unit string for slope (e.g. 'D' per day, 'W' per week).
        max_segments: Upper bound on the candidate segment count.
        min_segment_duration: Minimum duration enforced on every segment during
            the search. ``None`` disables the floor.

    Returns:
        Time-indexed breakpoints with ``value`` and ``rate`` columns.
        The final row's ``rate`` is NaN (no outgoing segment).
    """
    assert max_segments >= 1

    clean, t0, x_s, s_per_unit = _to_seconds(series, units)
    n = len(x_s)
    x_lo, x_hi = x_s[0], x_s[-1]
    min_segment_s = (
        min_segment_duration.total_seconds()
        if min_segment_duration is not None
        else 0.0
    )

    model = pwlf.PiecewiseLinFit(x_s, clean.values)
    best_bic = float("inf")
    best_breaks_s = None

    # Gaussian-likelihood BIC; Number of parameters k = n_segs slopes +
    # 1 intercept + (n_segs-1) interior breakpoints + noise variance
    for n_segs in range(1, max_segments + 1):
        candidate = _fit_n_segments(model, x_lo, x_hi, n_segs, min_segment_s)
        if candidate is None:
            break  # Larger n_segs only needs more slack
        breaks_s, ssr = candidate
        k = 2 * n_segs + 1
        bic = n * np.log(ssr / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_breaks_s = breaks_s

    assert best_breaks_s is not None, (
        f"min_segment_duration={min_segment_duration} exceeds the series span"
    )
    return _build_result(model, best_breaks_s, t0, s_per_unit)


def piecewise_fit_cached(
    series: pd.Series,
    units: str,
    max_segments: int = 6,
    min_segment_duration: pd.Timedelta | None = None,
    cache_dir: Path | None = None,
    max_cached: int = 30,
) -> pd.DataFrame:
    """Disk-cached wrapper around ``piecewise_fit_auto``.

    Keys the cache on the cleaned series and parameter arguments; a hit is
    read back from parquet, a miss is computed and written.

    Args:
        series: Time-indexed values. NaN entries are dropped before fitting.
        units: Timedelta unit string for slope (e.g. 'D' per day, 'W' per week).
        max_segments: Upper bound on the candidate segment count.
        min_segment_duration: Minimum duration for any segment.
        cache_dir: Directory for caching the result. Pass ``None`` to skip
            caching and fit directly.
        max_cached: Maximum number of cached results to retain; least recently
            accessed are pruned.

    Returns:
        Time-indexed breakpoints with ``value`` and ``rate`` columns.
        The final row's ``rate`` is NaN (no outgoing segment).
    """
    if cache_dir is None:
        return piecewise_fit_auto(
            series, units, max_segments, min_segment_duration
        )

    clean = series.dropna()
    key = hashlib.md5(
        pd.util.hash_pandas_object(clean, index=True).values.tobytes()
        + units.encode()
        + str(max_segments).encode()
        + str(min_segment_duration).encode()
    ).hexdigest()[:16]
    cache_path = cache_dir / PIECEWISE_CACHE_DIR / f"{key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    result = piecewise_fit_auto(
        clean, units, max_segments, min_segment_duration
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path)
    invalidate_piecewise_fit_cache(cache_dir, max_cached)

    return result


def invalidate_piecewise_fit_cache(
    cache_dir: Path | None = None,
    max_cached: int = 0,
) -> None:
    """Invalidate cached piecewise_fit_cached results.

    Args:
        cache_dir: Directory to prune. Pass ``None`` (the default) to no-op.
        max_cached: Maximum number of cached results to retain; least recently
            accessed are pruned. Defaults to 0 (delete all).
    """
    if cache_dir is None:
        return

    target = cache_dir / PIECEWISE_CACHE_DIR
    if not target.exists():
        return

    files = sorted(target.iterdir(), key=lambda f: f.stat().st_atime)
    for f in files[:-max_cached] if max_cached > 0 else files:
        f.unlink(missing_ok=True)
    if max_cached <= 0:
        target.rmdir()
