"""Common utility functions for fitness_analysis module."""

import multiprocessing
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import activity_parser
import numpy as np
import pandas as pd
import pint
import pwlf
import sklearn.linear_model
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

RECORDS_CACHE_DIR = "activity_records"

# Global objects that can be used throughout the fitness_analysis module
parser = activity_parser.ActivityParser()
tz_finder = timezonefinder.TimezoneFinder()


def merge_excel_files(path: str | PathLike[str]) -> dict[str, pd.DataFrame]:
    """Loads and merges data from all Excel files in a directory.

    Loads all sheets of all [xls,xlsx] files in a directory and merges them into
    one mapping keyed by sheet name. Identically named sheets from each Excel
    file are concatenated in alphabetical order of Excel file name.

    Args:
        path: Directory of Excel files.

    Returns:
        A mapping of merged sheet data keyed by sheet name.
    """

    data_parts: dict[str, list[pd.DataFrame]] = {}
    files = sorted(Path(path).iterdir())

    for f in files:
        if f.suffix.lower() not in [".xls", ".xlsx"]:
            continue

        excel = pd.read_excel(f, sheet_name=None)
        for sheet_name, sheet_data in excel.items():
            if sheet_data.empty:
                continue
            data_parts.setdefault(sheet_name, []).append(sheet_data)

    merged_data: dict[str, pd.DataFrame] = {}
    for sheet_name, parts in data_parts.items():
        if len(parts) == 1:
            merged_data[sheet_name] = parts[0]
        else:
            merged_data[sheet_name] = pd.concat(parts).reset_index(drop=True)

    return merged_data


def _per_unit_us_factor(units: str) -> np.uint64:
    """Get conversion factor from 1 ``units`` to microseconds."""

    return np.uint64(np.timedelta64(np.timedelta64(1, units), "us"))


def _to_seconds(
    index: pd.Index | Iterable[pd.Timestamp], t0_us: float
) -> np.ndarray:
    """Convert timestamps to float seconds since a reference time."""

    us = np.array(index).astype("datetime64[us]").astype(np.float64)
    return (us - t0_us) / 1e6


def _pwlf_segments(
    series: pd.Series,
    x: np.ndarray,
    *,
    num_segments: int,
) -> tuple[pwlf.PiecewiseLinFit, np.ndarray, np.ndarray]:
    """Fit a pwlf model using a fixed number of segments."""

    model = pwlf.PiecewiseLinFit(x, series.values)
    breaks_x = model.fit(num_segments)
    regression = model.predict(x)

    return model, breaks_x, regression


def _pwlf_breaks(
    series: pd.Series,
    x: np.ndarray,
    *,
    breaks_x: np.ndarray,
) -> tuple[pwlf.PiecewiseLinFit, np.ndarray, np.ndarray]:
    """Fit a pwlf model using explicit breakpoints."""

    model = pwlf.PiecewiseLinFit(x, series.values)
    model.fit_with_breaks(breaks_x)
    regression = model.predict(x)

    return model, breaks_x, regression


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

    # x in seconds from t0: avoids float64 precision loss with raw µs (~1.7e18)
    t0_us = float(
        np.array(series.index[:1])
        .astype("datetime64[us]")
        .astype(np.float64)[0]
    )
    x_s = _to_seconds(series.index, t0_us)
    s_per_unit = float(_per_unit_us_factor(units)) / 1e6  # µs/unit → s/unit

    if breaks is not None:
        breaks_s = _to_seconds(breaks, t0_us)
        model, bp_s, regression = _pwlf_breaks(series, x_s, breaks_x=breaks_s)
    else:
        model, bp_s, regression = _pwlf_segments(
            series, x_s, num_segments=num_segments
        )

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


def _to_time_us_uint64(index: pd.Index | Iterable[pd.Timestamp]) -> np.ndarray:
    """Convert datetime-like values to ``datetime64[us]`` as ``uint64``."""

    return np.array(index).astype("datetime64[us]").astype(np.uint64)


def time_series_linear_rate(series: pd.Series, units: str) -> float:
    """Calculate the slope of the linear regression of a time series.

    Args:
        series: Time-indexed values on which to perform regression.
        units: Time unit to use for calculating slope (e.g., 'D' means calculate
          rate per day).

    Returns:
        Calculated slope as a rate per ``units``.
    """

    # Convert time index to a uint array (in us) for scipy calculations
    x = _to_time_us_uint64(series.index)

    # Do an ordinary linear regression
    model = sklearn.linear_model.LinearRegression()
    model.fit(x.reshape(-1, 1), series.values)

    # Get the conversion factor to desired rate units
    c = _per_unit_us_factor(units)

    return c * model.coef_[0]


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


def parse_record_cached(
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Parse a FIT/TCX/GPX file, using a Parquet cache when available.

    Worker-safe: picklable and suitable for use inside multiprocessing pools.

    The cache key is the filename only — activity files are assumed immutable
    after Strava export. To invalidate, delete the cache subdirectory.

    Args:
        path: Activity file to parse.
        cache_dir: Optional cache directory.

    Returns:
        Parsed records.
    """

    if cache_dir is None:
        records, _, _ = parser.parse(path)
        return records

    parquet_path = (
        Path(cache_dir) / RECORDS_CACHE_DIR / (Path(path).name + ".parquet")
    )
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    records, _, _ = parser.parse(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Coerce object-dtype columns to string — mixed-type columns cannot be
    # serialized by pyarrow without a consistent type.
    obj_cols = records.select_dtypes(include="object").columns
    records.astype({col: "string" for col in obj_cols}).to_parquet(parquet_path)

    return records


def ensure_record_cached(
    path: str | PathLike[str],
    cache_dir: str | PathLike[str],
) -> None:
    """Parse a FIT/TCX/GPX file and write its parquet cache if not present.

    Returns None so pool workers avoid pickling DataFrames over IPC.
    Worker-safe: picklable and suitable for use inside multiprocessing pools.
    """

    parse_record_cached(path, cache_dir)


def warm_records_cache(
    files: Iterable[str | PathLike[str]],
    path: str | PathLike[str],
    cache_dir: str | PathLike[str],
) -> None:
    """Ensure parquet files exist for all given activity files.

    Identifies files without a parquet cache and parses them, pooling workers
    when there are enough cold files to justify the overhead. Files with an
    existing parquet cache are skipped.

    Args:
        files: Activity filenames (relative to ``path``).
        path: Directory containing the activity files.
        cache_dir: Cache directory containing the parquet subdirectory.
    """

    parquet_dir = Path(cache_dir) / RECORDS_CACHE_DIR
    cold = [
        f
        for f in files
        if not (parquet_dir / (Path(f).name + ".parquet")).exists()
    ]
    if not cold:
        return

    args = [(Path(path) / f, cache_dir) for f in cold]
    if len(cold) > multiprocessing.cpu_count() * 2:
        with multiprocessing.Pool() as p:
            p.starmap(ensure_record_cached, args)
    else:
        for a in args:
            ensure_record_cached(*a)


def invalidate_records_cache(
    cache_dir: str | PathLike[str],
    files: Iterable[str | PathLike[str]] | None = None,
) -> None:
    """Invalidate the records (parquet) cache.

    If ``files`` is None, deletes all parquet files in the cache directory.
    Otherwise removes only the parquet files for the given activity filenames.

    Args:
        cache_dir: Cache directory passed to ``load_activity_records`` or
            ``load_strava_activities``.
        files: Activity filenames whose parquet files should be removed.
            If None, the entire records cache is cleared.
    """

    parquet_dir = Path(cache_dir) / RECORDS_CACHE_DIR
    if not parquet_dir.exists():
        return
    if files is None:
        for p in parquet_dir.glob("*.parquet"):
            p.unlink()
        return
    for f in files:
        parquet_path = parquet_dir / (Path(f).name + ".parquet")
        if parquet_path.exists():
            parquet_path.unlink()


def load_activity_records(
    files: pd.Series,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> list[pd.DataFrame]:
    """Load parsed records for a set of activity files.

    Parses each file using a Parquet cache when available. When a cache
    directory is provided, cold files (no parquet yet) are pooled for
    parallel parsing, then all files are read serially from the warm cache.
    Without a cache directory, parsing is pooled directly for large batches.

    Args:
        files: Activity filenames (relative to ``path``).
        path: Directory containing the activity files.
        cache_dir: Optional cache directory.

    Returns:
        Parsed records, one per file, in the same order as ``files``.
    """

    if cache_dir is None:
        args = [(Path(path) / f, None) for f in files]
        if len(files) > multiprocessing.cpu_count() * 2:
            with multiprocessing.Pool() as p:
                return p.starmap(parse_record_cached, args)
        return [parse_record_cached(*a) for a in args]

    # Pass 1: pool cold files (raw FIT parse → write parquet)
    warm_records_cache(files, path, cache_dir)

    # Pass 2: read all from parquet serially
    return [parse_record_cached(Path(path) / f, cache_dir) for f in files]
