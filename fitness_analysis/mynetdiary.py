"""Functions for processing MyNetDiary fitness data."""

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils

MND_CACHE_DIR = "mnd_data"
MND_FINGERPRINT_FNAME = "_fingerprint.txt"


@dataclass
class MndTuning:
    """Tuning parameters for ``load_mnd_data``.

    Attributes:
        weight_halflife: Exponential half-life for weight smoothing.
        calorie_halflife: Exponential half-life for calorie smoothing.
        rate_window_days: Rolling window (days) for weight-rate regression.
    """

    weight_halflife: str = "3D"
    calorie_halflife: str = "9D"
    rate_window_days: int = 28


# ---------------------------------------------------------------------------
# Excel loading and cache
# ---------------------------------------------------------------------------


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
    excel_files = sorted(
        f for f in Path(path).iterdir() if f.suffix.lower() in {".xls", ".xlsx"}
    )

    for f in excel_files:
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


def merge_excel_files_cached(
    path: str | PathLike[str],
    cache_dir: str | PathLike[str],
) -> dict[str, pd.DataFrame]:
    """Load and merge Excel files, using a parquet cache when available.

    The cache is invalidated whenever the set of Excel files or any of their
    modification times changes. On a cache hit all sheets are read from
    parquet; on a miss the Excel files are read and the cache is written.

    Args:
        path: Directory of Excel files.
        cache_dir: Directory for the parquet cache.

    Returns:
        A mapping of merged sheet data keyed by sheet name.
    """

    excel_files = sorted(
        f for f in Path(path).iterdir() if f.suffix.lower() in {".xls", ".xlsx"}
    )
    fingerprint = hashlib.md5(
        json.dumps(
            sorted((f.name, f.stat().st_mtime) for f in excel_files)
        ).encode()
    ).hexdigest()

    cache_path = Path(cache_dir) / MND_CACHE_DIR
    fingerprint_path = cache_path / MND_FINGERPRINT_FNAME

    stored = fingerprint_path.read_text() if fingerprint_path.exists() else None
    cached = stored == fingerprint
    if cached:
        return {
            p.stem: pd.read_parquet(p) for p in cache_path.glob("*.parquet")
        }

    data = merge_excel_files(path)
    cache_path.mkdir(parents=True, exist_ok=True)

    for sheet_name, df in data.items():
        obj_cols = df.select_dtypes(include="object").columns
        df.astype({col: "string" for col in obj_cols}).to_parquet(
            cache_path / f"{sheet_name}.parquet"
        )
    fingerprint_path.write_text(fingerprint)

    return data


def invalidate_mnd_cache(cache_dir: str | PathLike[str]) -> None:
    """Invalidate the MyNetDiary parquet cache.

    Deletes the cache directory, forcing a full Excel re-read on the next
    call to ``load_mnd_data``.

    Args:
        cache_dir: Cache directory passed to ``load_mnd_data``.
    """

    cache_path = Path(cache_dir) / MND_CACHE_DIR
    if cache_path.exists():
        shutil.rmtree(cache_path)


# ---------------------------------------------------------------------------
# Estimated energy requirements
# ---------------------------------------------------------------------------


def eer_male(
    weight: pd.Series,
    height: float,
    dob: str | np.datetime64 | pd.Timestamp,
    pa: float = 1.0,
) -> pd.Series:
    """Male estimated energy requirements (per day) from MyNetDiary.

    Args:
        weight: Time-indexed weight measurements in pounds.
        height: Height, in inches.
        dob: Date of birth.
        pa: Activity level, 1.0 = sedentary, up to 1.45 for very active.

    Returns:
        Estimated daily energy requirement for each timestamp in ``weight``.
    """

    # Calculate time series of age in fractional years
    age = (weight.index - np.datetime64(dob)).days / 365.25

    # Perform male EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 662 - 9.53 * age + pa * (7.23 * weight + 13.71 * height)


def eer_female(
    weight: pd.Series,
    height: float,
    dob: str | np.datetime64 | pd.Timestamp,
    pa: float = 1.0,
) -> pd.Series:
    """Female estimated energy requirements (per day) from MyNetDiary.

    Args:
        weight: Time-indexed weight measurements in pounds.
        height: Height, in inches.
        dob: Date of birth.
        pa: Activity level, 1.0 = sedentary, up to 1.45 for very active.

    Returns:
        Estimated daily energy requirement for each timestamp in ``weight``.
    """

    # Calculate time series of age in fractional years
    age = (weight.index - np.datetime64(dob)).astype("timedelta64[D]") / 365.25

    # Perform female EER calculation per MyNetDiary
    # https://www.mynetdiary.com/supportArticle.do?articleId=328
    return 354 - 6.91 * age + pa * (4.25 * weight + 18.44 * height)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def load_mnd_data(
    path: str | PathLike[str],
    eer_func: Callable[[pd.Series], pd.Series],
    cache_dir: str | PathLike[str] | None = None,
    tuning: MndTuning | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process weight and calorie data from a MyNetDiary data export.

    Args:
        path: MyNetDiary export directory.
        eer_func: Callable that wraps eer_male or eer_female with height
            and dob fields specified by caller.
        cache_dir: Optional cache directory for the parquet cache.
        tuning: Optional tuning parameters. Defaults to ``MndTuning()``.

    Returns:
        Tuple containing processed weight and calorie metrics.
    """

    if tuning is None:
        tuning = MndTuning()

    # Load MyNetDiary data
    if cache_dir is not None:
        mnd_data = merge_excel_files_cached(path, cache_dir)
    else:
        mnd_data = merge_excel_files(path)

    # Tuned averaging controls
    weight_coverage = 0.50
    calorie_coverage = 0.67
    accuracy_coverage = 0.95

    weight_min_periods = utils.ewm_min_periods_from_halflife(
        tuning.weight_halflife,
        coverage=weight_coverage,
    )
    calorie_min_periods = utils.ewm_min_periods_from_halflife(
        tuning.calorie_halflife,
        coverage=calorie_coverage,
    )
    # Window matched to the effective span of the calorie EWM
    accuracy_window_days = utils.ewm_min_periods_from_halflife(
        tuning.calorie_halflife,
        coverage=accuracy_coverage,
    )

    rate_min_periods = max(2, tuning.rate_window_days // 2)
    accuracy_min_periods = max(2, accuracy_window_days // 2)

    # Construct a table of actual & smoothed weights
    weight = pd.DataFrame()
    bw = mnd_data["Measurements"].query('Measurement == "Body Weight"')
    unexpected_units = set(bw["Unit"].str.lower().unique()) - {"lbs"}
    if unexpected_units:
        raise ValueError(f"Unexpected body weight units: {unexpected_units}")
    weight["actual"] = bw.set_index("Date")["Value"].resample("D").mean()
    weight["smoothed"] = (
        weight["actual"]
        .ewm(
            halflife=tuning.weight_halflife,
            times=weight.index,
            min_periods=weight_min_periods,
        )
        .mean()
    )

    # Calculate weight gain/loss rate over time
    weight["rate"] = utils.rolling_linear_rate(
        weight["actual"], tuning.rate_window_days, rate_min_periods, "W"
    )

    # Construct a table of calorie information
    calories = pd.DataFrame()
    calories["food"] = (
        mnd_data["Food"]
        .resample("D", on="Date & Time")["Calories, cals"]
        .sum(min_count=1)
    )
    calories["exercise"] = (
        mnd_data["Exercise"].resample("D", on="Date & Time")["Calories"].sum()
    )
    calories["exercise"] = calories["exercise"].fillna(0)
    calories["baseline"] = eer_func(
        weight["smoothed"].reindex(calories.index, method="ffill")
    )
    calories.index.rename("date", inplace=True)

    # Impute unlogged days with the rolling average of logged days
    food_imputed = calories["food"].fillna(
        calories["food"]
        .ewm(
            halflife=tuning.calorie_halflife,
            times=calories.index,
            min_periods=calorie_min_periods,
        )
        .mean()
    )

    # Calculate rolling average of net calorie balance
    net_daily = food_imputed - calories["baseline"] - calories["exercise"]
    calories["net_recorded"] = net_daily.ewm(
        halflife=tuning.calorie_halflife,
        times=calories.index,
        min_periods=calorie_min_periods,
    ).mean()

    # Convert observed weight gain/loss in lbs/week to calories/day.
    calories["net_observed"] = (
        utils.CAL_PER_LB_WEEK
        * utils.rolling_linear_rate(
            weight["actual"],
            accuracy_window_days,
            accuracy_min_periods,
            "W",
            center=False,
        )
    )

    # Calculate "accuracy" of calorie counting relative to actual weight loss
    avg_food_recorded = food_imputed.ewm(
        halflife=tuning.calorie_halflife,
        times=calories.index,
        min_periods=calorie_min_periods,
    ).mean()
    avg_exercise = (
        calories["exercise"]
        .ewm(
            halflife=tuning.calorie_halflife,
            times=calories.index,
            min_periods=calorie_min_periods,
        )
        .mean()
    )
    avg_consumption_observed = (
        calories["baseline"] + avg_exercise + calories["net_observed"]
    )
    calories["accuracy"] = avg_food_recorded / avg_consumption_observed

    return weight, calories
