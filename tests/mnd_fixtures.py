"""Synthetic MyNetDiary export builders for testing.

Builds only ``.xlsx``, via openpyxl. The ``.xls`` read path is tested separately by the
checked-in ``tests/files/mnd_export.xls``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path

import pandas as pd

T0 = datetime(2026, 1, 5)


def _default_measurements(n_days: int, start: datetime, start_weight: float) -> pd.DataFrame:
    """``n_days`` daily weights on a straight -1 lb/week line (a known-answer rate)."""
    return pd.DataFrame(
        {
            "Measurement": "Body Weight",
            "Date": [start + timedelta(days=i) for i in range(n_days)],
            "Value": [start_weight - i * (1.0 / 7.0) for i in range(n_days)],
            "Unit": "lbs",
        }
    )


def _default_food(n_days: int, start: datetime, gap_every: int = 7) -> pd.DataFrame:
    """~2000 kcal/day, with a gap day (no logged row) every ``gap_every`` days."""
    return pd.DataFrame(
        {"Date & Time": start + timedelta(days=i), "Calories, cals": 2000.0}
        for i in range(n_days)
        if (i + 1) % gap_every != 0
    )


def _default_exercise(n_days: int, start: datetime, every: int = 4) -> pd.DataFrame:
    """A sparse exercise log: one row every ``every`` days."""
    return pd.DataFrame(
        {"Date & Time": start + timedelta(days=i), "Calories": 400.0}
        for i in range(0, n_days, every)
    )


def write_mnd_export(
    export_dir: str | PathLike[str],
    *,
    measurements: pd.DataFrame | None = None,
    food: pd.DataFrame | None = None,
    exercise: pd.DataFrame | None = None,
    filename: str = "export-1.xlsx",
    start: datetime = T0,
    n_days: int = 30,
    start_weight: float = 180.0,
) -> Path:
    """Writes a synthetic MyNetDiary xlsx export: Measurements/Food/Exercise sheets.

    Defaults: ``n_days`` daily weights on a -1 lb/week line, ~2000 kcal/day food with
    weekly gap days, and sparse exercise rows.

    Args:
        export_dir: Directory to write the export into.
        measurements: Overrides the default Measurements sheet.
        food: Overrides the default Food sheet.
        exercise: Overrides the default Exercise sheet.
        filename: Export filename.
        start: First day of the default series.
        n_days: Number of days in the default series.
        start_weight: Starting weight (lbs) for the default measurements series.

    Returns:
        Path to the written file.
    """
    if measurements is None:
        measurements = _default_measurements(n_days, start, start_weight)
    if food is None:
        food = _default_food(n_days, start)
    if exercise is None:
        exercise = _default_exercise(n_days, start)

    path = Path(export_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        measurements.to_excel(writer, sheet_name="Measurements", index=False)
        food.to_excel(writer, sheet_name="Food", index=False)
        exercise.to_excel(writer, sheet_name="Exercise", index=False)

    return path
