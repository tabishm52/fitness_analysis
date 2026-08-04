"""Synthetic Strava export builders for testing.

Builds a minimal ``activities.csv`` + ``activities/`` directory matching a real Strava
bulk-export, so ``load_strava_activities`` can be exercised without any personal data.
"""

from __future__ import annotations

import math
from datetime import datetime
from os import PathLike
from pathlib import Path

import pandas as pd

# Strava's export date format, hardcoded at strava.py's `load_strava_activities_raw`.
STRAVA_DATE_FORMAT = "%b %d, %Y, %I:%M:%S %p"

CSV_COLUMNS = [
    "Activity Date",
    "Activity Type",
    "Activity Name",
    "Activity Gear",
    "Commute",
    "Distance",
    "Elevation Gain",
    "Elapsed Time",
    "Moving Time",
    "Filename",
]


def activity_row(
    date: datetime,
    *,
    activity_type: str = "Ride",
    name: str = "Morning Ride",
    gear: str = "Road Bike",
    commute: bool = False,
    distance_km: float = 10.0,
    elevation_m: float = 100.0,
    elapsed_s: int = 1800,
    moving_s: int = 1700,
    filename: str | None = None,
) -> dict:
    """Builds one row of a Strava ``activities.csv`` export.

    Args:
        date: Activity start date/time; formatted with ``STRAVA_DATE_FORMAT``.
        activity_type: ``"Ride"`` or ``"Virtual Ride"`` (other types are filtered out
            by ``load_strava_activities_raw``).
        name: Activity name.
        gear: Bike/gear name.
        commute: Whether the activity is flagged as a commute.
        distance_km: Distance in km.
        elevation_m: Elevation gain in meters.
        elapsed_s: Elapsed time in seconds.
        moving_s: Moving time in seconds.
        filename: Activity filename relative to the export directory (e.g.
            ``"activities/ride1.gpx"``), or ``None`` for a fileless (manually-logged)
            activity.

    Returns:
        A dict with the export's exact column names.
    """
    return {
        "Activity Date": date.strftime(STRAVA_DATE_FORMAT),
        "Activity Type": activity_type,
        "Activity Name": name,
        "Activity Gear": gear,
        "Commute": commute,
        "Distance": distance_km,
        "Elevation Gain": elevation_m,
        "Elapsed Time": elapsed_s,
        "Moving Time": moving_s,
        "Filename": filename if filename is not None else math.nan,
    }


def write_export(
    export_dir: str | PathLike[str],
    rows: list[dict],
    files: dict[str, bytes],
) -> None:
    """Writes a synthetic Strava export directory: ``activities.csv`` + activity files.

    Args:
        export_dir: Directory to write the export into.
        rows: Rows as built by ``activity_row``.
        files: Mapping of activity filename (as referenced in ``rows``, e.g.
            ``"activities/ride1.gpx"``) to its file contents.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(export_dir / "activities.csv", index=False)

    for name, data in files.items():
        path = export_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
