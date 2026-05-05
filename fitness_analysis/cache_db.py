"""SQLite cache database shared by the activities and commutes caches."""

from collections.abc import Generator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd
import sqlite_utils

DB_FILE = "fitness_cache.db"


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


def to_sql(v: Any) -> Any:
    """Coerce a value to a sqlite3-safe scalar.

    Converts NaN to None and numpy scalars to their Python equivalents.
    """
    if pd.isna(v):
        return None

    return v.item() if hasattr(v, "item") else v


def segment_from_db(seg: int) -> int | None:
    """Translate a segment value from its SQLite representation to Python."""
    return None if seg == -1 else seg


def segment_to_db(seg: int | None) -> int:
    """Translate a segment value from Python to its SQLite representation."""
    return -1 if seg is None or pd.isna(seg) else int(seg)


def cache_key(fn: str | float, seg: int | None) -> tuple[str, int] | None:
    """Build a ``(filename, segment)`` DB primary key.

    Uses -1 as a sentinel for whole-file (None) segments so keys are hashable
    and map directly to the ``segment`` column in both DB tables.

    Args:
        fn: Activity filename, or NaN for fileless activities.
        seg: Segment index, or ``None`` for single-segment activities.

    Returns:
        ``(filename, segment)`` tuple, or ``None`` if ``fn`` is NaN.
    """
    if pd.isna(fn):
        return None

    return (fn, segment_to_db(seg))


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def db_path(cache_dir: str | PathLike[str]) -> Path:
    """Return the path to the cache database file."""
    return Path(cache_dir) / DB_FILE


@contextmanager
def open_db(
    cache_dir: str | PathLike[str],
) -> Generator[sqlite_utils.Database]:
    """Open (or create) the cache DB, ensure tables exist, and yield it.

    The connection is closed on exit even if an exception is raised. Use
    ``with db.conn:`` around raw ``db.conn.execute()`` calls that must commit
    atomically; sqlite-utils methods manage their own transactions internally.
    """
    db = sqlite_utils.Database(db_path(cache_dir))
    try:
        ensure_tables(db)
        yield db
    finally:
        db.close()


def ensure_tables(db: sqlite_utils.Database) -> None:
    """Create cache tables if they do not already exist."""
    db.conn.executescript("""
        CREATE TABLE IF NOT EXISTS activities (
            filename       TEXT NOT NULL,
            segment        INTEGER NOT NULL,
            timezone       TEXT,
            has_location   INTEGER,
            max_heart_rate REAL,
            estimated_ftp  REAL,
            cluster_id     REAL,
            cluster_name   TEXT,
            start_lat      REAL,
            start_lon      REAL,
            end_lat        REAL,
            end_lon        REAL,
            start_address  TEXT,
            end_address    TEXT,
            PRIMARY KEY (filename, segment)
        );
        CREATE TABLE IF NOT EXISTS commutes (
            filename       TEXT NOT NULL,
            segment        INTEGER NOT NULL,
            date           TEXT,
            description    TEXT,
            direction      TEXT,
            distance       REAL,
            elapsed_time_s REAL,
            moving_time_s  REAL,
            cluster_id     REAL,
            cluster_name   TEXT,
            start_lat      REAL,
            start_lon      REAL,
            end_lat        REAL,
            end_lon        REAL,
            start_address  TEXT,
            end_address    TEXT,
            PRIMARY KEY (filename, segment)
        );
        CREATE TABLE IF NOT EXISTS cluster_fingerprints (
            table_name  TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS geocode_cache (
            lat          REAL NOT NULL,
            lon          REAL NOT NULL,
            display_name TEXT,
            provider     TEXT,
            PRIMARY KEY (lat, lon)
        );
    """)
