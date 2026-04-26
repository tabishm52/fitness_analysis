"""SQLite cache database shared by the activities and commutes caches."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

DB_FILE = "fitness_cache.db"


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


def to_sql(v: Any) -> Any:
    """Coerce a value to a sqlite3-safe scalar.

    Converts NaN to None and numpy scalars to their Python equivalents.

    Args:
        v: Value to coerce.

    Returns:
        sqlite3-safe scalar, or ``None`` if ``v`` is NaN.
    """

    if pd.isna(v):
        return None

    return v.item() if hasattr(v, "item") else v


def segment_from_db(seg: int) -> int | None:
    """Translate a segment value from its SQLite representation to Python.

    Args:
        seg: Segment integer read from the database.

    Returns:
        ``None`` for whole-file activities, otherwise the segment index.
    """

    return None if seg == -1 else seg


def segment_to_db(seg: int | None) -> int:
    """Translate a segment value from Python to its SQLite representation.

    Args:
        seg: Segment index, or ``None`` / NaN for single-segment activities.

    Returns:
        -1 for whole-file activities, otherwise the segment index as an ``int``.
    """

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
    """Return the path to the cache database file.

    Args:
        cache_dir: Directory containing the cache database.

    Returns:
        Path to the cache database file within ``cache_dir``.
    """

    return Path(cache_dir) / DB_FILE


@contextmanager
def open_db(
    cache_dir: str | PathLike[str],
) -> Generator[sqlite3.Connection]:
    """Open (or create) the cache DB, ensure tables exist, and yield connection.

    Uses ``autocommit=False``. Callers should wrap write operations in a
    ``with conn:`` block to commit them as a transaction. The connection is
    closed on exit even if an exception is raised.

    Args:
        cache_dir: Directory containing the cache database.

    Yields:
        Open ``sqlite3.Connection``.
    """

    conn = sqlite3.connect(db_path(cache_dir), autocommit=False)
    try:
        ensure_tables(conn)
        yield conn
    finally:
        conn.close()


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create cache tables if they do not already exist.

    Args:
        conn: Open database connection.
    """

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activities (
            filename       TEXT NOT NULL,
            segment        INTEGER NOT NULL,
            timezone       TEXT,
            has_location   INTEGER,
            max_heart_rate REAL,
            estimated_ftp  REAL,
            cluster_id     REAL,
            cluster_name   TEXT,
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
            PRIMARY KEY (filename, segment)
        );
        CREATE TABLE IF NOT EXISTS cluster_fingerprints (
            table_name  TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL
        );
    """)
