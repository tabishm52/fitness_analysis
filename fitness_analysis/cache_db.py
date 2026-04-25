"""SQLite cache database shared by the activities and commutes caches."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

DB_FILE = "fitness_cache.db"


def to_sql(v: Any) -> Any:
    """Coerce a value into a sqlite3-safe Python scalar.

    Returns None for NaN values (sqlite3 rejects NaN floats) and calls
    ``.item()`` on numpy scalars so sqlite3 does not fall back to storing
    them as BLOBs.

    Args:
        v: Value to coerce.

    Returns:
        A sqlite3-safe Python scalar, or None if v is NaN.
    """

    if pd.isna(v):
        return None

    return v.item() if hasattr(v, "item") else v


def cache_key(fn: str | float, seg: int | None) -> tuple[str, int] | None:
    """Return ``(fn, int(seg))`` or ``None`` if fn is NaN.

    Uses -1 as a sentinel for whole-file (None) segments so keys are hashable
    and map directly to the ``segment`` column in both DB tables.

    Args:
        fn: Activity filename, or ``float`` NaN for activities without a file.
        seg: Segment index, or ``None`` for whole-file (single-segment)
            activities.

    Returns:
        ``(filename, segment_int)`` tuple, or ``None`` if ``fn`` is NaN.
    """

    if pd.isna(fn):
        return None

    return (fn, -1 if seg is None or pd.isna(seg) else int(seg))


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

    Uses ``autocommit=True`` — callers do not need to commit. The connection is
    closed on exit even if an exception is raised.

    Args:
        cache_dir: Directory containing the cache database.

    Yields:
        Open ``sqlite3.Connection`` with autocommit enabled.
    """

    conn = sqlite3.connect(db_path(cache_dir), autocommit=True)
    ensure_tables(conn)
    try:
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
    """)
