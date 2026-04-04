"""Parsing and caching of activity record files (FIT/TCX/GPX)."""

import shutil
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import PathLike
from pathlib import Path

import activity_parser
import pandas as pd

RECORDS_CACHE_DIR = "activity_records"

# Minimum cold files to justify ProcessPoolExecutor startup overhead.
PARALLEL_MIN_FILES = 30

# Global parser shared across the fitness_analysis module
parser = activity_parser.ActivityParser()


# ---------------------------------------------------------------------------
# Single file parsing
# ---------------------------------------------------------------------------


def parse_record_cached(
    filename: str | PathLike[str],
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Parse a FIT/TCX/GPX file, using a Parquet cache when available.

    Worker-safe: picklable and suitable for use inside multiprocessing pools.

    The cache key is the filename only — activity files are assumed immutable
    after Strava export. To invalidate, use ``invalidate_records_cache``.

    Args:
        filename: Activity filename (relative to ``path``).
        path: Directory containing the activity file.
        cache_dir: Optional cache directory.

    Returns:
        Parsed records.
    """

    full_path = Path(path) / filename

    if cache_dir is None:
        records, _, _ = parser.parse(full_path)
        return records

    parquet_path = (
        Path(cache_dir) / RECORDS_CACHE_DIR / (Path(filename).name + ".parquet")
    )
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    records, _, _ = parser.parse(full_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Coerce object-dtype columns to string — mixed-type columns cannot be
    # serialized by pyarrow without a consistent type.
    obj_cols = records.select_dtypes(include="object").columns
    records.astype({col: "string" for col in obj_cols}).to_parquet(parquet_path)

    return records


def parse_coords_cached(
    filename: str | PathLike[str],
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame | None:
    """Parse a FIT/TCX/GPX file and return trimmed lat/lon columns.

    Worker-safe: picklable and suitable for use inside multiprocessing pools.

    Args:
        filename: Activity filename (relative to ``path``).
        path: Directory containing the activity file.
        cache_dir: Optional cache directory.

    Returns:
        Trimmed ``latitude``/``longitude`` data, or ``None`` if GPS data are
        absent.
    """

    rec = parse_record_cached(filename, path, cache_dir)

    if "latitude" not in rec.columns:
        return None

    lat = rec["latitude"]
    first = lat.first_valid_index()
    if first is None:
        return None

    last = lat.last_valid_index()
    return rec.loc[first:last, ["latitude", "longitude"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


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

    fn = partial(parse_record_cached, path=path, cache_dir=cache_dir)
    if len(cold) >= PARALLEL_MIN_FILES:
        with ProcessPoolExecutor() as ex:
            list(ex.map(fn, cold))
    else:
        for f in cold:
            parse_record_cached(f, path, cache_dir)


def invalidate_records_cache(
    files: Iterable[str | PathLike[str]] | None,
    cache_dir: str | PathLike[str],
) -> None:
    """Invalidate the records (parquet) cache.

    If ``files`` is None, deletes the entire cache directory. Otherwise
    removes only the parquet files for the given activity filenames.

    Args:
        files: Activity filenames whose parquet files should be removed.
            If None, the entire records cache is cleared.
        cache_dir: Cache directory passed to ``load_activity_records`` or
            ``load_strava_activities``.
    """

    parquet_dir = Path(cache_dir) / RECORDS_CACHE_DIR
    if not parquet_dir.exists():
        return
    if files is None:
        shutil.rmtree(parquet_dir)
        return
    for f in files:
        parquet_path = parquet_dir / (Path(f).name + ".parquet")
        if parquet_path.exists():
            parquet_path.unlink()


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def load_activity_records(
    files: pd.Series,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> list[pd.DataFrame]:
    """Load parsed records for a set of activity files.

    Parses each file using the records parquet cache when available.

    Args:
        files: Activity filenames (relative to ``path``).
        path: Directory containing the activity files.
        cache_dir: Optional cache directory.

    Returns:
        Parsed records, one per file, in the same order as ``files``.
    """

    if cache_dir is None:
        with ProcessPoolExecutor() as ex:
            return list(ex.map(partial(parse_record_cached, path=path), files))

    # Pass 1: pool cold files (raw FIT parse → write parquet)
    warm_records_cache(files, path, cache_dir)

    # Pass 2: read all from parquet serially
    return [parse_record_cached(f, path, cache_dir) for f in files]


def load_activity_coords(
    files: pd.Series,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> list[pd.DataFrame | None]:
    """Load trimmed lat/lon data for a set of activity files.

    Parses each file using the records parquet cache when available.

    Args:
        files: Activity filenames (relative to ``path``).
        path: Directory containing the activity files.
        cache_dir: Optional cache directory.

    Returns:
        Trimmed ``latitude``/``longitude`` data, one per file, in the same
        order as ``files``. ``None`` for files with no GPS data.
    """

    if cache_dir is None:
        with ProcessPoolExecutor() as ex:
            return list(ex.map(partial(parse_coords_cached, path=path), files))

    # Pass 1: pool cold files (raw FIT parse → write parquet)
    warm_records_cache(files, path, cache_dir)

    # Pass 2: read all from parquet serially
    return [parse_coords_cached(f, path, cache_dir) for f in files]
