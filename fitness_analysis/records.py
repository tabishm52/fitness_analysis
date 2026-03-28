"""Parsing and caching of activity record files (FIT/TCX/GPX)."""

import multiprocessing
import shutil
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import activity_parser
import pandas as pd

RECORDS_CACHE_DIR = "activity_records"

# Global parser shared across the fitness_analysis module
parser = activity_parser.ActivityParser()


def parse_record_cached(
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Parse a FIT/TCX/GPX file, using a Parquet cache when available.

    Worker-safe: picklable and suitable for use inside multiprocessing pools.

    The cache key is the filename only — activity files are assumed immutable
    after Strava export. To invalidate, use ``invalidate_records_cache``.

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


def _ensure_record_cached(
    path: str | PathLike[str],
    cache_dir: str | PathLike[str],
) -> None:
    """Pool worker wrapper around ``parse_record_cached``.

    Returns None instead of the DataFrame so workers avoid pickling large
    results back over IPC. Picklable and safe for use in multiprocessing pools.
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
            p.starmap(_ensure_record_cached, args)
    else:
        for a in args:
            _ensure_record_cached(*a)


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
