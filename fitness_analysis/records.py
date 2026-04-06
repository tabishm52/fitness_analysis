"""Parsing and caching of activity record files (FIT/TCX/GPX)."""

import itertools
import shutil
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
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
# Parquet cache helpers
# ---------------------------------------------------------------------------


def parquet_path(
    filename: str | PathLike[str],
    segment: int | None,
    cache_dir: str | PathLike[str],
) -> Path:
    """Return the parquet cache path for an activity file or segment.

    Strips any trailing ``.gz`` wrapper so the parquet name always ends with
    the underlying format extension (e.g. ``.fit.parquet``). When ``segment``
    is given, inserts ``-{segment}`` before the extension to produce a unique
    path for each segment of a multi-segment activity.

    Args:
        filename: Activity filename (relative path or bare name).
        segment: 1-based segment index. ``None`` for whole-file activities.
        cache_dir: Cache directory.

    Returns:
        Path to the parquet file, e.g.
        ``{cache_dir}/activity_records/1234567890.fit.parquet`` or
        ``{cache_dir}/activity_records/1234567890-1.fit.parquet``.
    """

    name = Path(filename).name
    base = name[:-3] if name.endswith(".gz") else name

    stem, ext = Path(base).stem, Path(base).suffix
    key = f"{stem}-{segment}{ext}" if segment is not None else base

    return Path(cache_dir) / RECORDS_CACHE_DIR / (key + ".parquet")


def cache_record(
    records_df: pd.DataFrame,
    filename: str | PathLike[str],
    segment: int | None,
    cache_dir: str | PathLike[str],
) -> None:
    """Write activity records to the parquet cache.

    Coerces object / mixed-type columns to string for pyarrow compatibility.

    Args:
        records_df: Activity records to cache.
        filename: Activity filename used to derive the cache path.
        segment: 1-based segment index. ``None`` for whole-file activities.
        cache_dir: Cache directory.
    """

    path = parquet_path(filename, segment, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    obj_cols = records_df.select_dtypes(include="object").columns
    coerced = records_df.astype({col: "string" for col in obj_cols})

    coerced.to_parquet(path)


# ---------------------------------------------------------------------------
# Single file parsing
# ---------------------------------------------------------------------------


def parse_record_cached(
    filename: str | PathLike[str],
    segment: int | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Parse a FIT/TCX/GPX file, using a Parquet cache when available.

    Worker-safe: picklable and suitable for use inside multiprocessing pools.

    Activity files are assumed immutable after Strava export. To invalidate,
    use ``invalidate_records_cache``.

    Args:
        filename: Activity filename (relative to ``path``).
        segment: 1-based segment index. ``None`` for whole-file activities.
        path: Directory containing the activity file.
        cache_dir: Optional cache directory.

    Returns:
        Parsed records.
    """

    full_path = Path(path) / filename

    if segment is not None and cache_dir is None:
        raise ValueError(
            f"cache_dir is required to retrieve segment {segment} of "
            f"{Path(filename).name}; segment parquets cannot be stored "
            "or retrieved without a cache directory."
        )

    if cache_dir is None:
        records, _, _ = parser.parse(full_path)
        return records

    p = parquet_path(filename, segment, cache_dir)
    if p.exists():
        return pd.read_parquet(p)

    if segment is not None:
        raise FileNotFoundError(
            f"No cached records for segment {segment} of "
            f"{Path(filename).name}. Segment parquets must be written "
            "explicitly via cache_record."
        )

    records, _, _ = parser.parse(full_path)
    cache_record(records, filename, segment, cache_dir)

    return records


def parse_coords_cached(
    filename: str | PathLike[str],
    segment: int | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame | None:
    """Parse a FIT/TCX/GPX file and return trimmed lat/lon columns.

    Worker-safe: picklable and suitable for use inside multiprocessing pools.

    Args:
        filename: Activity filename (relative to ``path``).
        segment: 1-based segment index. ``None`` for whole-file activities.
        path: Directory containing the activity file.
        cache_dir: Optional cache directory.

    Returns:
        Trimmed ``latitude``/``longitude`` data, or ``None`` if GPS data are
        absent.
    """

    rec = parse_record_cached(filename, segment, path, cache_dir)

    if "latitude" not in rec.columns:
        return None

    lat = rec["latitude"]
    first = lat.first_valid_index()
    if first is None:
        return None

    last = lat.last_valid_index()
    return rec.loc[first:last, ["latitude", "longitude"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _record_args(
    files: Iterable[str | PathLike[str]],
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None,
) -> Iterator[tuple]:
    """Yield ``(filename, segment, path, cache_dir)`` tuples for a batch.

    When ``segments`` is ``None``, every file is treated as a whole-file
    activity (``segment=None``).
    """

    segs = segments if segments is not None else itertools.repeat(None)
    for f, seg in zip(files, segs):
        normalized = None if seg is None or pd.isna(seg) else int(seg)
        yield (f, normalized, path, cache_dir)


def _parse_record_cached_packed(args: tuple) -> pd.DataFrame:
    """Picklable single-argument wrapper around ``parse_record_cached``."""

    return parse_record_cached(*args)


def _parse_coords_cached_packed(args: tuple) -> pd.DataFrame | None:
    """Picklable single-argument wrapper around ``parse_coords_cached``."""

    return parse_coords_cached(*args)


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


def warm_records_cache(
    files: Iterable[str | PathLike[str]],
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str],
) -> None:
    """Ensure parquet files exist for all given activity files.

    Identifies whole-file (``segment=None``) entries without a parquet cache
    and parses them, pooling workers when there are enough cold files to
    justify the overhead. Segment entries (``segment`` is not ``None``) are
    skipped — their parquets must be written explicitly via ``cache_record``.

    Args:
        files: Activity filenames (relative to ``path``).
        segments: Per-file segment indices, or ``None`` to treat all files as
            whole-file activities.
        path: Directory containing the activity files.
        cache_dir: Cache directory containing the parquet subdirectory.
    """

    cold_args = [
        args
        for args in _record_args(files, segments, path, cache_dir)
        if args[1] is None
        and not parquet_path(args[0], None, cache_dir).exists()
    ]
    if not cold_args:
        return

    if len(cold_args) >= PARALLEL_MIN_FILES:
        with ProcessPoolExecutor() as ex:
            list(ex.map(_parse_record_cached_packed, cold_args))
    else:
        for args in cold_args:
            parse_record_cached(*args)


def invalidate_records_cache(
    files: Iterable[str | PathLike[str]] | None,
    segments: Iterable[int | None] | None,
    cache_dir: str | PathLike[str],
) -> None:
    """Invalidate the records (parquet) cache.

    If ``files`` is ``None``, deletes the entire cache directory. Otherwise
    removes only the parquet files for the given activity filenames and their
    corresponding segment indices.

    Args:
        files: Activity filenames whose parquet files should be removed.
            If ``None``, the entire records cache is cleared.
        segments: Per-file segment indices, or ``None`` to treat all files as
            whole-file activities.
        cache_dir: Cache directory passed to ``load_activity_records`` or
            ``load_strava_activities``.
    """

    parquet_dir = Path(cache_dir) / RECORDS_CACHE_DIR
    if not parquet_dir.exists():
        return

    if files is None:
        shutil.rmtree(parquet_dir)
        return

    segs = segments if segments is not None else itertools.repeat(None)
    for f, seg in zip(files, segs):
        p = parquet_path(f, seg, cache_dir)
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def load_activity_records(
    files: pd.Series,
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> list[pd.DataFrame]:
    """Load parsed records for a set of activity files.

    Parses each file using the records parquet cache when available.

    Args:
        files: Activity filenames (relative to ``path``).
        segments: Per-file segment indices, or ``None`` to treat all files as
            whole-file activities.
        path: Directory containing the activity files.
        cache_dir: Optional cache directory.

    Returns:
        Parsed records, one per file, in the same order as ``files``.
    """

    args = list(_record_args(files, segments, path, cache_dir))

    if cache_dir is None:
        with ProcessPoolExecutor() as ex:
            return list(ex.map(_parse_record_cached_packed, args))

    # Pass 1: pool cold whole-file parquets
    warm_records_cache(files, segments, path, cache_dir)

    # Pass 2: read all from parquet serially
    return [parse_record_cached(*a) for a in args]


def load_activity_coords(
    files: pd.Series,
    segments: Iterable[int | None] | None,
    path: str | PathLike[str],
    cache_dir: str | PathLike[str] | None = None,
) -> list[pd.DataFrame | None]:
    """Load trimmed lat/lon data for a set of activity files.

    Parses each file using the records parquet cache when available.

    Args:
        files: Activity filenames (relative to ``path``).
        segments: Per-file segment indices, or ``None`` to treat all files as
            whole-file activities.
        path: Directory containing the activity files.
        cache_dir: Optional cache directory.

    Returns:
        Trimmed ``latitude``/``longitude`` data, one per file, in the same
        order as ``files``. ``None`` for files with no GPS data.
    """

    args = list(_record_args(files, segments, path, cache_dir))

    if cache_dir is None:
        with ProcessPoolExecutor() as ex:
            return list(ex.map(_parse_coords_cached_packed, args))

    # Pass 1: pool cold whole-file parquets
    warm_records_cache(files, segments, path, cache_dir)

    # Pass 2: read all from parquet serially
    return [parse_coords_cached(*a) for a in args]
