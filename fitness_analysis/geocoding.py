"""Reverse geocoding with persistent SQLite cache."""

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike

import geopy.distance
import sqlite_utils
from geopy import geocoders
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders.base import Geocoder

from . import cache_db, utils


@dataclass
class GeocodingConfig:
    """Configuration parameters for reverse geocoding.

    Attributes:
        match_radius_m: Distance threshold for cache hits, in metres. A cached
            address is returned if it was stored within this radius of the
            queried position.
        google_api_key_env: Name of the environment variable holding a
            Google Cloud API key with the Geocoding API enabled. When set,
            Google is used as the geocoding provider; otherwise Nominatim is
            used.
    """

    match_radius_m: float = 200.0
    google_api_key_env: str = "GOOGLE_CLOUD_API_KEY"


class GeocodingProvider:
    """Rate-limited geocoder wrapping a geopy backend."""

    def __init__(
        self, geocoder: Geocoder, min_delay_seconds: float, name: str
    ) -> None:
        self._geocode_fn = RateLimiter(
            geocoder.geocode, min_delay_seconds=min_delay_seconds
        )
        self._reverse_fn = RateLimiter(
            geocoder.reverse, min_delay_seconds=min_delay_seconds
        )
        self.name = name

    @classmethod
    def from_env(cls, google_api_key_env: str) -> GeocodingProvider:
        """Build a rate-limited geocoder from an environment variable.

        Uses Google Maps if the given environment variable contains a key; falls
        back to Nominatim otherwise.

        Args:
            google_api_key_env: Name of the environment variable holding a
                Google Cloud API key with the Geocoding API enabled.
        """
        google_api_key = os.getenv(google_api_key_env)
        if google_api_key:
            return cls(
                geocoders.GoogleV3(api_key=google_api_key),
                min_delay_seconds=0.04,
                name="google",
            )

        return cls(
            geocoders.Nominatim(user_agent="fitness-analysis"),
            min_delay_seconds=1,
            name="nominatim",
        )

    def geocode(self, address: str) -> tuple[float, float] | None:
        """Forward-geocode an address string to ``(lat, lon)``, or ``None``."""
        loc = self._geocode_fn(address, exactly_one=True)
        return (loc.latitude, loc.longitude) if loc is not None else None

    def reverse(self, lat: float, lon: float) -> str | None:
        """Reverse-geocode ``(lat, lon)`` to an address string, or ``None``."""
        loc = self._reverse_fn((lat, lon), language="en", exactly_one=True)
        return loc.address if loc is not None else None


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def invalidate_geocode_cache(
    cache_dir: str | PathLike[str] | None,
    provider: str | None = None,
) -> None:
    """Delete entries from the geocode cache.

    Also deletes all cluster fingerprints so the next call to
    ``cluster_routes_cached`` re-geocodes activity endpoints from the
    updated cache rather than serving stale addresses from the cluster
    cache.

    Args:
        cache_dir: Cache directory passed to ``geocode_positions``. If
            ``None``, this function does nothing.
        provider: If given, only entries from this provider are deleted
            (e.g. ``"nominatim"`` or ``"google"``). If ``None``, all
            entries are deleted.
    """
    if cache_dir is None or not cache_db.db_path(cache_dir).exists():
        return

    with cache_db.open_db(cache_dir) as db:
        if provider is None:
            with db.conn:
                db["geocode_cache"].drop()
                db["cluster_fingerprints"].delete_where()
        else:
            with db.conn:
                db["geocode_cache"].delete_where("provider = ?", [provider])
                db["cluster_fingerprints"].delete_where()


def seed_geocode_cache(
    addresses: Iterable[str],
    cache_dir: str | PathLike[str] | None,
    config: GeocodingConfig | None = None,
) -> None:
    """Forward-geocode address strings and store results in the cache.

    Forward-geocodes each address to coordinates, then reverse-geocodes
    those coordinates to store the provider's normalized display name.

    Args:
        addresses: Address strings to seed (e.g. ``["123 Main St, City"]``).
        cache_dir: Cache directory. If ``None``, this function does nothing.
        config: Geocoding configuration. Defaults to ``GeocodingConfig()``.
    """
    if cache_dir is None:
        return
    if config is None:
        config = GeocodingConfig()

    provider = GeocodingProvider.from_env(config.google_api_key_env)
    with cache_db.open_db(cache_dir) as db:
        for address in addresses:
            pos = provider.geocode(address)
            if pos is None:
                continue
            store_geocode_cache(db, pos, address, "seeded")


# ---------------------------------------------------------------------------
# Cache load / store
# ---------------------------------------------------------------------------


def round_pos(pos: tuple[float, float]) -> tuple[float, float]:
    """Round to 4 decimal places (~11 m precision) for stable cache keys."""
    return round(pos[0], 4), round(pos[1], 4)


def lookup_geocode_cache(
    db: sqlite_utils.Database,
    pos: tuple[float, float],
    match_radius_m: float,
) -> str | None:
    """Look up a reverse-geocoded address in the cache.

    First tries an exact match on the rounded coordinate key. If that misses,
    searches a bounding box and returns the nearest entry within
    ``match_radius_m``.

    Args:
        db: Open cache database.
        pos: ``(lat, lon)`` coordinate to look up.
        match_radius_m: Maximum distance in metres for a proximity hit.

    Returns:
        Cached address string, or ``None`` if no entry is within
        ``match_radius_m``.
    """
    row = db.conn.execute(
        "SELECT display_name FROM geocode_cache WHERE lat=? AND lon=?",
        round_pos(pos),
    ).fetchone()
    if row:
        return row[0]

    lat, lon = pos
    dlat = match_radius_m / utils.EARTH_M_PER_DEG
    dlon = match_radius_m / (
        utils.EARTH_M_PER_DEG * math.cos(math.radians(lat))
    )

    candidates = db.conn.execute(
        "SELECT lat, lon, display_name FROM geocode_cache"
        " WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
        (lat - dlat, lat + dlat, lon - dlon, lon + dlon),
    ).fetchall()
    if not candidates:
        return None

    best = min(
        candidates, key=lambda r: geopy.distance.great_circle(pos, r[:2]).m
    )
    if geopy.distance.great_circle(pos, best[:2]).m <= match_radius_m:
        return best[2]

    return None


def store_geocode_cache(
    db: sqlite_utils.Database,
    pos: tuple[float, float],
    display_name: str | None,
    provider_name: str,
) -> None:
    """Insert or replace an entry in the geocode cache.

    Args:
        db: Open cache database.
        pos: ``(lat, lon)`` coordinate being cached.
        display_name: Address string returned by the provider, or ``None`` if
            the provider returned no result.
        provider_name: Name of the geocoding provider (e.g. ``"nominatim"``).
    """
    with db.conn:
        db.conn.execute(
            "INSERT OR REPLACE INTO geocode_cache"
            " (lat, lon, display_name, provider)"
            " VALUES (?, ?, ?, ?)",
            (*round_pos(pos), display_name, provider_name),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def geocode_positions(
    positions: Iterable[tuple[float, float]],
    cache_dir: str | PathLike[str] | None,
    config: GeocodingConfig | None = None,
) -> dict[tuple[float, float], str | None]:
    """Reverse-geocode a collection of ``(lat, lon)`` positions.

    Deduplicates positions, serves addresses from the SQLite cache where
    available, and fetches from the configured provider for misses. When
    ``cache_dir`` is ``None`` every position is fetched from the provider
    directly with no caching.

    Args:
        positions: ``(lat, lon)`` pairs. Duplicates are resolved with a single
            lookup.
        cache_dir: Cache directory containing the SQLite database, or ``None``
            to disable caching.
        config: Geocoding configuration. Defaults to ``GeocodingConfig()``.

    Returns:
        Mapping from each input ``(lat, lon)`` pair to its address string,
        or ``None`` if no result was found.
    """
    if config is None:
        config = GeocodingConfig()
    unique = dict.fromkeys(positions)

    if cache_dir is None:
        provider = GeocodingProvider.from_env(config.google_api_key_env)
        for pos in unique:
            unique[pos] = provider.reverse(*pos)
        return unique

    with cache_db.open_db(cache_dir) as db:
        for pos in unique:
            unique[pos] = lookup_geocode_cache(db, pos, config.match_radius_m)

        misses = [pos for pos, addr in unique.items() if addr is None]
        if misses:
            provider = GeocodingProvider.from_env(config.google_api_key_env)
            for pos in misses:
                name = provider.reverse(*pos)
                store_geocode_cache(db, pos, name, provider.name)
                unique[pos] = name

    return unique
