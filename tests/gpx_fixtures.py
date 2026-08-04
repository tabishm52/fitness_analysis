"""Synthetic GPX byte builders for testing.

Points carry Strava's bare ``<power>`` extension and ``gpxtpx:hr``, which is what
``ActivityParser`` normalizes to the ``power``/``heart_rate`` record columns.

Unlike activity_parser's checked-in fixture files, this module builds GPX
programmatically: fitness_analysis trusts activity_parser to parse correctly and only
needs parameterized data (arbitrary lat/lon, power, hr) to drive its own logic, not
coverage of XML-format edge cases.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

T0 = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)

# Anchor in San Francisco so infer_timezone resolves unambiguously to
# America/Los_Angeles; FAR is ~5 km east, for cases that need a distinct route.
SF = (37.7749, -122.4194)
FAR = (37.7749, -122.3600)

_GPX_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx creator="fitness_analysis tests" version="1.1"
    xmlns="http://www.topografix.com/GPX/1/1"
    xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v2">
  <metadata>
    <time>{start_time}</time>
  </metadata>
  <trk>
    <trkseg>
{trkpts}    </trkseg>
  </trk>
</gpx>
"""

_TRKPT_TEMPLATE = """\
      <trkpt lat="{lat:.6f}" lon="{lon:.6f}">
        <ele>10.0</ele>
        <time>{time}</time>
        <extensions>{power_el}
          <gpxtpx:TrackPointExtension>{hr_el}
          </gpxtpx:TrackPointExtension>
        </extensions>
      </trkpt>
"""


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def gpx_bytes(
    points: list[tuple[int, float, float]],
    hr: list[int] | None = None,
    power: list[int] | None = None,
) -> bytes:
    """Encodes a single-segment track into GPX bytes.

    Args:
        points: ``(offset_s, lat, lon)`` tuples, offsets relative to ``T0``.
        hr: Per-point heart rate (bpm), same length as ``points``, or ``None``.
        power: Per-point power (W), same length as ``points``, or ``None``.

    Returns:
        UTF-8 encoded GPX document bytes.
    """
    hrs = hr if hr is not None else [None] * len(points)
    powers = power if power is not None else [None] * len(points)

    trkpts = "".join(
        _TRKPT_TEMPLATE.format(
            lat=lat,
            lon=lon,
            time=_iso(T0 + timedelta(seconds=offset_s)),
            power_el=f"\n          <power>{p}</power>" if p is not None else "",
            hr_el=f"\n            <gpxtpx:hr>{h}</gpxtpx:hr>" if h is not None else "",
        )
        for (offset_s, lat, lon), h, p in zip(points, hrs, powers, strict=True)
    )

    return _GPX_TEMPLATE.format(start_time=_iso(T0), trkpts=trkpts).encode()


def line_route(
    start: tuple[float, float],
    end: tuple[float, float],
    n: int = 30,
) -> list[tuple[int, float, float]]:
    """``n`` evenly spaced points, 10 s apart, along a straight line."""
    lat0, lon0 = start
    lat1, lon1 = end
    return [
        (10 * i, lat0 + (lat1 - lat0) * i / (n - 1), lon0 + (lon1 - lon0) * i / (n - 1))
        for i in range(n)
    ]


def offset_route(
    points: list[tuple[int, float, float]],
    dlat: float = 0.0003,
) -> list[tuple[int, float, float]]:
    """A ~30 m jittered near-duplicate of ``points`` (inside the similarity floor)."""
    return [(offset_s, lat + dlat, lon) for offset_s, lat, lon in points]
