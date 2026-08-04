"""Synthetic FIT byte builders for testing.

Encodes a tiny FIT message stream via the garmin-fit-sdk ``Encoder``. GPS/power/
heart-rate coverage already comes from ``gpx_fixtures``; these builders exist to pin
the FIT/``.fit.gz`` path itself.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from garmin_fit_sdk import Encoder

RECORD_MESG_NUM = 20

T0 = datetime(2026, 1, 5, 8, 0, 0, tzinfo=UTC)


def at(seconds: int) -> datetime:
    """Returns the timestamp `seconds` after the fixtures' shared start time."""
    return T0 + timedelta(seconds=seconds)


def encode(messages: list[dict[str, Any]]) -> bytes:
    """Encodes a list of FIT messages into file bytes."""
    encoder = Encoder()
    for message in messages:
        encoder.write_mesg(message)
    return encoder.close()


def simple_ride(n: int = 5, hr_start: int = 100) -> bytes:
    """``n`` records, 1 s apart, with heart rate counting up from ``hr_start``."""
    return encode(
        [
            {"mesg_num": RECORD_MESG_NUM, "timestamp": at(i), "heart_rate": hr_start + i}
            for i in range(n)
        ]
    )
