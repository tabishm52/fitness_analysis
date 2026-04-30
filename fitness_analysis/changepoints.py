"""Changepoint-based span detection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import ruptures as rpt


@dataclass
class SpanConfig:
    """Configuration for PELT changepoint detection.

    Attributes:
        pen: Penalty parameter. Higher values produce fewer spans.
        min_size: Minimum observations per span.
    """

    pen: float = 100.0
    min_size: int = 2


def detect_spans(
    signal: np.ndarray,
    index: pd.DatetimeIndex,
    config: SpanConfig,
) -> pd.DataFrame:
    """Detect changepoints in a pre-scaled signal using PELT.

    Args:
        signal: (N, D) array of already-scaled features, one row per
            observation.
        index: Timestamps aligned to signal rows.
        config: PELT parameters.

    Returns:
        One row per detected span with ``start`` and ``end`` columns.
        Empty if the signal has zero rows.
    """
    if len(signal) == 0:
        return pd.DataFrame(columns=["start", "end"])

    breakpoints = (
        rpt.Pelt(model="l2", min_size=config.min_size, jump=1)
        .fit(signal)
        .predict(pen=config.pen)
    )

    starts = [index[0]] + [index[bp] for bp in breakpoints[:-1]]
    ends = [index[bp - 1] for bp in breakpoints[:-1]] + [index[-1]]
    return pd.DataFrame({"start": starts, "end": ends})
