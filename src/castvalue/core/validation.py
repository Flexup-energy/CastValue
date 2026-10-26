"""Generic input validation shared by every benchmark.

Benchmark-specific dataclasses (e.g. pv.imbalance's DayForecasts/DayActuals)
call these rather than re-implementing length/NaN checks per benchmark.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

# Standard delivery-interval length across every benchmark in this package
# (15-minute market time units). Never hardcode a point count (96) anywhere
# that depends on this -- a day can legitimately be 92 or 100 intervals on
# the two annual DST transition days.
INTERVAL_HOURS = 0.25


def validate_equal_length(named_vectors: Mapping[str, Sequence[float]]) -> int:
    """Check every named vector has the same length. Returns that length.

    Raises ValueError naming every mismatched vector, not just the first
    one found, so a multi-vector mismatch doesn't require fixing and
    re-running one at a time.
    """
    if not named_vectors:
        raise ValueError("validate_equal_length called with no vectors")
    lengths = {name: len(vec) for name, vec in named_vectors.items()}
    distinct = set(lengths.values())
    if len(distinct) > 1:
        raise ValueError(f"Vector length mismatch: {lengths}")
    return distinct.pop()


def reject_missing(named_vectors: Mapping[str, Sequence[float]]) -> None:
    """Raise on the first NaN/None found, naming the vector and index.

    No silent interpolation here by design -- a single missing price must
    not silently poison a day's totals. A caller that wants gap-filling
    should do so explicitly, upstream of this check.
    """
    for name, vec in named_vectors.items():
        for i, value in enumerate(vec):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                raise ValueError(f"{name}[{i}] is missing (NaN/None)")
