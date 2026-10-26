"""Perfect-Foresight ceiling check and %-of-PF aggregation, shared by any
benchmark whose PF run is a true (or near-true) upper bound on any
forecast-driven run."""

from __future__ import annotations

import warnings
from typing import Sequence


class CeilingViolation(Exception):
    """A forecast-driven run scored strictly above its ceiling run by more
    than tolerance -- always a policy/constraint bug, never a legitimate
    result, when raised."""


def assert_ceiling(
    forecast_score: float,
    ceiling_score: float,
    mode: str,
    tol: float = 1e-6,
    context: str = "",
) -> bool:
    """Returns True if the ceiling held (within tol); False if violated and
    mode="soft" (a warning is also emitted). Raises CeilingViolation if
    violated and mode="hard".

    `mode` is per-benchmark (`Benchmark.ceiling_mode`): PV imbalance is
    "hard" (PF is a genuine mathematical ceiling there); a benchmark should
    only use "soft" after confirming its own policy can't beat its own PF
    run as a logical/structural matter -- see bess.multimarket's planned
    scope note and the brief's instruction to check this for BESS DAA
    specifically before deciding.
    """
    violated = forecast_score > ceiling_score + tol
    if not violated:
        return True
    msg = (
        f"Ceiling violated{': ' + context if context else ''}: "
        f"forecast score {forecast_score} > ceiling {ceiling_score} (tol={tol})"
    )
    if mode == "hard":
        raise CeilingViolation(msg)
    elif mode == "soft":
        warnings.warn(msg)
        return False
    else:
        raise ValueError(f"Unknown ceiling_mode: {mode!r} (expected 'hard' or 'soft')")


def percent_of_pf(numerators: Sequence[float], denominators: Sequence[float]) -> float | None:
    """Aggregate numerators and denominators across periods *before*
    dividing -- never average per-period ratios, which silently weights
    every period equally regardless of its actual scale.

    Returns None (not a nonsensical ratio) when the aggregated PF
    denominator is <= 0.
    """
    num_sum = sum(numerators)
    den_sum = sum(denominators)
    if den_sum <= 0:
        return None
    return num_sum / den_sum * 100
