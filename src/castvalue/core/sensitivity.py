"""Generic sensitivity-curve sweep driver.

Deliberately thin: core doesn't know how a benchmark's forecasts are
structured into named fields (that's `PLAYER_FIELDS`-style knowledge that
lives in each benchmark's own module), so the actual alpha-blended re-run
is the caller's `run_fn` closure. This module only owns the sweep
mechanics and the slope computation, so every benchmark computes "euro
slope near alpha=1" the same way.
"""

from __future__ import annotations

from typing import Callable

DEFAULT_ALPHAS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def sensitivity_curve(
    run_fn: Callable[[float], float],
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict:
    """Evaluate `run_fn(alpha)` -> score at each alpha and report the
    finite-difference slope near alpha=1 (the last two points), per the
    spec's "€-per-unit-error slopes near alpha=1" requirement.
    """
    if len(alphas) < 1:
        raise ValueError("sensitivity_curve needs at least one alpha")
    points = [{"alpha": a, "score": run_fn(a)} for a in alphas]
    slope_near_1 = None
    if len(points) >= 2:
        d_alpha = points[-1]["alpha"] - points[-2]["alpha"]
        if d_alpha != 0:
            slope_near_1 = (points[-1]["score"] - points[-2]["score"]) / d_alpha
    return {"points": points, "slope_near_1": slope_near_1}
