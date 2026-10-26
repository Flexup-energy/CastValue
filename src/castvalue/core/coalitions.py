"""Generic 2^n coalition runner, working against any object satisfying the
Benchmark protocol -- this is what makes PF "just the full coalition" and
every Shapley/Harsanyi computation benchmark-agnostic.
"""

from __future__ import annotations

import itertools
from typing import Any

from castvalue.core.benchmark import Benchmark, BaselineType, DayResult
from castvalue.core.config import PolicyConfig


def all_coalitions(players: tuple[str, ...]) -> list[frozenset[str]]:
    """Every subset of `players`, including the empty set and the full set
    -- 2^len(players) of them, in a deterministic (size-then-lexical) order."""
    result: list[frozenset[str]] = []
    for r in range(len(players) + 1):
        for combo in itertools.combinations(players, r):
            result.append(frozenset(combo))
    return result


def run_all_coalitions(
    benchmark: Benchmark,
    forecasts: Any,
    actuals: Any,
    config: PolicyConfig,
    baseline: BaselineType = "perfect",
    players: tuple[str, ...] | None = None,
) -> dict[frozenset[str], DayResult]:
    """Run `benchmark.run_day` once per coalition subset of `players`
    (defaults to `benchmark.players`).

    The empty coalition reproduces the plain forecast-driven run exactly
    (no player substituted); the full-player coalition reproduces Perfect
    Foresight exactly (baseline="perfect") or the naive-baseline run
    (baseline="naive") -- by construction, since both are literally the
    same `run_day` call other players also go through.
    """
    players = players if players is not None else benchmark.players
    results: dict[frozenset[str], DayResult] = {}
    for coalition in all_coalitions(players):
        substituted = benchmark.substitute(forecasts, actuals, coalition, baseline)
        results[coalition] = benchmark.run_day(substituted, actuals, config)
    return results
