"""The shared Benchmark protocol every benchmark (BESS DAA, PV imbalance,
future BESS multi-market) implements an adapter for.

Core code only ever talks to a benchmark through this interface -- it
never assumes per-interval separability (PV solves per interval; BESS is a
genuine multi-period LP since state of charge couples intervals).
`run_day()` is opaque to core either way.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Protocol

from castvalue import __version__ as CASTVALUE_VERSION
from castvalue.core.config import PolicyConfig, config_hash

BaselineType = Literal["perfect", "naive"]
CeilingMode = Literal["hard", "soft"]


@dataclasses.dataclass(frozen=True)
class DayResult:
    """One benchmark's output for one day (or one coalition run on one day).

    score: the scalar objective used for Shapley attribution and the
        ceiling check (risk-adjusted net revenue for PV, margin for BESS).
    raw: per-interval vectors and totals, benchmark-specific shape, meant
        for JSONB storage/plotting -- core never inspects this.
    provenance: see make_provenance() below.
    """

    score: float
    raw: dict[str, Any]
    provenance: dict[str, Any]


def make_provenance(benchmark: "Benchmark", config: PolicyConfig) -> dict[str, Any]:
    """Standard provenance block every DayResult should carry, so a stored
    result is always traceable to the exact method version, package
    version, and parameters that produced it."""
    return {
        "benchmark": benchmark.name,
        "method_version": benchmark.METHOD_VERSION,
        "castvalue_version": CASTVALUE_VERSION,
        "config_hash": config_hash(config),
    }


class Benchmark(Protocol):
    """Structural protocol -- an adapter satisfies this by having the right
    attributes/methods, no inheritance required."""

    name: str
    METHOD_VERSION: str
    players: tuple[str, ...]
    groups: dict[str, tuple[str, ...]] | None
    ceiling_mode: CeilingMode

    def substitute(
        self,
        forecasts: Any,
        actuals: Any,
        coalition: frozenset[str],
        baseline: BaselineType,
    ) -> Any:
        """Return an effective-forecasts object: for each player in
        `coalition`, substitute its actual (baseline="perfect") or a naive
        proxy (baseline="naive"); every other player keeps its real
        forecast. The returned object is whatever shape this benchmark's
        own `run_day` expects as its `forecasts` argument."""
        ...

    def run_day(self, forecasts: Any, actuals: Any, config: PolicyConfig) -> DayResult:
        ...
