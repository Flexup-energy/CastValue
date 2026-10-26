"""A minimal toy benchmark, structurally conforming to castvalue.core's
Benchmark protocol, with no BESS or PV code involved -- this is what proves
castvalue.core is actually benchmark-agnostic rather than secretly tied to
one benchmark's shape.

Deliberately includes one genuine pairwise interaction (the `a`+`b` bonus
below) so attribution tests exercise real nonzero pairwise/Shapley values,
not a degenerate all-zero-interaction case. All coalition values below are
hand-computed in the test files' docstrings/comments, not just asserted
structurally.
"""

from __future__ import annotations

import dataclasses

import pytest

from castvalue.core import DayResult, PolicyConfig, make_provenance


@dataclasses.dataclass(frozen=True)
class ToyConfig(PolicyConfig):
    pass


class ToyBenchmark:
    name = "toy"
    METHOD_VERSION = "0.1"
    players = ("a", "b", "c")
    groups = {"low": ("a",), "high": ("b", "c")}
    ceiling_mode = "hard"

    def substitute(self, forecasts, actuals, coalition, baseline="perfect"):
        return {p: (actuals[p] if p in coalition else forecasts[p]) for p in self.players}

    def run_day(self, forecasts, actuals, config):
        score = sum(forecasts.values())
        # Deliberate pairwise interaction: a bonus that only fires when
        # BOTH a and b are at their substituted (actual, i.e. "perfected")
        # value simultaneously -- gives attribution tests a real, nonzero
        # pairwise term to check instead of an all-additive game.
        if forecasts["a"] > 1.5 and forecasts["b"] > 15.0:
            score += 5.0
        return DayResult(score=score, raw={"inputs": dict(forecasts)}, provenance=make_provenance(self, config))


@pytest.fixture
def toy_benchmark():
    return ToyBenchmark()


@pytest.fixture
def toy_forecasts():
    return {"a": 1.0, "b": 10.0, "c": 100.0}


@pytest.fixture
def toy_actuals():
    return {"a": 2.0, "b": 20.0, "c": 200.0}


@pytest.fixture
def toy_config():
    return ToyConfig()
