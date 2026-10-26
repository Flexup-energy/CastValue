"""BESS day-ahead-only benchmark — RTE-aware LP with opportunity-cost bidding.

The benchmark-protocol adapter (`benchmark.py`) lands here once
`castvalue.core.benchmark` exists (Milestone 2) — this package currently
exposes just the optimizer, moved verbatim from the repo root.
"""

from .optimizer import Optimizer

METHOD_VERSION = "1.1"

__all__ = ["Optimizer", "METHOD_VERSION"]
