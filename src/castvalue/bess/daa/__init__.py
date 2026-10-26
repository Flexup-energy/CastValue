"""BESS day-ahead-only benchmark — RTE-aware LP with opportunity-cost bidding."""

from .benchmark import BessDaaBenchmark, DaaConfig, cast_value
from .optimizer import Optimizer

METHOD_VERSION = "1.1"

__all__ = ["Optimizer", "BessDaaBenchmark", "DaaConfig", "cast_value", "METHOD_VERSION"]
