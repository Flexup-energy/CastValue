"""PV imbalance-cost benchmark v2 -- two-stage (D-1 + intraday) decision
policy over 7 forecast inputs, TenneT NL dual imbalance pricing, exact
vertex-enumeration solver (no LP in the runtime path)."""

from .benchmark import PvImbalanceBenchmark
from .inputs import DayActuals, DayForecasts, PvImbalanceConfig, validate_day
from .players import ACTUAL_FOR_FIELD, GROUPS, PLAYER_FIELDS, PLAYERS
from .sensitivity import pv_sensitivity_curve

METHOD_VERSION = "2.0"

__all__ = [
    "PvImbalanceBenchmark",
    "DayActuals",
    "DayForecasts",
    "PvImbalanceConfig",
    "validate_day",
    "PLAYERS",
    "GROUPS",
    "PLAYER_FIELDS",
    "ACTUAL_FOR_FIELD",
    "pv_sensitivity_curve",
    "METHOD_VERSION",
]
