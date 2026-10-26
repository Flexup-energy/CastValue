"""PV-specific sensitivity-curve wiring (spec section 4): blend ONE
target player or group toward its forecast value by alpha while every
OTHER player stays perfected (set to actual) -- answers "how much does
degrading just this one input's accuracy cost, holding everything else
perfect?" Reuses castvalue.core's generic alpha_blend/sensitivity_curve;
only the "which fields does this target touch" wiring is PV-specific.
"""

from __future__ import annotations

import numpy as np

from castvalue.core import alpha_blend, sensitivity_curve

from .inputs import DayForecasts
from .players import ACTUAL_FOR_FIELD, GROUPS, PLAYER_FIELDS, PLAYERS


def _target_fields_and_players(target: str) -> tuple[tuple[str, ...], frozenset[str]]:
    if target in PLAYERS:
        return PLAYER_FIELDS[target], frozenset({target})
    if target in GROUPS:
        group_players = GROUPS[target]
        fields = tuple(f for p in group_players for f in PLAYER_FIELDS[p])
        return fields, frozenset(group_players)
    raise ValueError(f"Unknown player/group: {target!r}")


def pv_sensitivity_curve(benchmark, forecasts, actuals, config, target: str, alphas=None):
    target_fields, target_players = _target_fields_and_players(target)
    others_perfected_coalition = frozenset(benchmark.players) - target_players
    base = benchmark.substitute(forecasts, actuals, others_perfected_coalition, baseline="perfect")

    def run_fn(alpha: float) -> float:
        fields = base.as_dict()
        for field in target_fields:
            forecast_val = np.asarray(fields[field], dtype=float)
            actual_val = np.asarray(getattr(actuals, ACTUAL_FOR_FIELD[field]), dtype=float)
            fields[field] = alpha_blend(forecast_val, actual_val, alpha).tolist()
        day = DayForecasts(**fields)
        return benchmark.run_day(day, actuals, config).score

    if alphas is not None:
        return sensitivity_curve(run_fn, alphas=alphas)
    return sensitivity_curve(run_fn)
