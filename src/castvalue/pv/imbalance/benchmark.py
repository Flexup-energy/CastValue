"""Adapter wiring the PV imbalance two-stage policy into the
castvalue.core Benchmark protocol.

ceiling_mode="hard": Perfect Foresight is the same policy (solve_stage1 +
solve_stage2 + settle_day) fed actuals for all 7 forecast inputs -- not a
separately-coded path -- so by construction no forecast-driven coalition
can score higher (within solver tolerance); a violation means a policy or
constraint bug, which is exactly what the hard ceiling is for.
"""

from __future__ import annotations

import numpy as np

from castvalue.core import DayResult, make_provenance

from .inputs import DayActuals, DayForecasts, PvImbalanceConfig, validate_day
from .players import ACTUAL_FOR_FIELD, GROUPS, PLAYER_FIELDS, PLAYERS
from .policy import solve_stage1, solve_stage2
from .settlement import settle_day


class PvImbalanceBenchmark:
    name = "pv-imbalance"
    METHOD_VERSION = "2.0"
    players = PLAYERS
    groups = GROUPS
    ceiling_mode = "hard"

    def substitute(
        self,
        forecasts: DayForecasts,
        actuals: DayActuals,
        coalition: frozenset,
        baseline: str = "perfect",
    ) -> DayForecasts:
        updated = {}
        for player in self.players:
            for field in PLAYER_FIELDS[player]:
                if player in coalition:
                    if baseline == "perfect":
                        sub_value = getattr(actuals, ACTUAL_FOR_FIELD[field])
                    elif baseline == "naive":
                        if actuals.naive is None or field not in actuals.naive:
                            raise ValueError(
                                f'baseline="naive" requires actuals.naive["{field}"]'
                            )
                        sub_value = actuals.naive[field]
                    else:
                        raise ValueError(f"Unknown baseline: {baseline!r}")
                    updated[field] = list(sub_value)
                else:
                    updated[field] = list(getattr(forecasts, field))
        return DayForecasts(**updated)

    def run_day(self, forecasts: DayForecasts, actuals: DayActuals, config: PvImbalanceConfig) -> DayResult:
        validate_day(forecasts, actuals)

        forecasts_d1 = {
            "prod_fc_d1": forecasts.prod_fc_d1,
            "da_price_fc": forecasts.da_price_fc,
            "id1_fc_d1": forecasts.id1_fc_d1,
            "imb_price_long_fc_d1": forecasts.imb_price_long_fc_d1,
            "imb_price_short_fc_d1": forecasts.imb_price_short_fc_d1,
        }
        stage1 = solve_stage1(forecasts_d1, config)
        nom_da = stage1["nom_da"]

        forecasts_id = {
            "prod_fc_id": forecasts.prod_fc_id,
            # The actual DA price is known by the intraday stage (spec
            # section 0) -- not a forecast, not one of the 7 players.
            "da_price_actual": actuals.da_price_actual,
            "id1_fc_id": forecasts.id1_fc_id,
            "imb_price_long_fc_id": forecasts.imb_price_long_fc_id,
            "imb_price_short_fc_id": forecasts.imb_price_short_fc_id,
        }
        stage2 = solve_stage2(forecasts_id, nom_da, config)

        settled = settle_day(
            nom_da=nom_da,
            id_trade=stage2["id_trade"],
            setpoint=stage2["setpoint"],
            stage2_x=stage2["x"],
            stage2_d=stage2["d"],
            actuals=actuals,
            config=config,
        )

        raw = {
            "nom_da": nom_da.tolist(),
            "planned_id_trade": np.asarray(stage1["planned_id_trade"]).tolist(),
            "planned_setpoint": np.asarray(stage1["planned_setpoint"]).tolist(),
            "id_trade": np.asarray(stage2["id_trade"]).tolist(),
            "setpoint": np.asarray(stage2["setpoint"]).tolist(),
            **{
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in settled.items()
            },
        }
        return DayResult(
            score=settled["net_risk_adjusted_total"],
            raw=raw,
            provenance=make_provenance(self, config),
        )
