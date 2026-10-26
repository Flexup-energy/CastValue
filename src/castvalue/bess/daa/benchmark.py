"""Adapter wiring castvalue.bess.daa.optimizer.Optimizer into the
castvalue.core Benchmark protocol.

Single player (`da_price`): a BESS dispatch decision here only consumes
one forecast input, unlike PV imbalance's 7 -- so "full coalition" and
"da_price substituted" are the same condition, and there's no partial-
coalition case at all.

ceiling_mode="hard": Perfect Foresight solves the LP directly on ACTUAL
prices (no OTM/rejection friction) for the exact same initial_soc, power/
energy caps, cycle limit and RTE as every other coalition in a day's game
-- i.e. it's the true profit-maximizing schedule over the identical
feasible set every other run also operates in. Any other run's REALIZED
schedule (after OTM + paired rejection) is necessarily some feasible
schedule within that same set, so its actual-price profit can't exceed
PF's. This only holds because every coalition in one day's game shares
the same `config.initial_soc` (a day's game isn't the right place for the
old notebook's pattern of letting each named scenario's SOC evolve
independently across days -- that would NOT give this guarantee, since
PF's day-N starting SOC isn't then guaranteed to dominate another
scenario's day-N SOC). Verified empirically too: zero ceiling violations
across all 252 real forecast-day pairs in
tests/bess/daa/golden/daa_de_lu_golden.json, and again directly through
this adapter in test_bess_daa_benchmark.py.
"""

from __future__ import annotations

import dataclasses

from castvalue.core import DayResult, PolicyConfig, make_provenance

from .optimizer import Optimizer


@dataclasses.dataclass(frozen=True)
class DaaConfig(PolicyConfig):
    power_cap_mw: float = 1.0
    energy_cap_mwh: float = 2.0
    n_cycles: float = 1.0
    rte: float = 0.85
    initial_soc: float = 0.0
    glpk_path: str = "/usr/bin/glpsol"


class BessDaaBenchmark:
    name = "bess-daa"
    METHOD_VERSION = "1.1"
    players = ("da_price",)
    groups = None
    ceiling_mode = "hard"

    def substitute(self, forecasts, actuals, coalition, baseline="perfect"):
        """forecasts/actuals: {"daa": [...]}, actuals optionally also
        carrying {"daa_naive": [...]} when baseline="naive" is used.

        perfect_foresight is carried explicitly on the returned dict
        rather than inferred inside run_day by comparing vectors -- it
        must be True only for baseline="perfect" with da_price
        substituted (the trader genuinely knows the clearing price), never
        for baseline="naive" (a naive substitute is still just another
        imperfect forecast and must go through normal OTM/bidding).
        """
        if "da_price" in coalition:
            if baseline == "perfect":
                price_vec = actuals["daa"]
            elif baseline == "naive":
                if "daa_naive" not in actuals:
                    raise ValueError(
                        'baseline="naive" requires actuals["daa_naive"] (e.g. a persistence price series)'
                    )
                price_vec = actuals["daa_naive"]
            else:
                raise ValueError(f"Unknown baseline: {baseline!r}")
            perfect_foresight = baseline == "perfect"
        else:
            price_vec = forecasts["daa"]
            perfect_foresight = False
        return {"daa": list(price_vec), "perfect_foresight": perfect_foresight}

    def run_day(self, forecasts, actuals, config: DaaConfig) -> DayResult:
        opt = Optimizer(config.glpk_path)
        out = opt.run_day(
            forecasts={"daa": forecasts["daa"]},
            actuals={"daa": actuals["daa"]},
            power_cap=config.power_cap_mw,
            energy_cap=config.energy_cap_mwh,
            n_cycles=config.n_cycles,
            rte=config.rte,
            initial_soc=config.initial_soc,
            perfect_foresight=forecasts["perfect_foresight"],
        )
        return DayResult(score=out["profit_total"], raw=out, provenance=make_provenance(self, config))


def cast_value(forecast_result: DayResult, naive_result: DayResult) -> float:
    """CastValue = forecast-run margin - naive-baseline margin -- the
    headline metric this whole benchmark is named after."""
    return forecast_result.score - naive_result.score
