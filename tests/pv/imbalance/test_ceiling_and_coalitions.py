"""Spec section 7: PF ceiling fuzzed across random days/configs, and the
coalition-identity checks (empty=forecast run, full=PF) on the real PV
benchmark -- not just the toy benchmark in tests/core/.
"""

from __future__ import annotations

import pytest

from castvalue.core import assert_ceiling, run_all_coalitions
from castvalue.pv.imbalance import PvImbalanceBenchmark, PvImbalanceConfig

from .conftest import make_random_day


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize(
    "config_kwargs",
    [
        {},
        {"allow_da_position_skew": True},
        {"allow_passive_balancing": False},
        {"risk_lambda_eur_mwh": 5.0},
        {"id_spread_eur_mwh": 2.0},
        {"max_id_volume_mw": 0.0},
        {"imbalance_pricing": "single"},
    ],
)
def test_pf_ceiling_holds_across_random_days_and_configs(seed, config_kwargs):
    forecasts, actuals = make_random_day(seed=seed, capacity=5.0)
    config = PvImbalanceConfig(installed_capacity_mw=5.0, **config_kwargs)
    bm = PvImbalanceBenchmark()

    forecast_result = bm.run_day(forecasts, actuals, config)
    pf_forecasts = bm.substitute(forecasts, actuals, frozenset(bm.players), baseline="perfect")
    pf_result = bm.run_day(pf_forecasts, actuals, config)

    assert assert_ceiling(
        forecast_result.score, pf_result.score, mode=bm.ceiling_mode, tol=1e-4,
        context=f"seed={seed}, config={config_kwargs}",
    )


def test_empty_coalition_reproduces_plain_forecast_run():
    forecasts, actuals = make_random_day(seed=1)
    config = PvImbalanceConfig(installed_capacity_mw=5.0)
    bm = PvImbalanceBenchmark()
    results = run_all_coalitions(bm, forecasts, actuals, config)
    direct = bm.run_day(forecasts, actuals, config)
    assert results[frozenset()].score == pytest.approx(direct.score)


def test_full_coalition_reproduces_perfect_foresight():
    forecasts, actuals = make_random_day(seed=2)
    config = PvImbalanceConfig(installed_capacity_mw=5.0)
    bm = PvImbalanceBenchmark()
    results = run_all_coalitions(bm, forecasts, actuals, config)
    pf_forecasts = bm.substitute(forecasts, actuals, frozenset(bm.players), baseline="perfect")
    direct_pf = bm.run_day(pf_forecasts, actuals, config)
    assert results[frozenset(bm.players)].score == pytest.approx(direct_pf.score)
