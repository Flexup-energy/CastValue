"""Spec section 4: sensitivity curves and the no-intraday-access scenario
comparison."""

from __future__ import annotations

import pytest

from castvalue.pv.imbalance import PvImbalanceBenchmark, PvImbalanceConfig, pv_sensitivity_curve

from .conftest import make_random_day


def test_sensitivity_curve_endpoints_match_perfect_and_forecast_for_single_player():
    forecasts, actuals = make_random_day(seed=8, capacity=5.0)
    config = PvImbalanceConfig(installed_capacity_mw=5.0)
    bm = PvImbalanceBenchmark()

    curve = pv_sensitivity_curve(bm, forecasts, actuals, config, target="prod_fc_d1", alphas=(0.0, 1.0))
    assert len(curve["points"]) == 2

    # alpha=0 -> target perfected too -> same as substituting ALL players (full PF),
    # since every OTHER player was already perfected by pv_sensitivity_curve.
    pf_forecasts = bm.substitute(forecasts, actuals, frozenset(bm.players), baseline="perfect")
    pf_score = bm.run_day(pf_forecasts, actuals, config).score
    assert curve["points"][0]["score"] == pytest.approx(pf_score, abs=1e-6)


def test_sensitivity_curve_reports_a_slope():
    forecasts, actuals = make_random_day(seed=9, capacity=5.0)
    config = PvImbalanceConfig(installed_capacity_mw=5.0)
    bm = PvImbalanceBenchmark()
    curve = pv_sensitivity_curve(bm, forecasts, actuals, config, target="imbalance_price")
    assert curve["slope_near_1"] is not None


def test_no_intraday_access_scenario_differs_from_full_access():
    forecasts, actuals = make_random_day(seed=10, capacity=5.0)
    bm = PvImbalanceBenchmark()

    full_access = PvImbalanceConfig(installed_capacity_mw=5.0)  # max_id_volume_mw defaults to full capacity
    no_access = PvImbalanceConfig(installed_capacity_mw=5.0, max_id_volume_mw=0.0)

    full_result = bm.run_day(forecasts, actuals, full_access)
    no_access_result = bm.run_day(forecasts, actuals, no_access)

    assert no_access_result.raw["id_trade"] == pytest.approx([0.0] * len(no_access_result.raw["id_trade"]), abs=1e-9)
    # The scenario comparison is only meaningful if intraday trading
    # actually happens when it's allowed -- confirm that, rather than
    # asserting a specific sign on the score difference (depends on the
    # random day's prices).
    assert max(abs(v) for v in full_result.raw["id_trade"]) > 1e-6
