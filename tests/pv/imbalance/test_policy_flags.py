"""Spec section 7: flag-driven behaviors and tie-breaking stability."""

from __future__ import annotations

import numpy as np
import pytest

from castvalue.pv.imbalance import DayActuals, DayForecasts, PvImbalanceConfig, PvImbalanceBenchmark
from castvalue.pv.imbalance.policy import solve_xd


def test_tie_breaking_prefers_no_curtailment_then_smallest_id_trade():
    # Fully degenerate objective (every price 0) -> every (x, d) combo
    # ties exactly. Must deterministically pick: no curtailment (d=d_hi),
    # then smallest |id_trade| (x=nom_da).
    nom_da = np.array([1.5])
    x, d, score = solve_xd(
        nom_da=nom_da, d_hi=np.array([4.0]),
        pos_min=0.0, pos_max=5.0, max_id_volume_mw=5.0,
        price_da=np.array([0.0]), price_id=np.array([0.0]),
        price_long=np.array([0.0]), price_short=np.array([0.0]),
        id_spread=0.0, risk_lambda=0.0, h=0.25, allow_passive_balancing=True,
    )
    assert d[0] == pytest.approx(4.0)  # no curtailment
    assert x[0] == pytest.approx(1.5)  # x = nom_da, zero id_trade


def test_tie_breaking_stable_under_tiny_float_perturbation():
    base_kwargs = dict(
        nom_da=np.array([1.5]), d_hi=np.array([4.0]),
        pos_min=0.0, pos_max=5.0, max_id_volume_mw=5.0,
        price_id=np.array([0.0]), price_long=np.array([0.0]), price_short=np.array([0.0]),
        id_spread=0.0, risk_lambda=0.0, h=0.25, allow_passive_balancing=True,
    )
    x1, d1, _ = solve_xd(price_da=np.array([0.0]), **base_kwargs)
    x2, d2, _ = solve_xd(price_da=np.array([1e-10]), **base_kwargs)  # tiny perturbation, well within tol
    assert x1[0] == pytest.approx(x2[0])
    assert d1[0] == pytest.approx(d2[0])


def test_both_flags_false_with_perfect_production_forecast_gives_zero_imbalance(random_day, base_config):
    forecasts, actuals = random_day
    # Perfect production forecast for both vintages.
    forecasts = DayForecasts(
        **{**forecasts.as_dict(), "prod_fc_d1": actuals.prod_potential_actual, "prod_fc_id": actuals.prod_potential_actual}
    )
    config = PvImbalanceConfig(
        installed_capacity_mw=base_config.installed_capacity_mw,
        allow_passive_balancing=False,
        allow_da_position_skew=False,
    )
    bm = PvImbalanceBenchmark()
    result = bm.run_day(forecasts, actuals, config)
    assert np.allclose(result.raw["imb_long"], 0.0, atol=1e-6)
    assert np.allclose(result.raw["imb_short"], 0.0, atol=1e-6)


def test_skew_off_nom_da_equals_d1_forecast_after_planned_curtailment(random_day, base_config):
    forecasts, actuals = random_day
    config = PvImbalanceConfig(installed_capacity_mw=base_config.installed_capacity_mw, allow_da_position_skew=False)
    bm = PvImbalanceBenchmark()
    result = bm.run_day(forecasts, actuals, config)
    assert np.allclose(result.raw["nom_da"], result.raw["planned_setpoint"], atol=1e-9)


def test_max_id_volume_zero_gives_zero_id_revenue_and_spread_cost(random_day):
    forecasts, actuals = random_day
    config = PvImbalanceConfig(
        installed_capacity_mw=5.0, max_id_volume_mw=0.0, id_spread_eur_mwh=5.0,  # nonzero spread, shouldn't matter
    )
    bm = PvImbalanceBenchmark()
    result = bm.run_day(forecasts, actuals, config)
    assert np.allclose(result.raw["id_trade"], 0.0, atol=1e-9)
    assert np.allclose(result.raw["id_revenue"], 0.0, atol=1e-9)
