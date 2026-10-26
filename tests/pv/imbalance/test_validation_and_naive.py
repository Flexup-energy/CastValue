"""Spec section 7 (NaN handling) + section 4 (configurable naive baseline)."""

from __future__ import annotations

import math

import pytest

from castvalue.pv.imbalance import DayActuals, PvImbalanceBenchmark, validate_day

from .conftest import make_random_day


def test_nan_in_forecast_is_rejected():
    forecasts, actuals = make_random_day(seed=3, n=4)
    bad = {**forecasts.as_dict()}
    bad["prod_fc_d1"] = [bad["prod_fc_d1"][0], float("nan"), *bad["prod_fc_d1"][2:]]
    from castvalue.pv.imbalance import DayForecasts
    bad_forecasts = DayForecasts(**bad)
    with pytest.raises(ValueError, match=r"prod_fc_d1\[1\]"):
        validate_day(bad_forecasts, actuals)


def test_none_in_actual_is_rejected():
    forecasts, actuals = make_random_day(seed=4, n=4)
    bad = {**actuals.as_dict(), "naive": None}
    bad["da_price_actual"] = [None, *bad["da_price_actual"][1:]]
    bad_actuals = DayActuals(**bad)
    with pytest.raises(ValueError, match=r"da_price_actual\[0\]"):
        validate_day(forecasts, bad_actuals)


def test_length_mismatch_is_rejected():
    forecasts, actuals = make_random_day(seed=5, n=4)
    from castvalue.pv.imbalance import DayForecasts
    bad_forecasts = DayForecasts(**{**forecasts.as_dict(), "prod_fc_d1": forecasts.prod_fc_d1[:-1]})
    with pytest.raises(ValueError):
        validate_day(bad_forecasts, actuals)


def test_naive_baseline_substitution_uses_supplied_proxy():
    forecasts, actuals = make_random_day(seed=6, n=4)
    naive_prod = [v * 0.5 for v in actuals.prod_potential_actual]
    actuals_with_naive = DayActuals(**{**actuals.as_dict(), "naive": {"prod_fc_d1": naive_prod}})
    bm = PvImbalanceBenchmark()
    substituted = bm.substitute(forecasts, actuals_with_naive, frozenset({"prod_fc_d1"}), baseline="naive")
    assert substituted.prod_fc_d1 == naive_prod
    # Untouched players keep the original forecast
    assert substituted.da_price_fc == forecasts.da_price_fc


def test_naive_baseline_without_proxy_raises():
    forecasts, actuals = make_random_day(seed=6, n=4)
    bm = PvImbalanceBenchmark()
    with pytest.raises(ValueError, match="naive"):
        bm.substitute(forecasts, actuals, frozenset({"prod_fc_d1"}), baseline="naive")
