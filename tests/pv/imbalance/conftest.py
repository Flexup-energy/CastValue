from __future__ import annotations

import numpy as np
import pytest

from castvalue.pv.imbalance import DayActuals, DayForecasts, PvImbalanceBenchmark, PvImbalanceConfig


def make_random_day(seed: int, n: int = 96, capacity: float = 5.0):
    rng = np.random.default_rng(seed)
    prod_potential = np.clip(
        np.sin(np.linspace(0, np.pi, n)) * capacity * 0.8 + rng.normal(0, capacity * 0.05, n), 0, capacity
    )
    prod_fc_d1 = np.clip(prod_potential + rng.normal(0, capacity * 0.1, n), 0, capacity * 1.2)
    prod_fc_id = np.clip(prod_potential + rng.normal(0, capacity * 0.03, n), 0, capacity * 1.1)
    da_price_actual = rng.normal(40, 30, n)
    da_price_fc = da_price_actual + rng.normal(0, 10, n)
    id1_actual = da_price_actual + rng.normal(0, 5, n)
    id1_fc_d1 = id1_actual + rng.normal(0, 8, n)
    id1_fc_id = id1_actual + rng.normal(0, 3, n)
    imb_long_actual = id1_actual + rng.normal(0, 5, n)
    imb_short_actual = imb_long_actual + np.abs(rng.normal(5, 5, n))
    imb_long_fc_d1 = imb_long_actual + rng.normal(0, 10, n)
    imb_short_fc_d1 = imb_long_fc_d1 + np.abs(rng.normal(5, 5, n))
    imb_long_fc_id = imb_long_actual + rng.normal(0, 4, n)
    imb_short_fc_id = imb_long_fc_id + np.abs(rng.normal(5, 5, n))
    regulation_state = rng.choice([0, 1, -1, 2], size=n, p=[0.4, 0.2, 0.2, 0.2])

    forecasts = DayForecasts(
        prod_fc_d1=prod_fc_d1.tolist(),
        da_price_fc=da_price_fc.tolist(),
        id1_fc_d1=id1_fc_d1.tolist(),
        imb_price_short_fc_d1=imb_short_fc_d1.tolist(),
        imb_price_long_fc_d1=imb_long_fc_d1.tolist(),
        prod_fc_id=prod_fc_id.tolist(),
        id1_fc_id=id1_fc_id.tolist(),
        imb_price_short_fc_id=imb_short_fc_id.tolist(),
        imb_price_long_fc_id=imb_long_fc_id.tolist(),
    )
    actuals = DayActuals(
        prod_potential_actual=prod_potential.tolist(),
        da_price_actual=da_price_actual.tolist(),
        id1_actual=id1_actual.tolist(),
        imb_price_short_actual=imb_short_actual.tolist(),
        imb_price_long_actual=imb_long_actual.tolist(),
        regulation_state_actual=regulation_state.tolist(),
    )
    return forecasts, actuals


@pytest.fixture
def benchmark():
    return PvImbalanceBenchmark()


@pytest.fixture
def base_config():
    return PvImbalanceConfig(installed_capacity_mw=5.0)


@pytest.fixture
def random_day():
    return make_random_day(seed=12345)
