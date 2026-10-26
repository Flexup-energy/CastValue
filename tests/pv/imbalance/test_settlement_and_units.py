"""Spec section 7: units hand-case + physical curtailment, tested directly
against settlement.settle_day (not the full policy solve) so these pin the
settlement arithmetic in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from castvalue.pv.imbalance import DayActuals, PvImbalanceConfig
from castvalue.pv.imbalance.settlement import settle_day


def _actuals(prod_potential, da_price=50.0, id1=55.0, imb_long=40.0, imb_short=60.0, regulation_state=0):
    return DayActuals(
        prod_potential_actual=[prod_potential],
        da_price_actual=[da_price],
        id1_actual=[id1],
        imb_price_short_actual=[imb_short],
        imb_price_long_actual=[imb_long],
        regulation_state_actual=[regulation_state],
    )


def test_units_single_interval_known_case():
    # nom_DA=2, id_trade=1 (x=3), setpoint=3, prod_potential=3 (no
    # curtailment binding) -> delivered=3, imb=delivered-x=0.
    config = PvImbalanceConfig(installed_capacity_mw=10.0)  # id_spread=0, risk_lambda=0 defaults
    result = settle_day(
        nom_da=np.array([2.0]), id_trade=np.array([1.0]), setpoint=np.array([3.0]),
        stage2_x=np.array([3.0]), stage2_d=np.array([3.0]),
        actuals=_actuals(prod_potential=3.0, da_price=50.0, id1=55.0, imb_long=40.0, imb_short=60.0),
        config=config,
    )
    expected_da_revenue = 2.0 * 50.0 * 0.25  # 25.0
    expected_id_revenue = 1.0 * 55.0 * 0.25  # 13.75, id_spread=0
    expected_imbalance = 0.0  # delivered(3) == x(3)
    expected_net = expected_da_revenue + expected_id_revenue + expected_imbalance
    assert result["da_revenue"][0] == pytest.approx(expected_da_revenue)
    assert result["id_revenue"][0] == pytest.approx(expected_id_revenue)
    assert result["imbalance_settlement"][0] == pytest.approx(expected_imbalance)
    assert result["net_raw_total"] == pytest.approx(expected_net)
    assert result["net_risk_adjusted_total"] == pytest.approx(expected_net)  # risk_lambda=0


def test_physical_curtailment_zero_setpoint_gives_zero_delivered():
    config = PvImbalanceConfig(installed_capacity_mw=10.0)
    result = settle_day(
        nom_da=np.array([0.0]), id_trade=np.array([0.0]), setpoint=np.array([0.0]),
        stage2_x=np.array([0.0]), stage2_d=np.array([0.0]),
        actuals=_actuals(prod_potential=5.0),  # plenty of potential, fully curtailed
        config=config,
    )
    assert result["delivered"][0] == pytest.approx(0.0)
    # x=0 and delivered=0 -> imbalance reflects the actual net position only, also 0
    assert result["imb_long"][0] == pytest.approx(0.0)
    assert result["imb_short"][0] == pytest.approx(0.0)


def test_physical_curtailment_imbalance_reflects_net_position_not_potential():
    # Curtailed to zero (setpoint=0, delivered=0) but a nonzero committed
    # position (x=2) -> short by exactly 2, regardless of how much
    # potential production was curtailed away.
    config = PvImbalanceConfig(installed_capacity_mw=10.0)
    result = settle_day(
        nom_da=np.array([2.0]), id_trade=np.array([0.0]), setpoint=np.array([0.0]),
        stage2_x=np.array([2.0]), stage2_d=np.array([0.0]),
        actuals=_actuals(prod_potential=5.0),
        config=config,
    )
    assert result["delivered"][0] == pytest.approx(0.0)
    assert result["imb_short"][0] == pytest.approx(2.0)
    assert result["imb_long"][0] == pytest.approx(0.0)
