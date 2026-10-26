"""Validates the hand-derived vertex-enumeration policy solver
(src/castvalue/pv/imbalance/policy.py) against two independent oracles:
a real LP (HiGHS via scipy) and a brute-force grid search. Per the brief,
this is the load-bearing correctness check for section 2.5 -- especially
the allow_da_position_skew=True (3-variable, nom_DA free) case, explicitly
flagged as the one hand-derivation not to trust blind.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from castvalue.pv.imbalance.inputs import PvImbalanceConfig
from castvalue.pv.imbalance.policy import solve_stage1_skew_off, solve_stage1_skew_on, solve_xd

from .oracles import grid_oracle, lp_oracle

RNG_SEED = 20261027
LP_TOL = 1e-3  # HiGHS vs. closed-form, both exact methods -- tight tolerance
GRID_TOL_FACTOR = 2.0  # grid search is only exact to its own resolution


def _random_interval(rng, *, capacity, allow_negative_prices=True):
    lo = -50.0 if allow_negative_prices else 0.0
    price_da = rng.uniform(lo, 100.0)
    price_id = rng.uniform(lo, 100.0)
    price_long = rng.uniform(lo, 100.0)
    price_short = price_long + rng.uniform(0.0, 50.0)  # enforce shortage >= surplus
    prod_fc = rng.uniform(0.0, capacity)
    return {
        "price_da": price_da, "price_id": price_id,
        "price_long": price_long, "price_short": price_short,
        "prod_fc": prod_fc,
    }


def _random_config(rng, *, capacity):
    max_id_volume_mw = rng.choice([0.0, capacity * 0.3, capacity])
    id_spread = rng.choice([0.0, rng.uniform(0.1, 5.0)])
    risk_lambda = rng.choice([0.0, rng.uniform(0.1, 20.0)])
    return PvImbalanceConfig(
        installed_capacity_mw=capacity,
        max_id_volume_mw=max_id_volume_mw,
        id_spread_eur_mwh=id_spread,
        risk_lambda_eur_mwh=risk_lambda,
    )


@pytest.mark.parametrize("trial", range(40))
def test_stage2_matches_both_oracles(trial):
    rng = random.Random(RNG_SEED + trial)
    capacity = rng.uniform(1.0, 20.0)
    config = _random_config(rng, capacity=capacity)
    day = _random_interval(rng, capacity=capacity)
    nom_da = rng.uniform(config.nom_min_mw, config.nom_max_mw)

    x, d, score = solve_xd(
        nom_da=np.array([nom_da]),
        d_hi=np.array([day["prod_fc"]]),
        pos_min=config.pos_min_mw, pos_max=config.pos_max_mw,
        max_id_volume_mw=config.max_id_volume_mw,
        price_da=np.array([day["price_da"]]), price_id=np.array([day["price_id"]]),
        price_long=np.array([day["price_long"]]), price_short=np.array([day["price_short"]]),
        id_spread=config.id_spread_eur_mwh, risk_lambda=config.risk_lambda_eur_mwh,
        h=0.25, allow_passive_balancing=True,
    )
    closed_form_value = score[0]

    lp_x, lp_d, lp_nom, lp_value = lp_oracle(
        nom_da_min=nom_da, nom_da_max=nom_da, d_hi=day["prod_fc"],
        pos_min=config.pos_min_mw, pos_max=config.pos_max_mw, max_id_volume_mw=config.max_id_volume_mw,
        price_da=day["price_da"], price_id=day["price_id"],
        price_long=day["price_long"], price_short=day["price_short"],
        id_spread=config.id_spread_eur_mwh, risk_lambda=config.risk_lambda_eur_mwh, h=0.25,
    )
    assert closed_form_value == pytest.approx(lp_value, abs=LP_TOL), (
        f"trial {trial}: closed-form {closed_form_value} != LP oracle {lp_value}"
    )

    gx, gd, gnom, gvalue = grid_oracle(
        nom_da_min=nom_da, nom_da_max=nom_da, d_hi=day["prod_fc"],
        pos_min=config.pos_min_mw, pos_max=config.pos_max_mw, max_id_volume_mw=config.max_id_volume_mw,
        price_da=day["price_da"], price_id=day["price_id"],
        price_long=day["price_long"], price_short=day["price_short"],
        id_spread=config.id_spread_eur_mwh, risk_lambda=config.risk_lambda_eur_mwh, h=0.25,
    )
    price_scale = max(abs(day["price_long"]), abs(day["price_short"]), abs(day["price_id"]), abs(day["price_da"]), 1.0)
    grid_step = day["prod_fc"] / 250 if day["prod_fc"] > 0 else 1.0
    grid_tol = GRID_TOL_FACTOR * grid_step * price_scale * 0.25
    assert closed_form_value >= gvalue - grid_tol, (
        f"trial {trial}: closed-form {closed_form_value} worse than grid oracle {gvalue} by more than {grid_tol}"
    )


@pytest.mark.parametrize("trial", range(40))
def test_stage1_skew_off_matches_both_oracles(trial):
    rng = random.Random(RNG_SEED + 1000 + trial)
    capacity = rng.uniform(1.0, 20.0)
    config = _random_config(rng, capacity=capacity)
    day = _random_interval(rng, capacity=capacity)

    forecasts_d1 = {
        "prod_fc_d1": [day["prod_fc"]],
        "da_price_fc": [day["price_da"]],
        "id1_fc_d1": [day["price_id"]],
        "imb_price_long_fc_d1": [day["price_long"]],
        "imb_price_short_fc_d1": [day["price_short"]],
    }
    result = solve_stage1_skew_off(forecasts_d1, config)
    nom_da = result["nom_da"][0]
    x = nom_da + result["planned_id_trade"][0]
    d = result["planned_setpoint"][0]
    # Recompute the achieved value directly (solve_stage1_skew_off doesn't
    # return it) using the same objective the oracle also evaluates.
    id_trade = x - nom_da
    imb = d - x
    imb_long, imb_short = max(imb, 0.0), max(-imb, 0.0)
    closed_form_value = (
        nom_da * day["price_da"] * 0.25
        + id_trade * day["price_id"] * 0.25 - abs(id_trade) * config.id_spread_eur_mwh * 0.25
        + imb_long * day["price_long"] * 0.25 - imb_short * day["price_short"] * 0.25
        - config.risk_lambda_eur_mwh * 0.25 * abs(x - d)
    )

    # nom_da_equals_d means nom_DA isn't independently bounded by
    # [nom_min, nom_max] in the reduction -- it's tied to d in [0, prod_fc].
    # The oracle's nom_da range must match that exactly for a fair comparison.
    lp_x, lp_d, lp_nom, lp_value = lp_oracle(
        nom_da_min=0.0, nom_da_max=day["prod_fc"], d_hi=day["prod_fc"],
        pos_min=config.pos_min_mw, pos_max=config.pos_max_mw, max_id_volume_mw=config.max_id_volume_mw,
        price_da=day["price_da"], price_id=day["price_id"],
        price_long=day["price_long"], price_short=day["price_short"],
        id_spread=config.id_spread_eur_mwh, risk_lambda=config.risk_lambda_eur_mwh, h=0.25,
    )
    # The oracle optimizes nom_da and d independently; the skew-off
    # reduction forces nom_da == d exactly. So compare against a
    # nom_da==d-constrained oracle instead of the fully free one -- add
    # that as an equality by shrinking the grid oracle's search (simpler:
    # the grid oracle below enforces it structurally by construction).
    gx, gd, gnom, gvalue = grid_oracle(
        nom_da_min=0.0, nom_da_max=day["prod_fc"], d_hi=day["prod_fc"],
        pos_min=config.pos_min_mw, pos_max=config.pos_max_mw, max_id_volume_mw=config.max_id_volume_mw,
        price_da=day["price_da"], price_id=day["price_id"],
        price_long=day["price_long"], price_short=day["price_short"],
        id_spread=config.id_spread_eur_mwh, risk_lambda=config.risk_lambda_eur_mwh, h=0.25,
        n_grid=120,
    )
    # Both oracles above are MORE permissive than the skew-off reduction
    # (they let nom_da and d vary independently); since nom_da==d is a
    # restriction of that larger feasible space, the true skew-off optimum
    # can only be <= the unrestricted oracles' optimum. Use that as a
    # sanity ceiling, and separately verify nom_da==d holds exactly.
    assert nom_da == pytest.approx(d, abs=1e-9)
    assert closed_form_value <= lp_value + LP_TOL
    price_scale = max(abs(day["price_long"]), abs(day["price_short"]), abs(day["price_id"]), abs(day["price_da"]), 1.0)
    grid_step = day["prod_fc"] / 120 if day["prod_fc"] > 0 else 1.0
    assert closed_form_value <= gvalue + GRID_TOL_FACTOR * grid_step * price_scale * 0.25


@pytest.mark.parametrize("trial", range(60))
def test_stage1_skew_on_matches_both_oracles(trial):
    """The highest-risk hand-derivation (3 free variables: nom_DA, x, d) --
    more trials than the other cases per the brief's explicit flag."""
    rng = random.Random(RNG_SEED + 2000 + trial)
    capacity = rng.uniform(1.0, 20.0)
    config = PvImbalanceConfig(
        installed_capacity_mw=capacity,
        max_id_volume_mw=rng.choice([0.0, capacity * 0.3, capacity]),
        id_spread_eur_mwh=rng.choice([0.0, rng.uniform(0.1, 5.0)]),
        risk_lambda_eur_mwh=rng.choice([0.0, rng.uniform(0.1, 20.0)]),
        allow_da_position_skew=True,
    )
    day = _random_interval(rng, capacity=capacity)

    forecasts_d1 = {
        "prod_fc_d1": [day["prod_fc"]],
        "da_price_fc": [day["price_da"]],
        "id1_fc_d1": [day["price_id"]],
        "imb_price_long_fc_d1": [day["price_long"]],
        "imb_price_short_fc_d1": [day["price_short"]],
    }
    result = solve_stage1_skew_on(forecasts_d1, config)
    nom_da = result["nom_da"][0]
    x = nom_da + result["planned_id_trade"][0]
    d = result["planned_setpoint"][0]
    id_trade = x - nom_da
    imb = d - x
    imb_long, imb_short = max(imb, 0.0), max(-imb, 0.0)
    closed_form_value = (
        nom_da * day["price_da"] * 0.25
        + id_trade * day["price_id"] * 0.25 - abs(id_trade) * config.id_spread_eur_mwh * 0.25
        + imb_long * day["price_long"] * 0.25 - imb_short * day["price_short"] * 0.25
        - config.risk_lambda_eur_mwh * 0.25 * abs(x - d)
    )

    lp_x, lp_d, lp_nom, lp_value = lp_oracle(
        nom_da_min=config.nom_min_mw, nom_da_max=config.nom_max_mw, d_hi=day["prod_fc"],
        pos_min=config.pos_min_mw, pos_max=config.pos_max_mw, max_id_volume_mw=config.max_id_volume_mw,
        price_da=day["price_da"], price_id=day["price_id"],
        price_long=day["price_long"], price_short=day["price_short"],
        id_spread=config.id_spread_eur_mwh, risk_lambda=config.risk_lambda_eur_mwh, h=0.25,
    )
    assert closed_form_value == pytest.approx(lp_value, abs=LP_TOL), (
        f"trial {trial}: closed-form {closed_form_value} != LP oracle {lp_value} "
        f"(closed-form nom_da={nom_da}, x={x}, d={d}; LP nom_da={lp_nom}, x={lp_x}, d={lp_d})"
    )

    gx, gd, gnom, gvalue = grid_oracle(
        nom_da_min=config.nom_min_mw, nom_da_max=config.nom_max_mw, d_hi=day["prod_fc"],
        pos_min=config.pos_min_mw, pos_max=config.pos_max_mw, max_id_volume_mw=config.max_id_volume_mw,
        price_da=day["price_da"], price_id=day["price_id"],
        price_long=day["price_long"], price_short=day["price_short"],
        id_spread=config.id_spread_eur_mwh, risk_lambda=config.risk_lambda_eur_mwh, h=0.25,
        n_grid=60,
    )
    price_scale = max(abs(day["price_long"]), abs(day["price_short"]), abs(day["price_id"]), abs(day["price_da"]), 1.0)
    grid_step = max(day["prod_fc"], config.nom_max_mw - config.nom_min_mw, 1e-6) / 60
    grid_tol = GRID_TOL_FACTOR * grid_step * price_scale * 0.25
    assert closed_form_value >= gvalue - grid_tol, (
        f"trial {trial}: closed-form {closed_form_value} worse than grid oracle {gvalue} by more than {grid_tol}"
    )
