"""Test-only oracles for validating the vertex-enumeration policy solver.

scipy is a TEST dependency only (see pyproject.toml's `test` extra) --
never imported from src/castvalue/pv/imbalance/ runtime code. Both oracles
solve the exact same per-interval problem policy.solve_xd / solve_stage1_*
solve, just via two independent, much slower general-purpose methods: a
real LP solver (HiGHS via scipy.optimize.linprog) and a plain numpy grid
search. Agreement across both plus the closed-form solver is the actual
correctness evidence for the vertex-enumeration approach.

LP validity note: the long/short split (`d - x = long - short`, both >=0,
maximizing `long*price_long - short*price_short`) only collapses to the
correct max(0,.)-based settlement when price_short >= price_long --
otherwise the relaxation is unbounded (the LP would want long and short
both arbitrarily large). This matches the real-world TenneT invariant
("shortage price >= surplus price") the spec itself states, so test
inputs must respect it; it is not a gap in the policy solver itself,
which never uses this LP relaxation trick (see policy._objective's direct
max(...,0) formulas).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def lp_oracle(
    *,
    nom_da_min,
    nom_da_max,
    d_hi,
    pos_min,
    pos_max,
    max_id_volume_mw,
    price_da,
    price_id,
    price_long,
    price_short,
    id_spread,
    risk_lambda,
    h,
):
    """Exact LP solve for one interval (all args scalars). Pass
    nom_da_min == nom_da_max to fix nom_DA (the stage-2 / skew-off-style
    2-variable problem); pass a genuine range to let nom_DA float free
    (the stage-1 skew-on 3-variable problem).

    Variable order: [nom_da, x, d, long, short, idpos, idneg, devpos, devneg]
    """
    # maximize: nom_da*price_da*h + (idpos-idneg)*price_id*h - (idpos+idneg)*id_spread*h
    #           + long*price_long*h - short*price_short*h - risk_lambda*h*(devpos+devneg)
    c_maximize = np.array([
        price_da * h,
        0.0,
        0.0,
        price_long * h,
        -price_short * h,
        price_id * h - id_spread * h,
        -price_id * h - id_spread * h,
        -risk_lambda * h,
        -risk_lambda * h,
    ])
    c = -c_maximize  # linprog minimizes

    # Equality constraints:
    #  d - x - long + short = 0
    #  x - nom_da - idpos + idneg = 0
    #  x - d - devpos + devneg = 0
    A_eq = np.array([
        [0, -1, 1, -1, 1, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0, -1, 1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, -1, 1],
    ])
    b_eq = np.array([0.0, 0.0, 0.0])

    # Inequality constraints coupling x to nom_da via the ID volume window:
    #  x - nom_da <= maxid
    #  nom_da - x <= maxid
    A_ub = np.array([
        [-1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0, 0],
    ])
    b_ub = np.array([max_id_volume_mw, max_id_volume_mw])

    bounds = [
        (nom_da_min, nom_da_max),
        (pos_min, pos_max),
        (0.0, d_hi),
        (0.0, None), (0.0, None),
        (0.0, None), (0.0, None),
        (0.0, None), (0.0, None),
    ]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP oracle failed to solve: {result.message}")
    nom_da, x, d = result.x[0], result.x[1], result.x[2]
    value = -result.fun
    return x, d, nom_da, value


def grid_oracle(
    *,
    nom_da_min,
    nom_da_max,
    d_hi,
    pos_min,
    pos_max,
    max_id_volume_mw,
    price_da,
    price_id,
    price_long,
    price_short,
    id_spread,
    risk_lambda,
    h,
    n_grid=250,
):
    """Brute-force grid search over (nom_da, x, d) for one interval,
    evaluating the exact piecewise objective (the real max(0,.)-based
    formula, not an LP relaxation) at every combination. Independent of
    both the LP oracle and the closed-form solver.
    """
    nom_da_vals = np.linspace(nom_da_min, nom_da_max, n_grid if nom_da_max > nom_da_min else 1)
    d_vals = np.linspace(0.0, d_hi, n_grid)

    best_value = -np.inf
    best = None
    for nom_da in nom_da_vals:
        x_lo = max(pos_min, nom_da - max_id_volume_mw)
        x_hi = min(pos_max, nom_da + max_id_volume_mw)
        if x_hi < x_lo:
            continue
        x_vals = np.linspace(x_lo, x_hi, n_grid)
        for x in x_vals:
            id_trade = x - nom_da
            for d in d_vals:
                imb = d - x
                imb_long = max(imb, 0.0)
                imb_short = max(-imb, 0.0)
                value = (
                    nom_da * price_da * h
                    + id_trade * price_id * h - abs(id_trade) * id_spread * h
                    + imb_long * price_long * h - imb_short * price_short * h
                    - risk_lambda * h * abs(x - d)
                )
                if value > best_value:
                    best_value = value
                    best = (x, d, nom_da)
    return (*best, best_value)
