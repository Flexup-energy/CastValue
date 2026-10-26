"""Per-interval exact vertex-enumeration solver for the PV imbalance
two-stage decision policy. No LP solver anywhere in this module -- see
section 2.5 of the spec and tests/pv/imbalance/test_policy_optimality.py
for the dual-oracle (LP + grid search) validation this is checked against.

Reformulation used throughout: instead of (nom_DA, id_trade, setpoint),
work with
  x = net contracted position after the stage's ID trade (= nom_DA + id_trade)
  d = anticipated/realized delivered volume, 0 <= d <= available production.
d plays the role of "curtailment setpoint" directly: any d in that range
is always achievable by setting the setpoint to d itself (delivered =
min(potential, setpoint) = d when setpoint = d and d <= potential), so the
min() nonlinearity never needs to appear as a separate decision variable.
id_trade is recovered afterwards as x - nom_DA.

The anticipated/realized imbalance position is (d - x): positive = long
(imb_long = d - x), negative = short (imb_short = x - d).

Per-interval independence (spec section 0): if a cross-interval constraint
is ever added (daily volume cap, ramp limit, cross-interval position
limit), this stops being separable and needs a multi-period LP like
castvalue.bess.daa -- do not extend vertex enumeration past that point.

Each interval's objective is piecewise-linear with kinks only at x=nom_DA
(the |id_trade| spread-cost term) and x=d (the long/short imbalance split
and the risk penalty). A piecewise-linear objective over a box is always
maximized at a box corner or a kink -- never a segment interior -- so for
every d-candidate we only need to check x at its own box ends plus the two
kink locations clipped into that box (4 x-candidates), and the d-candidate
set itself only needs d's own box ends plus whatever d-value makes x's
*own* bounds coincide with a kink (handled below). This is exact, not a
heuristic: every candidate corresponds to a real vertex of the feasible
polytope (including kink lines), and the true optimum is always at one of
them.
"""

from __future__ import annotations

import numpy as np

TIE_TOL = 1e-6


def _clip(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def _objective(x, d, nom_da, price_da, price_id, price_long, price_short, id_spread, risk_lambda, h):
    """Full per-interval payoff (DA + ID + imbalance - risk penalty) as a
    function of (x, d), given nom_da (may itself be an array equal to d,
    for the allow_da_position_skew=False reduction -- see
    solve_stage1_skew_off)."""
    da_revenue = nom_da * price_da * h
    id_trade = x - nom_da
    id_revenue = id_trade * price_id * h - np.abs(id_trade) * id_spread * h
    imb = d - x
    imb_long = np.maximum(imb, 0.0)
    imb_short = np.maximum(-imb, 0.0)
    imbalance = imb_long * price_long * h - imb_short * price_short * h
    risk_penalty = risk_lambda * np.abs(x - d) * h
    return da_revenue + id_revenue + imbalance - risk_penalty


def _x_bounds(nom_da, pos_min, pos_max, max_id_volume_mw):
    x_lo = np.maximum(pos_min, nom_da - max_id_volume_mw)
    x_hi = np.minimum(pos_max, nom_da + max_id_volume_mw)
    return x_lo, x_hi


def _tie_break_select(values, xs, ds, nom_das, d_hi, tol=TIE_TOL):
    """values/xs/ds/nom_das: arrays of shape (n_candidates, n_intervals).
    Returns (best_x, best_d, best_val), each shape (n_intervals,).

    Tie-break priority on near-equal objective values (within tol):
      1. no curtailment over curtailment (d close to d_hi, the available
         production, preferred over d < d_hi),
      2. smallest |id_trade| = |x - nom_da|,
      3. smallest deliberate deviation |x - d|.
    Comparisons use tol throughout, never raw float equality, so results
    stay stable under tiny perturbations -- essential for reproducible
    Shapley values.
    """
    n_candidates, n = values.shape
    best_val = np.max(values, axis=0)
    is_tied = values >= (best_val[None, :] - tol)

    is_curtailing = (ds < (d_hi[None, :] - tol)).astype(float)
    id_trade_mag = np.abs(xs - nom_das)
    deviation = np.abs(xs - ds)

    # Rank candidates lexicographically; non-tied candidates get +inf so
    # they're never selected regardless of their (unused) tie-break stats.
    inf = np.inf
    rank_curtail = np.where(is_tied, is_curtailing, inf)
    rank_idtrade = np.where(is_tied, id_trade_mag, inf)
    rank_dev = np.where(is_tied, deviation, inf)

    # Lexicographic argmin across the candidate axis via a stable composite
    # key -- scale each tier below the previous tier's resolution so no
    # tier can ever be overridden by noise in a lower-priority tier.
    big = 1e12
    composite = rank_curtail * big * big + rank_idtrade * big + rank_dev
    best_idx = np.argmin(composite, axis=0)

    rows = best_idx
    cols = np.arange(n)
    return xs[rows, cols], ds[rows, cols], values[rows, cols]


def solve_xd(
    nom_da,
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
    allow_passive_balancing,
    nom_da_equals_d=False,
):
    """Solve the 2-variable (x, d) problem for one stage, vectorised across
    all intervals of a day (every argument is an array of length N).

    nom_da: the fixed, already-committed net position reference (stage 2:
        the committed nom_DA from stage 1). Ignored (recomputed per
        candidate as nom_da=d) when nom_da_equals_d=True -- the
        allow_da_position_skew=False stage-1 reduction, where "nom_DA must
        equal the D-1 production forecast after planned curtailment" i.e.
        nom_DA is literally tied to whatever d turns out to be.
    d_hi: available production this stage sees (the forecast production
        vector) -- d ranges over [0, d_hi].

    Returns (x, d, score) arrays, one value per interval.
    """
    d_lo = np.zeros_like(d_hi)

    # Start with the box ends and (when nom_da is fixed) the nom_DA-derived
    # candidate; then add d-values derived from probing x's own box bounds
    # at each of those -- this catches the vertex where d lands exactly on
    # x's box boundary (from pos_min/pos_max or the ID-volume window),
    # which is NOT the same point as d_lo, d_hi, or nom_DA in general.
    # Found by the LP/grid oracle tests catching exactly this gap -- see
    # tests/pv/imbalance/test_policy_optimality.py.
    probe_ds = [d_lo, d_hi]
    if not nom_da_equals_d:
        probe_ds.append(_clip(nom_da, d_lo, d_hi))

    d_candidates = list(probe_ds)
    for probe_d in probe_ds:
        nom_da_probe = probe_d if nom_da_equals_d else nom_da
        x_lo_probe, x_hi_probe = _x_bounds(nom_da_probe, pos_min, pos_max, max_id_volume_mw)
        d_candidates.append(_clip(x_lo_probe, d_lo, d_hi))
        d_candidates.append(_clip(x_hi_probe, d_lo, d_hi))

    all_x, all_d, all_nom = [], [], []
    for d_cand in d_candidates:
        nom_da_for_candidate = d_cand if nom_da_equals_d else nom_da
        x_lo, x_hi = _x_bounds(nom_da_for_candidate, pos_min, pos_max, max_id_volume_mw)
        if allow_passive_balancing:
            x_candidates = [
                x_lo,
                x_hi,
                _clip(nom_da_for_candidate, x_lo, x_hi),
                _clip(d_cand, x_lo, x_hi),
            ]
        else:
            # id_trade forced to close the gap to d as much as the volume
            # limit allows -- not a free decision, so there's exactly one
            # x for this d.
            forced_id_trade = _clip(d_cand - nom_da_for_candidate, -max_id_volume_mw, max_id_volume_mw)
            x_candidates = [_clip(nom_da_for_candidate + forced_id_trade, x_lo, x_hi)]
        for x_cand in x_candidates:
            all_x.append(x_cand)
            all_d.append(d_cand)
            all_nom.append(nom_da_for_candidate)

    xs = np.stack(all_x)
    ds = np.stack(all_d)
    noms = np.stack(all_nom)
    values = _objective(xs, ds, noms, price_da, price_id, price_long, price_short, id_spread, risk_lambda, h)

    return _tie_break_select(values, xs, ds, noms, d_hi)


def solve_stage2(forecasts_id, nom_da, config):
    """Stage 2: intraday re-solve with nom_DA fixed (committed by stage 1).
    forecasts_id: dict with prod_fc_id, da_price_actual (known by now),
    id1_fc_id, imb_price_short_fc_id, imb_price_long_fc_id -- all arrays.
    Returns (id_trade, setpoint, x, d, planned_score) arrays.
    """
    x, d, score = solve_xd(
        nom_da=nom_da,
        d_hi=np.asarray(forecasts_id["prod_fc_id"], dtype=float),
        pos_min=config.pos_min_mw,
        pos_max=config.pos_max_mw,
        max_id_volume_mw=config.max_id_volume_mw,
        price_da=np.asarray(forecasts_id["da_price_actual"], dtype=float),
        price_id=np.asarray(forecasts_id["id1_fc_id"], dtype=float),
        price_long=np.asarray(forecasts_id["imb_price_long_fc_id"], dtype=float),
        price_short=np.asarray(forecasts_id["imb_price_short_fc_id"], dtype=float),
        id_spread=config.id_spread_eur_mwh,
        risk_lambda=config.risk_lambda_eur_mwh,
        h=0.25,
        allow_passive_balancing=config.allow_passive_balancing,
    )
    id_trade = x - nom_da
    return {"id_trade": id_trade, "setpoint": d, "x": x, "d": d}


def solve_stage1_skew_off(forecasts_d1, config):
    """Stage 1, allow_da_position_skew=False (the v1 default): nom_DA is
    slaved to d (see module docstring) -- reduces to the same (x, d) shape
    as stage 2, just with nom_da_equals_d=True and D-1 forecasts.
    """
    n = len(forecasts_d1["prod_fc_d1"])
    x, d, score = solve_xd(
        nom_da=np.zeros(n),  # unused when nom_da_equals_d=True
        d_hi=np.asarray(forecasts_d1["prod_fc_d1"], dtype=float),
        pos_min=config.pos_min_mw,
        pos_max=config.pos_max_mw,
        max_id_volume_mw=config.max_id_volume_mw,
        price_da=np.asarray(forecasts_d1["da_price_fc"], dtype=float),
        price_id=np.asarray(forecasts_d1["id1_fc_d1"], dtype=float),
        price_long=np.asarray(forecasts_d1["imb_price_long_fc_d1"], dtype=float),
        price_short=np.asarray(forecasts_d1["imb_price_short_fc_d1"], dtype=float),
        id_spread=config.id_spread_eur_mwh,
        risk_lambda=config.risk_lambda_eur_mwh,
        h=0.25,
        allow_passive_balancing=True,  # stage 1's ID trade is always a free anticipated decision
        nom_da_equals_d=True,
    )
    nom_da = d  # by construction
    planned_id_trade = x - nom_da
    return {"nom_da": nom_da, "planned_id_trade": planned_id_trade, "planned_setpoint": d}


def solve_stage1_skew_on(forecasts_d1, config):
    """Stage 1, allow_da_position_skew=True (sensitivity case): nom_DA
    becomes a third free variable. Outer loop over a small, fixed set of
    nom_DA candidates; for each, the inner (x, d) problem is exactly the
    stage-2 shape with that nom_DA held fixed. This is the one hand-derived
    candidate set NOT to trust without the oracle test (see
    test_policy_optimality.py) -- flagged explicitly in the brief.
    """
    prod_fc_d1 = np.asarray(forecasts_d1["prod_fc_d1"], dtype=float)
    n = len(prod_fc_d1)

    nom_min = np.full(n, config.nom_min_mw, dtype=float)
    nom_max = np.full(n, config.nom_max_mw, dtype=float)
    pos_min = np.full(n, config.pos_min_mw, dtype=float)
    pos_max = np.full(n, config.pos_max_mw, dtype=float)
    maxid = np.full(n, config.max_id_volume_mw, dtype=float)

    # Candidate nom_DA values: box ends, the D-1 production forecast
    # itself (the "balanced programme" reference point), and every value
    # where the ID-volume window [nom_DA-maxid, nom_DA+maxid] starts/stops
    # covering one of the other relevant breakpoints (0, prod_fc_d1,
    # pos_min, pos_max) -- these are exactly the points where the inner
    # (x, d) optimum can switch regime as nom_DA moves.
    d_hi = prod_fc_d1
    d_lo = np.zeros(n)
    breakpoints = [d_lo, d_hi, pos_min, pos_max]
    nom_da_candidates = [nom_min, nom_max, _clip(prod_fc_d1, nom_min, nom_max)]
    for bp in breakpoints:
        nom_da_candidates.append(_clip(bp - maxid, nom_min, nom_max))
        nom_da_candidates.append(_clip(bp + maxid, nom_min, nom_max))

    best_x = best_d = best_nom = best_val = None
    for nom_da_cand in nom_da_candidates:
        x, d, score = solve_xd(
            nom_da=nom_da_cand,
            d_hi=prod_fc_d1,
            pos_min=config.pos_min_mw,
            pos_max=config.pos_max_mw,
            max_id_volume_mw=config.max_id_volume_mw,
            price_da=np.asarray(forecasts_d1["da_price_fc"], dtype=float),
            price_id=np.asarray(forecasts_d1["id1_fc_d1"], dtype=float),
            price_long=np.asarray(forecasts_d1["imb_price_long_fc_d1"], dtype=float),
            price_short=np.asarray(forecasts_d1["imb_price_short_fc_d1"], dtype=float),
            id_spread=config.id_spread_eur_mwh,
            risk_lambda=config.risk_lambda_eur_mwh,
            h=0.25,
            allow_passive_balancing=True,
        )
        if best_val is None:
            best_x, best_d, best_nom, best_val = x, d, nom_da_cand, score
        else:
            better = score > best_val + TIE_TOL
            tied = np.abs(score - best_val) <= TIE_TOL
            # On ties, re-use the same priority as _tie_break_select would,
            # comparing this candidate's nom_DA-fixed inner solution
            # against the best-so-far on (curtailing, |id_trade|, |x-d|).
            if np.any(better) or np.any(tied):
                cand_curtail = (d < d_hi - TIE_TOL).astype(float)
                best_curtail = (best_d < d_hi - TIE_TOL).astype(float)
                cand_idtrade = np.abs(x - nom_da_cand)
                best_idtrade = np.abs(best_x - best_nom)
                cand_dev = np.abs(x - d)
                best_dev = np.abs(best_x - best_d)
                prefer_cand = tied & (
                    (cand_curtail < best_curtail)
                    | ((cand_curtail == best_curtail) & (cand_idtrade < best_idtrade))
                    | ((cand_curtail == best_curtail) & (cand_idtrade == best_idtrade) & (cand_dev < best_dev))
                )
                take = better | prefer_cand
                best_x = np.where(take, x, best_x)
                best_d = np.where(take, d, best_d)
                best_nom = np.where(take, nom_da_cand, best_nom)
                best_val = np.where(take, score, best_val)

    planned_id_trade = best_x - best_nom
    return {"nom_da": best_nom, "planned_id_trade": planned_id_trade, "planned_setpoint": best_d}


def solve_stage1(forecasts_d1, config):
    if config.allow_da_position_skew:
        return solve_stage1_skew_on(forecasts_d1, config)
    return solve_stage1_skew_off(forecasts_d1, config)
