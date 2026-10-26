"""Spec section 4 + 7: Shapley efficiency and Harsanyi identity on the real
PV benchmark (not the toy benchmark) -- confirms the adapter feeds data
correctly into the already-proven-correct core attribution formulas
(tests/core/ already proves the formulas themselves are right).
"""

from __future__ import annotations

import pytest

from castvalue.core import (
    grouped_game,
    harsanyi_dividends,
    pairwise_and_residual,
    run_all_coalitions,
    shapley_values_combinatorial,
    shapley_values_from_dividends,
)
from castvalue.pv.imbalance import PvImbalanceBenchmark, PvImbalanceConfig

from .conftest import make_random_day


@pytest.fixture(scope="module")
def pv_game():
    forecasts, actuals = make_random_day(seed=7, capacity=5.0)
    config = PvImbalanceConfig(installed_capacity_mw=5.0)
    bm = PvImbalanceBenchmark()
    results = run_all_coalitions(bm, forecasts, actuals, config)
    v = {c: r.score for c, r in results.items()}
    return bm, v


def test_7_player_shapley_efficiency_and_cross_check(pv_game):
    bm, v = pv_game
    dividends = harsanyi_dividends(v)
    phi_dividend = shapley_values_from_dividends(dividends, bm.players)
    phi_combinatorial = shapley_values_combinatorial(v, bm.players)

    total_gap = v[frozenset(bm.players)] - v[frozenset()]
    assert sum(phi_dividend.values()) == pytest.approx(total_gap)
    for p in bm.players:
        assert phi_dividend[p] == pytest.approx(phi_combinatorial[p], abs=1e-6)


def test_harsanyi_identity_main_plus_pairwise_plus_residual(pv_game):
    bm, v = pv_game
    dividends = harsanyi_dividends(v)
    pairwise, residual = pairwise_and_residual(dividends, bm.players)

    main_sum = sum(d for T, d in dividends.items() if len(T) == 1)
    pairwise_sum = sum(pairwise.values())
    total_gap = v[frozenset(bm.players)] - v[frozenset()]
    assert main_sum + pairwise_sum + residual == pytest.approx(total_gap, abs=1e-6)


def test_4_player_grouped_game_efficiency_via_remapping_not_summed_7player(pv_game):
    bm, v = pv_game
    grouped_v = grouped_game(v, bm.groups)
    grouped_dividends = harsanyi_dividends(grouped_v)
    grouped_phi = shapley_values_from_dividends(grouped_dividends, tuple(bm.groups.keys()))

    total_gap = grouped_v[frozenset(bm.groups.keys())] - grouped_v[frozenset()]
    assert sum(grouped_phi.values()) == pytest.approx(total_gap)
    # Same total gap as the 7-player game (same empty/full coalitions),
    # but the grouped game is its own computation, not a sum of player-level phis.
    assert total_gap == pytest.approx(v[frozenset(bm.players)] - v[frozenset()])
