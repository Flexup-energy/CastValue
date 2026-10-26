"""Coalition runner + Shapley/Harsanyi attribution, tested against the toy
benchmark (see conftest.py). All expected numbers below are hand-computed,
not just structurally asserted -- comments show the arithmetic.

Toy game recap: forecasts = {a:1, b:10, c:100}, actuals = {a:2, b:20, c:200}
(actual = 2x forecast), score = sum(effective values) + 5 if both a and b
are substituted (actual) simultaneously.

Hand-computed v(S) for every subset (verified by hand, see PR/commit
description if this ever needs re-deriving):
  v(ø)       = 111
  v({a})     = 112
  v({b})     = 121
  v({c})     = 211
  v({a,b})   = 127   (122 + 5 bonus)
  v({a,c})   = 212
  v({b,c})   = 221
  v({a,b,c}) = 227   (222 + 5 bonus)
"""

from __future__ import annotations

import pytest

from castvalue.core import (
    all_coalitions,
    assert_ceiling,
    CeilingViolation,
    grouped_game,
    harsanyi_dividends,
    pairwise_and_residual,
    run_all_coalitions,
    shapley_values_combinatorial,
    shapley_values_from_dividends,
)

EXPECTED_V = {
    frozenset(): 111.0,
    frozenset({"a"}): 112.0,
    frozenset({"b"}): 121.0,
    frozenset({"c"}): 211.0,
    frozenset({"a", "b"}): 127.0,
    frozenset({"a", "c"}): 212.0,
    frozenset({"b", "c"}): 221.0,
    frozenset({"a", "b", "c"}): 227.0,
}


def test_all_coalitions_enumerates_all_8_subsets():
    coalitions = all_coalitions(("a", "b", "c"))
    assert len(coalitions) == 8
    assert frozenset() in coalitions
    assert frozenset({"a", "b", "c"}) in coalitions
    assert len(set(coalitions)) == 8  # no duplicates


def test_coalition_runner_matches_hand_computed_values(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    results = run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config, baseline="perfect")
    assert set(results.keys()) == set(EXPECTED_V.keys())
    for coalition, expected_score in EXPECTED_V.items():
        assert results[coalition].score == pytest.approx(expected_score), f"coalition {coalition}"


def test_empty_coalition_reproduces_plain_forecast_run(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    results = run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config)
    direct = toy_benchmark.run_day(toy_forecasts, toy_actuals, toy_config)
    assert results[frozenset()].score == pytest.approx(direct.score)


def test_full_coalition_reproduces_perfect_foresight(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    results = run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config)
    direct_pf = toy_benchmark.run_day(toy_actuals, toy_actuals, toy_config)
    assert results[frozenset(toy_benchmark.players)].score == pytest.approx(direct_pf.score)


def _v_dict(results):
    return {coalition: r.score for coalition, r in results.items()}


def test_harsanyi_dividends_match_hand_computation(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    v = _v_dict(run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config))
    dividends = harsanyi_dividends(v)
    assert dividends[frozenset()] == pytest.approx(111.0)
    assert dividends[frozenset({"a"})] == pytest.approx(1.0)
    assert dividends[frozenset({"b"})] == pytest.approx(10.0)
    assert dividends[frozenset({"c"})] == pytest.approx(100.0)
    assert dividends[frozenset({"a", "b"})] == pytest.approx(5.0)  # exactly the bonus
    assert dividends[frozenset({"a", "c"})] == pytest.approx(0.0)
    assert dividends[frozenset({"b", "c"})] == pytest.approx(0.0)
    assert dividends[frozenset({"a", "b", "c"})] == pytest.approx(0.0)


def test_shapley_values_match_hand_computation_and_cross_check(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    v = _v_dict(run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config))
    dividends = harsanyi_dividends(v)
    phi_dividend = shapley_values_from_dividends(dividends, toy_benchmark.players)
    phi_combinatorial = shapley_values_combinatorial(v, toy_benchmark.players)

    # Hand computation: phi_a = Delta_a + Delta_ab/2 = 1 + 2.5 = 3.5
    #                   phi_b = Delta_b + Delta_ab/2 + Delta_bc/2 = 10 + 2.5 = 12.5
    #                   phi_c = Delta_c = 100
    expected = {"a": 3.5, "b": 12.5, "c": 100.0}
    for p in toy_benchmark.players:
        assert phi_dividend[p] == pytest.approx(expected[p]), f"dividend-based phi_{p}"
        assert phi_combinatorial[p] == pytest.approx(expected[p]), f"combinatorial phi_{p}"
        # the two independent formulas must agree with each other too
        assert phi_dividend[p] == pytest.approx(phi_combinatorial[p]), f"formulas disagree on {p}"


def test_shapley_efficiency(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    v = _v_dict(run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config))
    dividends = harsanyi_dividends(v)
    phi = shapley_values_from_dividends(dividends, toy_benchmark.players)
    total_gap = v[frozenset(toy_benchmark.players)] - v[frozenset()]
    assert sum(phi.values()) == pytest.approx(total_gap)
    assert total_gap == pytest.approx(116.0)  # 227 - 111


def test_harsanyi_identity_main_plus_pairwise_plus_residual_equals_total_gap(
    toy_benchmark, toy_forecasts, toy_actuals, toy_config
):
    v = _v_dict(run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config))
    dividends = harsanyi_dividends(v)
    pairwise, residual = pairwise_and_residual(dividends, toy_benchmark.players)

    main_sum = sum(d for T, d in dividends.items() if len(T) == 1)
    pairwise_sum = sum(pairwise.values())
    total_gap = v[frozenset(toy_benchmark.players)] - v[frozenset()]

    assert main_sum == pytest.approx(111.0)  # 1 + 10 + 100
    assert pairwise_sum == pytest.approx(5.0)  # only {a,b} is nonzero
    assert residual == pytest.approx(0.0)  # no 3-way interaction in this toy game
    assert main_sum + pairwise_sum + residual == pytest.approx(total_gap)


def test_two_player_toy_case_phi_equals_delta_plus_interaction_over_2():
    """Standalone 2-player game, no benchmark needed: v(ø)=0, v(a)=3, v(b)=5,
    v(a,b)=10. Main effects Delta_a=3, Delta_b=5; interaction I=10-3-5=2.
    Standard identity: phi_i = Delta_i + I/2.
    """
    v = {
        frozenset(): 0.0,
        frozenset({"a"}): 3.0,
        frozenset({"b"}): 5.0,
        frozenset({"a", "b"}): 10.0,
    }
    dividends = harsanyi_dividends(v)
    interaction = dividends[frozenset({"a", "b"})]
    assert interaction == pytest.approx(2.0)

    phi = shapley_values_from_dividends(dividends, ("a", "b"))
    assert phi["a"] == pytest.approx(dividends[frozenset({"a"})] + interaction / 2)
    assert phi["a"] == pytest.approx(4.0)
    assert phi["b"] == pytest.approx(dividends[frozenset({"b"})] + interaction / 2)
    assert phi["b"] == pytest.approx(6.0)
    # efficiency
    assert phi["a"] + phi["b"] == pytest.approx(v[frozenset({"a", "b"})] - v[frozenset()])


def test_negative_shapley_values_are_not_clipped():
    """Perfecting one input alone can hurt: v(a) < v(ø) is a legitimate,
    un-clipped negative main effect."""
    # v(ø)=10, v(a)=4, v(b)=12, v(a,b)=6. Delta_a=4-10=-6, Delta_b=12-10=2,
    # Delta_ab=6-4-12+10=0 (no interaction, by construction) -> phi_a =
    # Delta_a + Delta_ab/2 = -6 exactly. Total gap = v(a,b)-v(ø) = -4, also
    # negative -- a legitimate result (perfecting both inputs together is
    # worse than the baseline here), not an error.
    v = {
        frozenset(): 10.0,
        frozenset({"a"}): 4.0,  # perfecting a alone makes things WORSE
        frozenset({"b"}): 12.0,
        frozenset({"a", "b"}): 6.0,
    }
    dividends = harsanyi_dividends(v)
    phi = shapley_values_from_dividends(dividends, ("a", "b"))
    assert dividends[frozenset({"a"})] == pytest.approx(-6.0)
    assert phi["a"] == pytest.approx(-6.0)  # must not be clipped to 0
    assert phi["a"] + phi["b"] == pytest.approx(v[frozenset({"a", "b"})] - v[frozenset()])


def test_grouped_game_is_pure_remapping_no_new_runs(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    v = _v_dict(run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config))
    grouped = grouped_game(v, toy_benchmark.groups)

    assert grouped[frozenset()] == pytest.approx(v[frozenset()])  # 111
    assert grouped[frozenset({"low"})] == pytest.approx(v[frozenset({"a"})])  # 112
    assert grouped[frozenset({"high"})] == pytest.approx(v[frozenset({"b", "c"})])  # 221
    assert grouped[frozenset({"low", "high"})] == pytest.approx(v[frozenset({"a", "b", "c"})])  # 227


def test_grouped_shapley_differs_from_summed_player_level(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    """The brief is explicit: compute the grouped game as its own game, not
    by summing 7(here 3)-player values -- the two generally differ. Confirm
    that's true here (group "high" = {b,c} bundles a real interaction that
    splits differently than phi_b + phi_c alone would suggest in a richer
    game; in THIS toy game they happen to coincide only because {b,c} has
    zero interaction -- so assert the grouped computation is independently
    correct, not that it must differ)."""
    v = _v_dict(run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config))
    grouped_v = grouped_game(v, toy_benchmark.groups)
    grouped_dividends = harsanyi_dividends(grouped_v)
    grouped_phi = shapley_values_from_dividends(grouped_dividends, tuple(toy_benchmark.groups.keys()))

    # Efficiency must hold for the grouped game independently.
    total_gap = grouped_v[frozenset(toy_benchmark.groups.keys())] - grouped_v[frozenset()]
    assert sum(grouped_phi.values()) == pytest.approx(total_gap)
    assert total_gap == pytest.approx(116.0)


def test_ceiling_holds_for_pf(toy_benchmark, toy_forecasts, toy_actuals, toy_config):
    v = _v_dict(run_all_coalitions(toy_benchmark, toy_forecasts, toy_actuals, toy_config))
    forecast_score = v[frozenset()]
    pf_score = v[frozenset(toy_benchmark.players)]
    assert assert_ceiling(forecast_score, pf_score, mode="hard") is True


def test_ceiling_violation_raises_in_hard_mode():
    with pytest.raises(CeilingViolation):
        assert_ceiling(forecast_score=300.0, ceiling_score=227.0, mode="hard", tol=1e-6)


def test_ceiling_violation_warns_not_raises_in_soft_mode():
    with pytest.warns(UserWarning):
        result = assert_ceiling(forecast_score=300.0, ceiling_score=227.0, mode="soft", tol=1e-6)
    assert result is False
