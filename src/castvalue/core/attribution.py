"""Shapley value / Harsanyi dividend attribution over a coalition game
v: frozenset[str] -> float.

Fully generic -- works on any coalition-value dict keyed by frozenset,
whether it came from a 7-player PV game, a 4-player grouped game, a toy
benchmark, or anything else. Nothing here knows about forecasts, prices,
or benchmarks at all.
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping


def harsanyi_dividends(v: Mapping[frozenset, float]) -> dict[frozenset, float]:
    """Mobius transform: Delta(T) = sum_{R subseteq T} (-1)^(|T|-|R|) v(R).

    Anchored at v(emptyset) (the all-forecast baseline run) -- the standard
    definition of a Harsanyi decomposition, and the only anchor under which
    the two-player identity phi = Delta_i + I/2 (tested below) holds.

    `v` must already contain every subset of every key's own player set
    (true for a full 2^n coalition run; also true for the subset of a
    grouped game, since each group-level coalition's dividend only needs
    subsets of the *groups* it's built from, each of which is itself a key
    in the grouped game dict passed in).
    """
    dividends: dict[frozenset, float] = {}
    for T in v.keys():
        T_list = list(T)
        total = 0.0
        for r in range(len(T_list) + 1):
            for R_tuple in itertools.combinations(T_list, r):
                R = frozenset(R_tuple)
                sign = (-1) ** (len(T) - len(R))
                total += sign * v[R]
        dividends[T] = total
    return dividends


def shapley_values_from_dividends(
    dividends: Mapping[frozenset, float], players: tuple[str, ...]
) -> dict[str, float]:
    """phi_i = sum_{T containing i} Delta(T) / |T|."""
    phi = {p: 0.0 for p in players}
    for T, d in dividends.items():
        if len(T) == 0:
            continue
        share = d / len(T)
        for p in T:
            if p in phi:
                phi[p] += share
    return phi


def shapley_values_combinatorial(
    v: Mapping[frozenset, float], players: tuple[str, ...]
) -> dict[str, float]:
    """The direct weighted-sum Shapley formula, independent of the
    dividend construction above -- cross-checked against it in tests as
    cheap insurance on a subtle formula.

    phi_i = sum_{S subseteq N\\{i}} [|S|!(n-|S|-1)!/n!] * (v(S union {i}) - v(S))
    """
    n = len(players)
    phi: dict[str, float] = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for r in range(len(others) + 1):
            weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for combo in itertools.combinations(others, r):
                S = frozenset(combo)
                total += weight * (v[S | {p}] - v[S])
        phi[p] = total
    return phi


def pairwise_and_residual(
    dividends: Mapping[frozenset, float], players: tuple[str, ...]
) -> tuple[dict[frozenset, float], float]:
    """Split the non-empty dividends into main effects (singletons, not
    returned separately -- they're `shapley`-adjacent and the caller
    already has them via dividends), pairwise interactions (returned), and
    a single higher-order residual (everything |T|>=3, collapsed to one
    number per the spec, rather than all 2^n-n-C(n,2)-1 terms).

    Test invariant (Harsanyi identity): sum(singles) + sum(pairwise) +
    higher_order_residual == total_gap == v(N) - v(emptyset) == sum of
    every non-empty dividend.
    """
    singles_sum = sum(d for T, d in dividends.items() if len(T) == 1)
    pairwise: dict[frozenset, float] = {}
    for a, b in itertools.combinations(players, 2):
        pairwise[frozenset((a, b))] = dividends.get(frozenset((a, b)), 0.0)
    pairwise_sum = sum(pairwise.values())
    all_nonempty_sum = sum(d for T, d in dividends.items() if len(T) > 0)
    higher_order_residual = all_nonempty_sum - singles_sum - pairwise_sum
    return pairwise, higher_order_residual


def grouped_game(
    v: Mapping[frozenset[str], float], groups: Mapping[str, tuple[str, ...]]
) -> dict[frozenset[str], float]:
    """Remap a player-level coalition game onto a group-level game by pure
    dict lookup -- no new runs. Every group-level coalition's underlying
    player set must already be a key in `v` (true whenever `v` is a full
    2^n coalition run over the players covered by `groups`).
    """
    group_names = tuple(groups.keys())
    result: dict[frozenset[str], float] = {}
    for r in range(len(group_names) + 1):
        for combo in itertools.combinations(group_names, r):
            group_coalition = frozenset(combo)
            underlying = frozenset(p for g in combo for p in groups[g])
            result[group_coalition] = v[underlying]
    return result


def aggregate(per_item_values: list[Mapping]) -> dict:
    """Sum a list of dicts (coalition values, dividends, or Shapley values
    -- anything keyed consistently) element-wise, e.g. across days or
    months. Ratio computation (e.g. "% of PF") must happen *after*
    aggregating sums, never by averaging per-item ratios.
    """
    total: dict = {}
    for item in per_item_values:
        for key, value in item.items():
            total[key] = total.get(key, 0.0) + value
    return total
