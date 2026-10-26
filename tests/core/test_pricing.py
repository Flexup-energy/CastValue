from __future__ import annotations

import numpy as np
import pytest

from castvalue.core import resolve_imbalance_prices, settle_imbalance


def test_single_pricing_broadcasts_to_both_sides():
    price_short, price_long = resolve_imbalance_prices("single", single_price=[50.0, 60.0])
    assert np.allclose(price_short, [50.0, 60.0])
    assert np.allclose(price_long, [50.0, 60.0])


def test_single_pricing_matches_plain_volume_times_price():
    price_short, price_long = resolve_imbalance_prices("single", single_price=[50.0])
    imb = 3.0  # net long position
    settlement = settle_imbalance(
        imb_long=np.array([max(imb, 0)]),
        imb_short=np.array([max(-imb, 0)]),
        price_long=price_long,
        price_short=price_short,
        interval_hours=0.25,
    )
    assert settlement[0] == pytest.approx(imb * 50.0 * 0.25)


def test_nl_tennet_dual_pricing_short_costs_more_than_single_equivalent():
    # Regulation state 2: shortage price > surplus price.
    price_short, price_long = resolve_imbalance_prices("nl_tennet", price_short=[100.0], price_long=[40.0])
    short_position = 2.0  # being short 2 MWh
    dual_settlement = settle_imbalance(
        imb_long=np.array([0.0]), imb_short=np.array([short_position]),
        price_long=price_long, price_short=price_short, interval_hours=0.25,
    )[0]
    # single-price equivalent using the average of short/long as a stand-in "single" price
    single_price, _ = resolve_imbalance_prices("single", single_price=[70.0])
    single_settlement = settle_imbalance(
        imb_long=np.array([0.0]), imb_short=np.array([short_position]),
        price_long=single_price, price_short=single_price, interval_hours=0.25,
    )[0]
    assert dual_settlement < single_settlement  # paying the higher shortage price costs more (settlement is negative)


def test_nl_tennet_states_0_plus1_minus1_match_single_pricing_exactly():
    # When shortage == surplus (regulation state 0/+-1), nl_tennet must
    # reduce to exactly the single-pricing formula, to the cent.
    price_short, price_long = resolve_imbalance_prices("nl_tennet", price_short=[55.0], price_long=[55.0])
    single_price, _ = resolve_imbalance_prices("single", single_price=[55.0])
    for imb_long, imb_short in [(2.0, 0.0), (0.0, 3.0), (0.0, 0.0)]:
        dual = settle_imbalance(np.array([imb_long]), np.array([imb_short]), price_long, price_short, 0.25)
        single = settle_imbalance(np.array([imb_long]), np.array([imb_short]), single_price, single_price, 0.25)
        assert dual[0] == pytest.approx(single[0])


def test_unknown_scheme_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        resolve_imbalance_prices("dual_continuous")
