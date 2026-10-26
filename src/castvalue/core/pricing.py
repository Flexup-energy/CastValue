"""Imbalance settlement schemes, shared by any benchmark that settles an
imbalance volume against a price (PV imbalance today; BESS multi-market
later, per its planned-scope README).

Key design point: settlement is ONE formula regardless of scheme. Single
pricing is just the degenerate case `price_long == price_short` of dual
pricing -- so `settle_imbalance` never branches on which scheme produced
its price arrays. `regulation_state` is carried through callers for
diagnostics/reporting only (e.g. "share of intervals in regulation state
2"), never used to pick a formula.
"""

from __future__ import annotations

import numpy as np

ImbalancePricingScheme = str  # "single" | "nl_tennet" (open for more later)


def settle_imbalance(
    imb_long: np.ndarray,
    imb_short: np.ndarray,
    price_long: np.ndarray,
    price_short: np.ndarray,
    interval_hours: float,
) -> np.ndarray:
    """Per-interval imbalance settlement cash flow (positive = income).

    imb_long/imb_short are the non-negative long/short split of the
    imbalance volume (imb_long = max(imb, 0), imb_short = max(-imb, 0)).
    Under single pricing (price_long == price_short == p) this reduces
    algebraically to `imb * p * h`, so no separate single-pricing code
    path is needed.
    """
    return imb_long * price_long * interval_hours - imb_short * price_short * interval_hours


def resolve_imbalance_prices(
    scheme: ImbalancePricingScheme,
    *,
    single_price: np.ndarray | None = None,
    price_short: np.ndarray | None = None,
    price_long: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shape a scheme's raw inputs into the (price_short, price_long) pair
    `settle_imbalance` expects.

    "single": one published price per interval, applies regardless of
        imbalance direction -- broadcast it to both sides.
    "nl_tennet": TenneT's NL scheme. In regulation states 0/+-1 the
        published shortage and surplus prices are already equal (single
        pricing is a special case of this, not a separate formula); in
        state 2 they genuinely differ (shortage >= surplus). Always use
        the published pair as-is -- never infer one price from the other
        or from `regulation_state`.
    """
    if scheme == "single":
        if single_price is None:
            raise ValueError('imbalance_pricing="single" requires `single_price`')
        p = np.asarray(single_price, dtype=float)
        return p, p
    elif scheme == "nl_tennet":
        if price_short is None or price_long is None:
            raise ValueError('imbalance_pricing="nl_tennet" requires `price_short` and `price_long`')
        return np.asarray(price_short, dtype=float), np.asarray(price_long, dtype=float)
    else:
        raise NotImplementedError(
            f"imbalance_pricing={scheme!r} not supported yet — "
            'only "single" and "nl_tennet" are implemented.'
        )
