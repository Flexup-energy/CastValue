"""DA / ID / imbalance settlement on actuals (spec section 2.3).

The risk penalty is NOT recomputed from actuals -- "deliberate deviation"
is defined against the stage's OWN forecast-implied expected delivered
(section 2.4), which is exactly stage 2's own committed (x, d) pair. Since
that penalty was already part of the objective policy.solve_stage2
selected against, settlement just reports the same `risk_lambda*|x-d|*h`
using the policy's committed values -- it is not a function of what
actually happened in the market.
"""

from __future__ import annotations

import numpy as np

from castvalue.core import INTERVAL_HOURS, resolve_imbalance_prices, settle_imbalance

from .inputs import DayActuals, PvImbalanceConfig


def settle_day(
    nom_da: np.ndarray,
    id_trade: np.ndarray,
    setpoint: np.ndarray,
    stage2_x: np.ndarray,
    stage2_d: np.ndarray,
    actuals: DayActuals,
    config: PvImbalanceConfig,
) -> dict:
    h = INTERVAL_HOURS
    prod_potential_actual = np.asarray(actuals.prod_potential_actual, dtype=float)
    da_price_actual = np.asarray(actuals.da_price_actual, dtype=float)
    id1_actual = np.asarray(actuals.id1_actual, dtype=float)

    delivered = np.minimum(prod_potential_actual, setpoint)

    da_revenue = nom_da * da_price_actual * h
    id_revenue = id_trade * id1_actual * h - np.abs(id_trade) * config.id_spread_eur_mwh * h

    x = nom_da + id_trade  # committed net position (should equal stage2_x)
    imb = delivered - x
    imb_long = np.maximum(imb, 0.0)
    imb_short = np.maximum(-imb, 0.0)

    if config.imbalance_pricing == "single":
        price_short, price_long = resolve_imbalance_prices(
            "single", single_price=actuals.imb_price_short_actual
        )
    else:
        price_short, price_long = resolve_imbalance_prices(
            config.imbalance_pricing,
            price_short=actuals.imb_price_short_actual,
            price_long=actuals.imb_price_long_actual,
        )
    imbalance_settlement = settle_imbalance(imb_long, imb_short, price_long, price_short, h)

    risk_penalty = config.risk_lambda_eur_mwh * np.abs(np.asarray(stage2_x) - np.asarray(stage2_d)) * h

    net_raw = da_revenue + id_revenue + imbalance_settlement
    net_risk_adjusted = net_raw - risk_penalty

    regulation_state = np.asarray(actuals.regulation_state_actual)
    regulation_state_2_share = float(np.mean(regulation_state == 2)) if regulation_state.size else 0.0

    return {
        "delivered": delivered,
        "da_revenue": da_revenue,
        "id_revenue": id_revenue,
        "imbalance_settlement": imbalance_settlement,
        "imb_long": imb_long,
        "imb_short": imb_short,
        "risk_penalty": risk_penalty,
        "net_raw": net_raw,
        "net_risk_adjusted": net_risk_adjusted,
        "net_raw_total": float(np.sum(net_raw)),
        "net_risk_adjusted_total": float(np.sum(net_risk_adjusted)),
        "regulation_state_2_share": regulation_state_2_share,
    }
