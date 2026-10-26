"""Player taxonomy for the PV imbalance benchmark's Shapley game.

Each imbalance price *player* is a shortage/surplus price *pair* -- one
substitutable unit, not two separate players (per the brief's framework
section). PLAYER_FIELDS and ACTUAL_FOR_FIELD are what let the generic
core.coalitions/core.baselines machinery substitute at the field level
without core itself knowing anything about PV-specific field names.
"""

from __future__ import annotations

PLAYERS: tuple[str, ...] = (
    "prod_fc_d1",
    "da_price_fc",
    "id1_fc_d1",
    "imb_fc_d1",
    "prod_fc_id",
    "id1_fc_id",
    "imb_fc_id",
)

GROUPS: dict[str, tuple[str, ...]] = {
    "production": ("prod_fc_d1", "prod_fc_id"),
    "da_price": ("da_price_fc",),
    "id1_price": ("id1_fc_d1", "id1_fc_id"),
    "imbalance_price": ("imb_fc_d1", "imb_fc_id"),
}

# Each player maps to the DayForecasts field name(s) it controls.
PLAYER_FIELDS: dict[str, tuple[str, ...]] = {
    "prod_fc_d1": ("prod_fc_d1",),
    "da_price_fc": ("da_price_fc",),
    "id1_fc_d1": ("id1_fc_d1",),
    "imb_fc_d1": ("imb_price_short_fc_d1", "imb_price_long_fc_d1"),
    "prod_fc_id": ("prod_fc_id",),
    "id1_fc_id": ("id1_fc_id",),
    "imb_fc_id": ("imb_price_short_fc_id", "imb_price_long_fc_id"),
}

# Each DayForecasts field name maps to the DayActuals field name that
# substitutes it under baseline="perfect" (i.e. both forecast vintages of
# a given underlying quantity substitute to the SAME actual).
ACTUAL_FOR_FIELD: dict[str, str] = {
    "prod_fc_d1": "prod_potential_actual",
    "prod_fc_id": "prod_potential_actual",
    "da_price_fc": "da_price_actual",
    "id1_fc_d1": "id1_actual",
    "id1_fc_id": "id1_actual",
    "imb_price_short_fc_d1": "imb_price_short_actual",
    "imb_price_long_fc_d1": "imb_price_long_actual",
    "imb_price_short_fc_id": "imb_price_short_actual",
    "imb_price_long_fc_id": "imb_price_long_actual",
}
