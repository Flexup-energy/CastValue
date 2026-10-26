"""DayForecasts / DayActuals / PolicyConfig for the PV imbalance benchmark."""

from __future__ import annotations

import dataclasses

from castvalue.core import PolicyConfig, reject_missing, validate_equal_length


@dataclasses.dataclass(frozen=True)
class DayForecasts:
    """The 7-player forecast vectors for one delivery day. All vectors must
    be the same length N -- 96 on an ordinary day, 92/100 on the two
    annual Berlin DST transition days; never hardcode 96."""

    prod_fc_d1: list
    da_price_fc: list
    id1_fc_d1: list
    imb_price_short_fc_d1: list
    imb_price_long_fc_d1: list
    prod_fc_id: list
    id1_fc_id: list
    imb_price_short_fc_id: list
    imb_price_long_fc_id: list

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DayActuals:
    prod_potential_actual: list
    da_price_actual: list
    id1_actual: list
    imb_price_short_actual: list
    imb_price_long_actual: list
    regulation_state_actual: list
    # Naive substitutes (persistence, climatology, previous interval, ...)
    # for baseline="naive" coalition runs -- keyed by DayForecasts field
    # name (see players.PLAYER_FIELDS), e.g. {"prod_fc_d1": [...]}. This
    # package is stateless/per-day (section 0), so it cannot derive a
    # naive baseline itself (no cross-day history) -- the caller supplies
    # it. Optional: only required when baseline="naive" is actually used.
    naive: dict | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if k != "naive"}


@dataclasses.dataclass(frozen=True)
class PvImbalanceConfig(PolicyConfig):
    installed_capacity_mw: float = 1.0
    nom_min_mw: float = 0.0
    nom_max_mw: float | None = None  # resolves to installed_capacity_mw
    max_id_volume_mw: float | None = None  # resolves to installed_capacity_mw (non-binding in v1)
    id_spread_eur_mwh: float = 0.0  # frictionless in v1
    pos_min_mw: float | None = None  # resolves to nom_min_mw
    pos_max_mw: float | None = None  # resolves to nom_max_mw
    allow_passive_balancing: bool = True
    max_passive_deviation_mw: float | None = None  # resolves to installed_capacity_mw
    allow_da_position_skew: bool = False
    risk_lambda_eur_mwh: float = 0.0  # v1: no risk damping
    imbalance_pricing: str = "nl_tennet"
    id_lead_time_h: float = 1.0

    def __post_init__(self):
        if self.nom_max_mw is None:
            object.__setattr__(self, "nom_max_mw", self.installed_capacity_mw)
        if self.max_id_volume_mw is None:
            object.__setattr__(self, "max_id_volume_mw", self.installed_capacity_mw)
        if self.pos_min_mw is None:
            object.__setattr__(self, "pos_min_mw", self.nom_min_mw)
        if self.pos_max_mw is None:
            object.__setattr__(self, "pos_max_mw", self.nom_max_mw)
        if self.max_passive_deviation_mw is None:
            object.__setattr__(self, "max_passive_deviation_mw", self.installed_capacity_mw)


def validate_day(forecasts: DayForecasts, actuals: DayActuals) -> int:
    """Validate every forecast + actual vector has equal length and no
    missing values. Returns N, the day's interval count."""
    named = {**forecasts.as_dict(), **actuals.as_dict()}
    n = validate_equal_length(named)
    reject_missing(named)
    return n
