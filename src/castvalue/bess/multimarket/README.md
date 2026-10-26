# BESS multi-market benchmark — planned scope

Not implemented. Importing `castvalue.bess.multimarket` raises `NotImplementedError`.

## Planned scope

- DA + intraday + imbalance (and later possibly aFRR) co-optimization for battery storage, rather than the DA-only scope of `castvalue.bess.daa`.
- Players: DA price, ID1 price, imbalance price pair, each per forecast vintage — the same player taxonomy as `castvalue.pv.imbalance`, since it's the same underlying market structure applied to a different asset.
- A genuine multi-period LP, not vertex enumeration — battery state of charge couples every interval to the next, so (unlike PV imbalance) the per-interval-independence assumption does not hold here.
- Reuses `castvalue.core.pricing` (`nl_tennet` imbalance settlement) and `castvalue.core.attribution` (Shapley/Harsanyi) unchanged — only the per-day solve is benchmark-specific.

No design work in `castvalue.core` should assume this benchmark can't exist — `run_day()` is opaque to core regardless of whether the benchmark solves per-interval or as one multi-period LP.
