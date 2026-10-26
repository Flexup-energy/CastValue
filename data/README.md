# Data

## `master_energy_data_v2.csv`

German/Luxembourg (DE-LU) EPEX SPOT Day-Ahead Auction prices at 15-minute resolution, 2025-10-18 to 2026-02-22 (126 complete days), used by `examples/DAA_DE_LU_Bench.ipynb` and `tests/bess/daa/golden/`.

| Column | Type | Description |
|---|---|---|
| `timestamp_utc` | ISO 8601 UTC timestamp | Start of the 15-minute delivery interval |
| `daa_actuals` | float, EUR/MWh | Realized (cleared) DE-LU day-ahead auction price |
| `daa_persistence_forecast` | float, EUR/MWh | Naive baseline: the actual price from the same quarter-hour one day earlier |
| `daa_em_forecast` | float, EUR/MWh | Electricity Maps' day-ahead price forecast |

12,186 rows (126 days x 96 intervals/day). Sourced from public EPEX SPOT auction results and a commercial Electricity Maps forecast feed; cleared for inclusion in this public repository.

Licence: see the repository root `LICENSE` (GPLv3) — this applies to the code; this data file is included as a fixed benchmark dataset for reproducing the results in `README.md` and the methodology doc under `docs/methodology/`.
