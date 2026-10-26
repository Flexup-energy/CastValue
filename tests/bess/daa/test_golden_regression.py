"""Golden regression test for the BESS DAA restructure.

Guards the Milestone 1 move (root DAA_optimizer.py -> castvalue.bess.daa.optimizer):
replays the exact rolling-SOC backtest from DAA_DE_LU_Bench.ipynb Cell 1 against
master_energy_data_v2.csv, using the *moved* package, and checks per-day profit
and end-of-day SOC against the golden/daa_de_lu_golden.json fixture (generated
from the pre-move code). A mismatch means the move changed numerics -- which
Milestone 1 explicitly must not do.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from castvalue.bess.daa.optimizer import Optimizer

GLPSOL_PATH = shutil.which("glpsol") or "/opt/anaconda3/bin/glpsol"

GOLDEN_PATH = Path(__file__).parent / "golden" / "daa_de_lu_golden.json"
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "master_energy_data_v2.csv"

SCENARIO_COLUMNS = {
    "Persistence":      {"daa_f": "daa_persistence_forecast", "perfect": False},
    "Electricity Maps": {"daa_f": "daa_em_forecast",          "perfect": False},
    "Perfect":          {"daa_f": "daa_actuals",              "perfect": True},
}

PROFIT_TOL_EUR = 1e-6
SOC_TOL_MWH = 1e-6


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def df(golden):
    data = pd.read_csv(DATA_PATH)
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    num_days = golden["num_days"]
    data = data.iloc[: num_days * 96].reset_index(drop=True)
    data["day_index"] = [d for d in range(1, num_days + 1) for _ in range(96)]
    return data


@pytest.fixture(scope="module")
def optimizer():
    return Optimizer(GLPSOL_PATH)


@pytest.mark.parametrize("scenario_name", list(SCENARIO_COLUMNS.keys()))
def test_scenario_matches_golden(scenario_name, golden, df, optimizer):
    cols = SCENARIO_COLUMNS[scenario_name]
    params = golden["params"]
    expected_daily = golden["scenarios"][scenario_name]["daily"]
    num_days = golden["num_days"]

    carryover_soc = 0.0
    for d in range(1, num_days + 1):
        day_data = df[df["day_index"] == d]
        f_vec = {"daa": day_data[cols["daa_f"]].tolist()}
        a_vec = {"daa": day_data["daa_actuals"].tolist()}
        out = optimizer.run_day(
            f_vec, a_vec,
            params["power_cap_mw"], params["energy_cap_mwh"], params["n_cycles"],
            rte=params["rte"], initial_soc=carryover_soc,
            perfect_foresight=cols["perfect"],
        )
        expected = expected_daily[d - 1]
        assert out["profit_total"] == pytest.approx(expected["profit_total"], abs=PROFIT_TOL_EUR), (
            f"{scenario_name} day {d} ({expected['date']}): "
            f"profit {out['profit_total']} != golden {expected['profit_total']}"
        )
        assert out["end_soc"] == pytest.approx(expected["end_soc"], abs=SOC_TOL_MWH), (
            f"{scenario_name} day {d} ({expected['date']}): "
            f"end_soc {out['end_soc']} != golden {expected['end_soc']}"
        )
        carryover_soc = out["end_soc"]


def test_annual_margins_match_golden(golden, df, optimizer):
    params = golden["params"]
    num_days = golden["num_days"]
    for scenario_name, cols in SCENARIO_COLUMNS.items():
        carryover_soc = 0.0
        annual = 0.0
        for d in range(1, num_days + 1):
            day_data = df[df["day_index"] == d]
            f_vec = {"daa": day_data[cols["daa_f"]].tolist()}
            a_vec = {"daa": day_data["daa_actuals"].tolist()}
            out = optimizer.run_day(
                f_vec, a_vec,
                params["power_cap_mw"], params["energy_cap_mwh"], params["n_cycles"],
                rte=params["rte"], initial_soc=carryover_soc,
                perfect_foresight=cols["perfect"],
            )
            annual += out["profit_total"]
            carryover_soc = out["end_soc"]
        expected_annual = golden["scenarios"][scenario_name]["annual_margin"]
        assert annual == pytest.approx(expected_annual, abs=1e-4), (
            f"{scenario_name}: annual margin {annual} != golden {expected_annual}"
        )
