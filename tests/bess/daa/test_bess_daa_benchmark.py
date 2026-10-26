"""Tests for the BessDaaBenchmark adapter -- structural wiring, the
perfect_foresight flag's correctness (not just inferred), and a real-data
ceiling check that turns the Milestone 2 investigation (0/252 violations
on the golden backtest) into a permanent regression test, now going
through castvalue.core's actual coalition/ceiling machinery rather than a
one-off script.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from castvalue.bess.daa import BessDaaBenchmark, DaaConfig, Optimizer, cast_value
from castvalue.core import assert_ceiling, run_all_coalitions

GLPSOL_PATH = shutil.which("glpsol") or "/opt/anaconda3/bin/glpsol"
GOLDEN_PATH = Path(__file__).parent / "golden" / "daa_de_lu_golden.json"
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "master_energy_data_v2.csv"


@pytest.fixture
def benchmark():
    return BessDaaBenchmark()


@pytest.fixture
def config():
    return DaaConfig(glpk_path=GLPSOL_PATH)


def test_benchmark_structural_attributes(benchmark):
    assert benchmark.name == "bess-daa"
    assert benchmark.METHOD_VERSION == "1.1"
    assert benchmark.players == ("da_price",)
    assert benchmark.groups is None
    assert benchmark.ceiling_mode == "hard"


def test_substitute_empty_coalition_keeps_forecast(benchmark):
    forecasts = {"daa": [10.0, 20.0]}
    actuals = {"daa": [15.0, 25.0]}
    result = benchmark.substitute(forecasts, actuals, frozenset(), baseline="perfect")
    assert result["daa"] == [10.0, 20.0]
    assert result["perfect_foresight"] is False


def test_substitute_full_coalition_perfect_uses_actuals(benchmark):
    forecasts = {"daa": [10.0, 20.0]}
    actuals = {"daa": [15.0, 25.0]}
    result = benchmark.substitute(forecasts, actuals, frozenset({"da_price"}), baseline="perfect")
    assert result["daa"] == [15.0, 25.0]
    assert result["perfect_foresight"] is True


def test_substitute_full_coalition_naive_uses_naive_proxy_not_perfect_foresight(benchmark):
    forecasts = {"daa": [10.0, 20.0]}
    actuals = {"daa": [15.0, 25.0], "daa_naive": [12.0, 22.0]}
    result = benchmark.substitute(forecasts, actuals, frozenset({"da_price"}), baseline="naive")
    assert result["daa"] == [12.0, 22.0]
    # Critical: naive substitution is still just another imperfect forecast,
    # must go through normal OTM/bidding -- never treated as true foresight.
    assert result["perfect_foresight"] is False


def test_substitute_naive_without_proxy_raises(benchmark):
    with pytest.raises(ValueError, match="daa_naive"):
        benchmark.substitute({"daa": [1.0]}, {"daa": [2.0]}, frozenset({"da_price"}), baseline="naive")


def test_run_day_full_coalition_matches_optimizer_directly(benchmark, config):
    forecasts = {"daa": [50.0] * 96}
    actuals = {"daa": [50.0 + i * 0.1 for i in range(96)]}
    substituted = benchmark.substitute(forecasts, actuals, frozenset({"da_price"}), baseline="perfect")
    result = benchmark.run_day(substituted, actuals, config)

    opt = Optimizer(config.glpk_path)
    direct = opt.run_day(
        forecasts={"daa": actuals["daa"]}, actuals={"daa": actuals["daa"]},
        power_cap=config.power_cap_mw, energy_cap=config.energy_cap_mwh, n_cycles=config.n_cycles,
        rte=config.rte, initial_soc=config.initial_soc, perfect_foresight=True,
    )
    assert result.score == pytest.approx(direct["profit_total"])


def test_cast_value_is_score_difference(benchmark, config):
    forecast_result = benchmark.run_day({"daa": [50.0] * 96, "perfect_foresight": False}, {"daa": [55.0] * 96}, config)
    naive_result = benchmark.run_day({"daa": [52.0] * 96, "perfect_foresight": False}, {"daa": [55.0] * 96}, config)
    assert cast_value(forecast_result, naive_result) == pytest.approx(forecast_result.score - naive_result.score)


@pytest.fixture(scope="module")
def golden_and_data():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    df = pd.read_csv(DATA_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    num_days = golden["num_days"]
    df = df.iloc[: num_days * 96].reset_index(drop=True)
    df["day_index"] = [d for d in range(1, num_days + 1) for _ in range(96)]
    return golden, df


@pytest.mark.parametrize("forecast_column", ["daa_persistence_forecast", "daa_em_forecast"])
def test_real_data_never_violates_pf_ceiling_through_core_machinery(benchmark, config, golden_and_data, forecast_column):
    """The Milestone 2 investigation as a permanent regression test: for
    every real day in the golden dataset, with a SHARED initial_soc across
    coalitions (the condition the hard-ceiling argument requires), run the
    forecast coalition and the full (PF) coalition through
    castvalue.core.run_all_coalitions and assert_ceiling -- not a one-off
    script, the actual production code path.
    """
    golden, df = golden_and_data
    num_days = golden["num_days"]
    for d in range(1, num_days + 1):
        day_data = df[df["day_index"] == d]
        forecasts = {"daa": day_data[forecast_column].tolist()}
        actuals = {"daa": day_data["daa_actuals"].tolist()}
        results = run_all_coalitions(benchmark, forecasts, actuals, config, baseline="perfect")
        forecast_score = results[frozenset()].score
        pf_score = results[frozenset({"da_price"})].score
        assert assert_ceiling(forecast_score, pf_score, mode=benchmark.ceiling_mode, tol=1e-6, context=f"day {d}")
