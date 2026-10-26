from __future__ import annotations

import pytest

from castvalue.core import alpha_blend, sensitivity_curve


def test_alpha_blend_endpoints():
    assert alpha_blend(forecast_value=10.0, substitute_value=2.0, alpha=1.0) == pytest.approx(10.0)
    assert alpha_blend(forecast_value=10.0, substitute_value=2.0, alpha=0.0) == pytest.approx(2.0)


def test_alpha_blend_is_linear_interpolation():
    assert alpha_blend(forecast_value=10.0, substitute_value=0.0, alpha=0.5) == pytest.approx(5.0)


def test_sensitivity_curve_reports_points_and_slope():
    # score = 100 * alpha (a simple linear run_fn) -> slope should be exactly 100
    result = sensitivity_curve(run_fn=lambda a: 100.0 * a, alphas=(0.0, 0.5, 1.0))
    assert len(result["points"]) == 3
    assert result["points"][-1]["alpha"] == 1.0
    assert result["points"][-1]["score"] == pytest.approx(100.0)
    assert result["slope_near_1"] == pytest.approx(100.0)


def test_sensitivity_curve_single_alpha_has_no_slope():
    result = sensitivity_curve(run_fn=lambda a: a, alphas=(1.0,))
    assert result["slope_near_1"] is None
