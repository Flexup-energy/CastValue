"""The alpha-blend primitive shared by discrete coalition substitution
(alpha in {0, 1}) and continuous sensitivity curves (alpha swept across
[0, 1]) -- one implementation, two callers, so they can never drift apart.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

BaselineType = Literal["perfect", "naive"]


def alpha_blend(forecast_value: np.ndarray, substitute_value: np.ndarray, alpha: float) -> np.ndarray:
    """eff = substitute + alpha * (forecast - substitute).

    alpha=1 -> pure forecast (player untouched); alpha=0 -> pure substitute
    (player "perfected", i.e. replaced by its actual -- or, when
    baseline="naive", replaced by a naive proxy the caller supplies in
    place of the actual; this function doesn't care which).
    """
    forecast_value = np.asarray(forecast_value, dtype=float)
    substitute_value = np.asarray(substitute_value, dtype=float)
    return substitute_value + alpha * (forecast_value - substitute_value)
