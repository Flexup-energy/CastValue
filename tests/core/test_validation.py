from __future__ import annotations

import math

import pytest

from castvalue.core import reject_missing, validate_equal_length


def test_equal_length_passes_and_returns_length():
    n = validate_equal_length({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert n == 3


def test_length_mismatch_raises_naming_all_vectors():
    with pytest.raises(ValueError, match="a.*2.*b.*3|b.*3.*a.*2"):
        validate_equal_length({"a": [1, 2], "b": [1, 2, 3]})


def test_reject_missing_rejects_nan():
    with pytest.raises(ValueError, match=r"x\[1\]"):
        reject_missing({"x": [1.0, float("nan"), 3.0]})


def test_reject_missing_rejects_none():
    with pytest.raises(ValueError, match=r"x\[0\]"):
        reject_missing({"x": [None, 1.0]})


def test_reject_missing_passes_clean_data():
    reject_missing({"x": [1.0, 2.0], "y": [3.0, 4.0]})  # no raise
