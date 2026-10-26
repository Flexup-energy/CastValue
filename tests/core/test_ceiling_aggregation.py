from __future__ import annotations

import pytest

from castvalue.core import percent_of_pf


def test_percent_of_pf_sums_before_dividing_not_average_of_ratios():
    # Day 1: forecast=1, pf=100 -> ratio 1%. Day 2: forecast=99, pf=100 -> ratio 99%.
    # Averaging the two daily ratios gives 50%. Summing first gives (1+99)/(100+100)=50%
    # too in this symmetric case -- use an asymmetric PF to actually distinguish them.
    numerators = [1.0, 99.0]
    denominators = [100.0, 300.0]  # asymmetric PF across the two days
    sum_then_divide = percent_of_pf(numerators, denominators)
    average_of_ratios = sum((n / d * 100) for n, d in zip(numerators, denominators)) / len(numerators)
    assert sum_then_divide == pytest.approx((1.0 + 99.0) / (100.0 + 300.0) * 100)
    assert sum_then_divide != pytest.approx(average_of_ratios)


def test_percent_of_pf_guards_nonpositive_denominator():
    assert percent_of_pf([10.0], [0.0]) is None
    assert percent_of_pf([10.0], [-5.0]) is None
