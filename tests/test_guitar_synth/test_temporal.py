from typing import Any
from decimal import Decimal
from fractions import Fraction
from contextlib import nullcontext as does_not_raise

import pytest

from guitar_synth.temporal import Time, Hertz, Numeric


# Test cases for the constructor
@pytest.mark.parametrize(
    "seconds, expected_seconds, expectation",
    [
        (1, Decimal("1"), does_not_raise()),
        (2, Decimal("2"), does_not_raise()),
        (Decimal("1.5"), Decimal("1.5"), does_not_raise()),
        (Fraction(3, 2), Decimal("1.5"), does_not_raise()),
        (
            "invalid",
            None,
            pytest.raises(TypeError, match="Unexpected type 'str' for seconds"),
        ),
    ],
)
def test_time_constructor(
    seconds: Numeric, expected_seconds: Numeric, expectation: Any
) -> None:
    with expectation:
        t = Time(seconds)
        assert t.seconds == expected_seconds


# Test cases for the form_milliseconds method
@pytest.mark.parametrize(
    "milliseconds, expected_seconds",
    [
        (1000, Decimal("1")),
        (1500, Decimal("1.5")),
        (2500, Decimal("2.5")),
        (500, Decimal("0.5")),
    ],
)
def test_time_from_milliseconds(milliseconds: Numeric, expected_seconds: Numeric) -> None:
    t = Time.from_milliseconds(milliseconds)
    assert t.seconds == expected_seconds


# Test cases for the get_num_samples method
@pytest.mark.parametrize(
    "seconds, rate, expected_samples",
    [
        (1, 44100, 44100),
        (2, 44100, 88200),
        (1.5, 44100, 66150),
        (0.5, 44100, 22050),
    ],
)
def test_time_get_num_samples(
    seconds: Numeric, rate: Hertz, expected_samples: int
) -> None:
    t = Time(seconds)
    assert t.get_num_samples(rate) == expected_samples
