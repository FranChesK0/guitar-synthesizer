from typing import Any
from contextlib import nullcontext as does_not_raise

import pytest

from guitar_synth.pitch import Pitch
from guitar_synth.temporal import Hertz


# Test from_scientific_notation
@pytest.mark.parametrize(
    "notation, expected_frequency, expectation",
    [
        ("A4", 440.0, does_not_raise()),
        ("C4", 261.625565, does_not_raise()),
        ("C#5", 554.365261, does_not_raise()),
        ("B7", 3951.066409, does_not_raise()),
        ("H4", 0, pytest.raises(ValueError, match="Invalid pitch notation: 'H4'")),
        ("Z5", 0, pytest.raises(ValueError, match="Invalid pitch notation: 'Z5'")),
    ],
)
def test_from_scientific_notation(
    notation: str, expected_frequency: Hertz, expectation: Any
) -> None:
    with expectation:
        pitch = Pitch.from_scientific_notation(notation)
        assert abs(pitch.frequency - expected_frequency) < 1e-3
