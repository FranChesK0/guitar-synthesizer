import numpy as np
import pytest  # noqa: F401
from numpy.testing import assert_almost_equal

from guitar_synth.temporal import Time
from guitar_synth.synthesis import Synthesis


# Test that vibrate generates the correct number of samples
def test_vibrate_output_shape() -> None:
    synthesis = Synthesis()
    frequency = 440  # A4
    duration = Time(2)
    sample_rate = synthesis.sample_rate

    output = synthesis.vibrate(frequency, duration)

    # The output should have the correct number of samples
    expected_samples = duration.get_num_samples(sample_rate)
    assert output.shape == (expected_samples,)


# Test that vibrate generates normalized samples
def test_vibrate_output_normalization() -> None:
    synthesis = Synthesis()
    frequency = 440  # A4
    duration = Time(2)

    output = synthesis.vibrate(frequency, duration)

    # The output should be normalized
    assert_almost_equal(np.abs(output).max(), 1.0, decimal=5)


# Test that vibrate removes DC offset
def test_vibrate_removes_dc() -> None:
    synthesis = Synthesis()
    frequency = 440  # A4
    duration = Time(2)

    output = synthesis.vibrate(frequency, duration)

    # The DC offset should be removed
    assert_almost_equal(output.mean(), 0.0, decimal=5)


# Test vibrate damping factor limits
@pytest.mark.parametrize("damping", [0.1, 0.5])
def test_vibrate_damping_limits(damping: float) -> None:
    synthesis = Synthesis()
    frequency = 440  # A4
    duration = Time(2)

    output = synthesis.vibrate(frequency, duration, damping=damping)

    # The output should be normalized
    assert_almost_equal(np.abs(output).max(), 1.0, decimal=5)
