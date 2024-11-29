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


# Test overlay output length
def test_overlay_output_length() -> None:
    synthesis = Synthesis()
    sample_rate = synthesis.sample_rate
    delay_time = Time(0.1)

    sound1 = np.ones(44100)  # 1 second
    sound2 = np.ones(22050) * 2  # 0.5 seconds
    sound3 = np.ones(11025) * 3  # 0.25 seconds

    sounds = [sound1, sound2, sound3]
    output = synthesis.overlay(sounds, delay_time)

    # Check the number of delay samples
    num_delay_samples = delay_time.get_num_samples(sample_rate)

    # Expected number of samples
    expected_num_samples = max(
        sound1.size,
        sound2.size + num_delay_samples,
        sound3.size + 2 * num_delay_samples,
    )
    assert output.size == expected_num_samples


# Test no overlap in non-overlapping areas
def test_overlay_no_overlap() -> None:
    synthesis = Synthesis()
    sample_rate = synthesis.sample_rate
    delay_time = Time(0.1)

    sound1 = np.ones(44100)  # 1 second
    sound2 = np.ones(22050) * 2  # 0.5 seconds

    sounds = [sound1, sound2]
    output = synthesis.overlay(sounds, delay_time)

    # Check the number of delay samples
    num_delay_samples = delay_time.get_num_samples(sample_rate)

    # Ensure there is no addition in the non-overlapping areas
    assert np.all(output[sound1.size : num_delay_samples] == 0)  # noqa: E203


# Test overlay with no delay
def test_overlay_no_delay() -> None:
    synthesis = Synthesis()
    delay_time = Time(0)

    sound1 = np.ones(44100)  # 1 second
    sound2 = np.ones(22050) * 2  # 0.5 seconds

    sounds = [sound1, sound2]
    output = synthesis.overlay(sounds, delay_time)

    # Expected number of samples
    expected_num_samples = max(sound1.size, sound2.size)
    assert output.size == expected_num_samples

    # Check that the sounds are summed together correctly
    assert_almost_equal(
        output[: sound2.size], sound1[: sound2.size] + sound2
    )  # noqa: E203
    assert_almost_equal(output[sound2.size :], sound1[sound2.size :])  # noqa: E203
