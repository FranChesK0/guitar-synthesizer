import numpy as np
import pytest  # noqa: F401

from guitar_synth.burst import WhiteNoise


# Test WhiteNoise class
def test_white_noise_output_shape() -> None:
    num_samples = 100
    sample_rate = 44100
    generator = WhiteNoise()
    output = generator(num_samples, sample_rate)

    # Check if the output is a numpy array
    assert isinstance(output, np.ndarray)

    # Check if the output length matches the number of samples
    assert len(output) == num_samples
    # Check if the output shape is (num_samples,)
    assert output.shape == (num_samples,)


def test_white_noise_output_range() -> None:
    num_samples = 100
    sample_rate = 44100
    generator = WhiteNoise()
    output = generator(num_samples, sample_rate)

    # Check if the output values are between -1 and 1
    assert np.all(output >= -1.0)
    assert np.all(output <= 1.0)
