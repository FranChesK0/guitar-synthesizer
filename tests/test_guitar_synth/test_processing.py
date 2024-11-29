import numpy as np
import pytest  # noqa: F401

from guitar_synth.processing import normalize, remove_dc


# Test remove_dc function
def test_remove_dc() -> None:
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = remove_dc(samples)

    # The mean of the result should be 0
    assert np.isclose(result.mean(), 0.0)

    # Check if the DC offset is removed correctly
    expected = samples - samples.mean()
    assert np.allclose(result, expected)


# Test normalize function
def test_normalize() -> None:
    samples = np.array([1.0, -3.0, 2.0])
    result = normalize(samples)

    # The absolute maximum value of the result should be 1
    assert np.isclose(np.abs(result).max(), 1.0)

    # Check if normalization is applied correctly
    expected = samples / np.abs(samples).max()
    assert np.allclose(result, expected)
