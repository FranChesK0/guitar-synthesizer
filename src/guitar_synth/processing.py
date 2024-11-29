import numpy as np
from numpy.typing import NDArray


def remove_dc(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    return samples - np.float64(samples.mean())


def normalize(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    return samples / np.float64(np.abs(samples).max())
