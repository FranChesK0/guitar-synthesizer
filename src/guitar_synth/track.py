from decimal import Decimal

import numpy as np
from numpy.typing import NDArray

from guitar_synth.temporal import Time, Hertz


class AudioTrack:
    def __init__(self, sampling_rate: Hertz) -> None:
        self.sampling_rate = sampling_rate
        self.samples: NDArray[np.float64] = np.array([], dtype=np.float64)

    def __len__(self) -> int:
        return self.samples.size

    @property
    def duration(self) -> Time:
        return Time(len(self) / self.sampling_rate)

    def add(self, samples: NDArray[np.float64]) -> None:
        self.samples = np.append(self.samples, samples)

    def add_at(self, instant: Time, samples: NDArray[np.float64]) -> None:
        samples_offset = round(
            Decimal(str(float(instant.seconds))) * Decimal(self.sampling_rate)
        )
        if samples_offset == len(self):
            self.add(samples)
        elif samples_offset > len(self):
            self.add(np.zeros(samples_offset - len(self)))
            self.add(samples)
        else:
            end = samples_offset + len(samples)
            if end > len(self):
                self.add(np.zeros(end - len(self)))
            self.samples[samples_offset:end] += samples
