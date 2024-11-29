from typing import Final, Iterator
from itertools import cycle
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from guitar_synth.burst import WhiteNoise, BurstGenerator
from guitar_synth.temporal import Time, Hertz
from guitar_synth.processing import normalize, remove_dc

AUDIO_CD_SAMPLING_RATE: Final[int] = 44100


@dataclass(frozen=True)
class Synthesis:
    burst_generator: BurstGenerator = WhiteNoise()
    sample_rate: int = AUDIO_CD_SAMPLING_RATE

    def vibrate(
        self, frequency: Hertz, duration: Time, damping: float = 0.5
    ) -> NDArray[np.float64]:
        assert 0 < damping <= 0.5

        def feedback_loop() -> Iterator[float]:
            buffer = self.burst_generator(
                num_samples=round(self.sample_rate / frequency),
                sample_rate=self.sample_rate,
            )
            for i in cycle(range(buffer.size)):
                yield (current_sample := buffer[i])
                next_sample = buffer[(i + 1) % buffer.size]
                buffer[i] = (current_sample + next_sample) * damping

        return normalize(
            remove_dc(
                np.fromiter(
                    feedback_loop(),
                    np.float64,
                    duration.get_num_samples(self.sample_rate),
                )
            )
        )
