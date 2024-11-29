from typing import Final, Iterator, Sequence
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

    def overlay(
        self, sounds: Sequence[NDArray[np.float64]], delay: Time
    ) -> NDArray[np.float64]:
        num_delay_samples = delay.get_num_samples(self.sample_rate)
        num_samples = max(
            i * num_delay_samples + sound.size for i, sound in enumerate(sounds)
        )
        samples = np.zeros(num_samples, dtype=np.float64)
        for i, sound in enumerate(sounds):
            offset = i * num_delay_samples
            samples[offset : offset + sound.size] += sound  # noqa: E203
        return samples
