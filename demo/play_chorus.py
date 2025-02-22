import os
import sys
from typing import Tuple, Iterator
from itertools import cycle

from pedalboard import (  # type: ignore[attr-defined]
    Gain,
    Reverb,
    Pedalboard,
    Convolution,
    LowShelfFilter,
)
from pedalboard.io import AudioFile  # type: ignore[attr-defined]

root_dir = os.path.abspath(os.path.dirname(__file__)).removesuffix("demo")
python_path = os.path.join(root_dir, "src")
if python_path not in sys.path:
    sys.path.append(python_path)

from guitar_synth.chord import Chord  # noqa: E402
from guitar_synth.track import AudioTrack  # noqa: E402
from guitar_synth.stroke import Velocity  # noqa: E402
from guitar_synth.temporal import Time, TimeLine  # noqa: E402
from guitar_synth.synthesis import Synthesizer  # noqa: E402
from guitar_synth.instrument import StringTuning, PluckedStringInstrument  # noqa: E402
from guitar_synth.processing import normalize  # noqa: E402


def main() -> None:
    ukulele = PluckedStringInstrument(
        tuning=StringTuning.from_notes("A4", "E4", "C4", "G4"),
        vibration=Time(5.0),
        damping=0.498,
    )
    synthesizer = Synthesizer(instrument=ukulele)
    audio_track = AudioTrack(synthesizer.sample_rate)
    timeline = TimeLine()
    for interval, chord, stroke in strumming_pattern():
        audio_samples = synthesizer.strum_strings(chord, stroke)
        audio_track.add_at(timeline.instant, audio_samples)
        timeline >> interval

    effects = Pedalboard(
        [
            Reverb(),
            Convolution(impulse_response_filename="ir/ukulele.wav", mix=0.95),
            LowShelfFilter(cutoff_frequency_hz=440, gain_db=10, q=1),
            Gain(gain_db=15),
        ]
    )
    samples = effects(audio_track.samples, audio_track.sampling_rate)

    with AudioFile("chorus.mp3", "w", audio_track.sampling_rate) as file:
        file.write(normalize(samples))


def strumming_pattern() -> Iterator[Tuple[float, Chord, Velocity]]:
    chords = (
        Chord.from_numbers(0, 0, 0, 3),
        Chord.from_numbers(0, 2, 3, 2),
        Chord.from_numbers(2, 0, 0, 0),
        Chord.from_numbers(2, 0, 1, 0),
    )

    fast = Time.from_milliseconds(10)
    slow = Time.from_milliseconds(25)

    strokes = [
        Velocity.down(slow),
        Velocity.down(slow),
        Velocity.up(slow),
        Velocity.up(fast),
        Velocity.down(fast),
        Velocity.up(slow),
    ]

    interval = cycle([0.65, 0.45, 0.75, 0.2, 0.4, 0.25])

    for chord in chords:
        for _ in range(2):  # Repeat each chord twice
            for stroke in strokes:
                yield next(interval), chord, stroke


if __name__ == "__main__":
    main()
