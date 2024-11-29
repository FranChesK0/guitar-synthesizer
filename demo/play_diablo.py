import os
import sys
from typing import Final, Tuple
from fractions import Fraction
from dataclasses import dataclass

import numpy as np
from pedalboard import (  # type: ignore[attr-defined]
    Gain,
    Reverb,
    Pedalboard,
    Convolution,
    LowShelfFilter,
)
from numpy.typing import NDArray
from pedalboard.io import AudioFile  # type: ignore[attr-defined]

root_dir = os.path.abspath(os.path.dirname(__file__)).removesuffix("demo")
python_path = os.path.join(root_dir, "src")
if python_path not in sys.path:
    sys.path.append(python_path)


from guitar_synth.chord import Chord  # noqa: E402
from guitar_synth.track import AudioTrack  # noqa: E402
from guitar_synth.stroke import Velocity  # noqa: E402
from guitar_synth.temporal import Time, MeasuredTimeline  # noqa: E402
from guitar_synth.synthesis import Synthesizer  # noqa: E402
from guitar_synth.instrument import StringTuning, PluckedStringInstrument  # noqa: E402
from guitar_synth.processing import normalize  # noqa: E402

BEATS_PER_MINUTE: Final[int] = 75
BEATS_PER_MEASURE: Final[int] = 4
NOTE_VALUE: Final[Fraction] = Fraction(1, 4)


class MeasureTiming:
    BEAT: Final[Time] = Time(60 / BEATS_PER_MINUTE)
    MEASURE: Final[Time] = BEAT * BEATS_PER_MEASURE


class Note:
    WHOLE: Final[Time] = MeasureTiming.BEAT * NOTE_VALUE.denominator
    SEVEN_SIXTEENTH: Final[Time] = WHOLE * Fraction(7, 16)
    FIVE_SIXTEENTH: Final[Time] = WHOLE * Fraction(5, 16)
    THREE_SIXTEENTH: Final[Time] = WHOLE * Fraction(3, 16)
    ONE_EIGHTH: Final[Time] = WHOLE * Fraction(1, 8)
    ONE_SIXTH: Final[Time] = WHOLE * Fraction(1, 6)
    ONE_THIRTY_SECOND: Final[Time] = WHOLE * Fraction(1, 32)


class StrummingSpeed:
    SLOW: Final[Time] = Time.from_milliseconds(40)
    FAST: Final[Time] = Time.from_milliseconds(20)
    SUPER_FAST: Final[Time] = Time.from_milliseconds(5)


@dataclass(frozen=True)
class Stroke:
    instant: Time
    chord: Chord
    velocity: Velocity


def main() -> None:
    acoustic_guitar = PluckedStringInstrument(
        tuning=StringTuning.from_notes("E2", "A2", "D3", "G3", "B3", "E4"),
        vibration=Time(10),
        damping=0.498,
    )
    synthesizer = Synthesizer(acoustic_guitar)
    audio_track = AudioTrack(synthesizer.sample_rate)
    timeline = MeasuredTimeline(measure=MeasureTiming.MEASURE)
    for measure in measures(timeline):
        for stroke in measure:
            audio_track.add_at(
                stroke.instant, synthesizer.strum_strings(stroke.chord, stroke.velocity)
            )
    save(audio_track, "diablo.mp3")


def measures(timeline: MeasuredTimeline) -> Tuple[Tuple[Stroke, ...], ...]:
    return (measure_01(timeline), measure_02(timeline))


def measure_01(timeline: MeasuredTimeline) -> Tuple[Stroke, ...]:
    return (
        Stroke(
            timeline.instant,
            Chord.from_numbers(0, 0, 2, 2, 0, None),
            Velocity.down(StrummingSpeed.SLOW),
        ),
        Stroke(
            (timeline >> Note.THREE_SIXTEENTH).instant,
            Chord.from_numbers(None, 0, 2, None, None, None),
            Velocity.up(StrummingSpeed.FAST),
        ),
        Stroke(
            (timeline >> Note.ONE_EIGHTH).instant,
            Chord.from_numbers(0, 0, 2, 2, 0, None),
            Velocity.down(StrummingSpeed.SLOW),
        ),
    )


def measure_02(timeline: MeasuredTimeline) -> Tuple[Stroke, ...]:
    return (
        Stroke(
            next(timeline).instant,
            Chord.from_numbers(0, 4, 2, 1, 0, None),
            Velocity.down(StrummingSpeed.SLOW),
        ),
        Stroke(
            (timeline >> Note.THREE_SIXTEENTH).instant,
            Chord.from_numbers(None, None, 2, None, None, None),
            Velocity.down(StrummingSpeed.SUPER_FAST),
        ),
        Stroke(
            (timeline >> Note.ONE_EIGHTH).instant,
            Chord.from_numbers(0, 4, 2, 1, 0, None),
            Velocity.down(StrummingSpeed.SLOW),
        ),
        Stroke(
            (timeline >> Note.SEVEN_SIXTEENTH).instant,
            Chord.from_numbers(7, None, None, None, None, None),
            Velocity.down(StrummingSpeed.SUPER_FAST),
        ),
    )


def save(audio_track: AudioTrack, filename: str) -> None:
    with AudioFile(filename, "w", audio_track.sampling_rate) as file:
        file.write(normalize(apply_effects(audio_track)))
    print(f"\nSaved file {filename!r}")


def apply_effects(audio_track: AudioTrack) -> NDArray[np.float64]:
    effects = Pedalboard(
        [
            Reverb(),
            Convolution(
                impulse_response_filename="ir/acoustic.wav",
                mix=0.95,
            ),
            LowShelfFilter(cutoff_frequency_hz=440, gain_db=10, q=1),
            Gain(gain_db=6),
        ]
    )
    result: NDArray[np.float64] = effects(audio_track.samples, audio_track.sampling_rate)
    return result


if __name__ == "__main__":
    main()
