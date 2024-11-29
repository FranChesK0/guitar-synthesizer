import os
from typing import Any, Dict, List, Final, Generator
from pathlib import Path
from argparse import Namespace, ArgumentParser
from fractions import Fraction
from contextlib import contextmanager

import numpy as np
import pedalboard
from numpy.typing import NDArray
from pedalboard.io import AudioFile  # type: ignore[attr-defined]

from tablature import models
from guitar_synth.chord import Chord
from guitar_synth.track import AudioTrack
from guitar_synth.stroke import Velocity
from guitar_synth.temporal import Time, MeasuredTimeline
from guitar_synth.synthesis import Synthesizer
from guitar_synth.instrument import StringTuning, PluckedStringInstrument
from guitar_synth.processing import normalize

SAMPLING_RATE: Final[int] = 44100


def main() -> None:
    play(parse_args())


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to tablature file in YAML format")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Path to output audio file"
    )
    return parser.parse_args()


def play(args: Namespace) -> None:
    song = models.Song.from_file(args.path)
    with chdir(args.path.parent):
        samples = normalize(
            np.sum(
                pad_to_longest(
                    [track.weight * synthesize(track) for track in song.tracks.values()]
                ),
                axis=0,
            )
        )
    save(samples, args.output or Path.cwd() / args.path.with_suffix(".mp3").name)


@contextmanager
def chdir(directory: Path) -> Generator[Any, None, None]:
    current_dir = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(current_dir)


def pad_to_longest(tracks: List[NDArray[np.float64]]) -> List[NDArray[np.float64]]:
    max_length = max(array.size for array in tracks)
    return [np.pad(array, (0, max_length - array.size)) for array in tracks]


def save(samples: NDArray[np.float64], path: Path) -> None:
    with AudioFile(str(path), "w", SAMPLING_RATE) as file:
        file.write(samples)
    print(f"Saved file {path.absolute()}")


def synthesize(track: models.Track) -> NDArray[np.float64]:
    synthesizer = Synthesizer(
        instrument=PluckedStringInstrument(
            tuning=StringTuning.from_notes(*track.instrument.tuning),
            damping=track.instrument.damping,
            vibration=Time(track.instrument.vibration),
        ),
        sample_rate=SAMPLING_RATE,
    )
    audio_track = AudioTrack(synthesizer.sample_rate)
    timeline = MeasuredTimeline()
    read(track.tablature, synthesizer, audio_track, timeline)
    return apply_effects(audio_track, track.instrument)


def read(
    tablature: models.Tablature,
    synthesizer: Synthesizer,
    audio_track: AudioTrack,
    timeline: MeasuredTimeline,
) -> None:
    beat = Time(60 / tablature.beats_per_minute)
    for measure in tablature.measures:
        timeline.measure = beat * measure.beats_per_measure
        whole_note = beat * measure.note_value.denominator
        for note in measure.notes:
            stroke = Velocity.up if note.upstroke else Velocity.down
            audio_track.add_at(
                (timeline >> (whole_note * Fraction(note.offset))).instant,
                synthesizer.strum_strings(
                    chord=Chord(note.frets),
                    velocity=stroke(delay=Time(note.arpeggio)),
                    vibration=Time(note.vibration) if note.vibration else None,
                ),
            )
        next(timeline)


def apply_effects(
    audio_track: AudioTrack, instrument: models.Instrument
) -> NDArray[np.float64]:
    effects = pedalboard.Pedalboard(get_plugins(instrument))
    res: NDArray[np.float64] = effects(audio_track.samples, audio_track.sampling_rate)
    return res


def get_plugins(instrument: models.Instrument) -> List[Any]:
    return [get_plugin(effect) for effect in instrument.effects]


def get_plugin(effect: str | Dict[Any, Any]) -> Any:
    match effect:
        case str() as class_name:
            return getattr(pedalboard, class_name)()
        case dict() as plugin_dict if len(plugin_dict) == 1:
            class_name, params = list(plugin_dict.items())[0]
            return getattr(pedalboard, class_name)(**params)
