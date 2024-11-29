from typing import Self, Tuple, Optional
from functools import cache, cached_property
from dataclasses import dataclass

from guitar_synth.chord import Chord
from guitar_synth.pitch import Pitch
from guitar_synth.temporal import Time


@dataclass(frozen=True)
class VibratingString:
    pitch: Pitch

    def press_fret(self, fret_number: Optional[int] = None) -> Pitch:
        if fret_number is None:
            return self.pitch
        return self.pitch.adjust(fret_number)


@dataclass(frozen=True)
class StringTuning:
    strings: Tuple[VibratingString, ...]

    @classmethod
    def from_notes(cls, *notes: str) -> Self:
        return cls(
            strings=tuple(
                VibratingString(Pitch.from_scientific_notation(note))
                for note in reversed(notes)
            )
        )


@dataclass(frozen=True)
class PluckedStringInstrument:
    tuning: StringTuning
    vibration: Time
    damping: float = 0.5

    def __post_init__(self) -> None:
        if not (0 < self.damping <= 0.5):
            raise ValueError("Damping must be between 0 and 0.5")

    @cached_property
    def num_strings(self) -> int:
        return len(self.tuning.strings)

    @cache
    def downstroke(self, chord: Chord) -> Tuple[Pitch, ...]:
        return tuple(reversed(self.upstroke(chord)))

    @cache
    def upstroke(self, chord: Chord) -> Tuple[Pitch, ...]:
        if len(chord) != self.num_strings:
            raise ValueError(
                "Chord must have the same number of strings as the instrument"
            )
        return tuple(
            string.press_fret(fret_number)
            for string, fret_number in zip(self.tuning.strings, chord)
            if fret_number is not None
        )
