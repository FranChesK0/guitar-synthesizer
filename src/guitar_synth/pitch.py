import re
from typing import Self
from dataclasses import dataclass

from guitar_synth.temporal import Hertz


@dataclass(frozen=True)
class Pitch:
    frequency: Hertz

    @classmethod
    def from_scientific_notation(cls, notation: str) -> Self:
        if match := re.match(r"([A-G]#?)(-?\d+)?", notation):
            note = match.group(1)
            octave = int(match.group(2) or 0)
            semitones = "C C# D D# E F F# G G# A A# B".split()
            index = octave * 12 + semitones.index(note) - 57
            return cls(frequency=440.0 * 2 ** (index / 12))
        else:
            raise ValueError(f"Invalid pitch notation: '{notation}'")

    def adjust(self, num_semitones: int) -> "Pitch":
        return Pitch(self.frequency * 2 ** (num_semitones / 12))
