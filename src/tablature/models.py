from typing import Any, Dict, List, Self, Final, Tuple, Optional, Annotated
from pathlib import Path
from fractions import Fraction
from functools import cached_property

import yaml
from pydantic import (
    Field,
    HttpUrl,
    BaseModel,
    PositiveInt,
    PositiveFloat,
    NonNegativeInt,
    NonNegativeFloat,
    model_validator,
)

DEFAULT_STRING_DAMPING: Final[float] = 0.5
DEFAULT_ARPEGGIO_SECONDS: Final[float] = 0.005


class Note(BaseModel):
    frets: Annotated[List[Optional[NonNegativeInt]], Field(min_length=1)]
    offset: Annotated[str, Field(pattern=r"\d+/\d+")] = "0/1"
    upstroke: bool = False
    arpeggio: NonNegativeFloat = DEFAULT_ARPEGGIO_SECONDS
    vibration: Optional[PositiveFloat] = None


class Measure(BaseModel):
    time_signature: Annotated[str, Field(pattern=r"\d+/\d+")]
    notes: Tuple[Note, ...] = tuple()

    @cached_property
    def beats_per_measure(self) -> int:
        return int(self.time_signature.split("/")[0])

    @cached_property
    def note_value(self) -> Fraction:
        return Fraction(1, int(self.time_signature.split("/")[1]))


class Tablature(BaseModel):
    beats_per_minute: PositiveInt
    measures: Tuple[Measure, ...]


class Instrument(BaseModel):
    tuning: Annotated[
        List[Annotated[str, Field(pattern=r"([A-G]#?)(-?\d+)?")]], Field(min_length=1)
    ]
    vibration: PositiveFloat
    damping: Annotated[float, Field(ge=0, le=0.5)] = DEFAULT_STRING_DAMPING
    effects: Tuple[str | Dict[Any, Any], ...] = tuple()


class Track(BaseModel):
    url: Optional[HttpUrl] = None
    weight: NonNegativeFloat = 1.0
    instrument: Instrument
    tablature: Tablature

    @model_validator(mode="after")
    def check_frets(self) -> Self:
        num_strings = len(self.instrument.tuning)
        for measure in self.tablature.measures:
            for notes in measure.notes:
                if len(notes.frets) != num_strings:
                    raise ValueError("All notes must have the same number of frets")
        return self


class Song(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    tracks: Dict[str, Track]

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        with Path(path).open(encoding="utf-8") as file:
            return cls(**yaml.safe_load(file))
