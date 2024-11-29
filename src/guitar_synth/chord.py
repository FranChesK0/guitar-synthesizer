from typing import Self, Tuple, Optional


class Chord(Tuple[int | None, ...]):
    @classmethod
    def from_numbers(cls, *numbers: Optional[int]) -> Self:
        return cls(numbers)
