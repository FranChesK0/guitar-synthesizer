import pytest

from guitar_synth.chord import Chord
from guitar_synth.pitch import Pitch
from guitar_synth.temporal import Time
from guitar_synth.instrument import StringTuning, VibratingString, PluckedStringInstrument


# Test VibratingString class
@pytest.mark.parametrize("is_fret_pressed", [True, False])
def test_vibrating_string_press_fret(is_fret_pressed: bool) -> None:
    pitch = Pitch.from_scientific_notation("A4")
    string = VibratingString(pitch)

    if is_fret_pressed:
        adjusted_pitch = string.press_fret(2)
        assert adjusted_pitch.frequency == Pitch.from_scientific_notation("B4").frequency
    else:
        assert string.press_fret() == pitch


# Test PluckedStringInstrument class
def test_plucked_string_instrument_dumping() -> None:
    with pytest.raises(ValueError, match="Damping must be between 0 and 0.5"):
        PluckedStringInstrument(
            tuning=StringTuning.from_notes("E4", "A3", "D4", "G4", "B3", "E2"),
            vibration=Time(1),
            damping=0.6,  # Invalid damping value
        )


def test_plucked_string_instrument_num_strings() -> None:
    tuning = StringTuning.from_notes("E4", "A3", "D4", "G4", "B3", "E2")
    instrument = PluckedStringInstrument(
        tuning=tuning,
        vibration=Time(1),
    )
    assert instrument.num_strings == 6


def test_plucked_string_instrument_upstroke_chord() -> None:
    tuning = StringTuning.from_notes("E4", "A3", "D4", "G4", "B3", "E2")
    instrument = PluckedStringInstrument(
        tuning=tuning,
        vibration=Time(1),
    )
    chord = Chord([0, 2, 2, 1, 0, 0])  # A major chord
    upstroke_pitches = instrument.upstroke(chord)

    # Check the resulting upstroke pitches
    assert len(upstroke_pitches) == 6
    assert upstroke_pitches[0].frequency == tuning.strings[0].pitch.adjust(0).frequency
    assert upstroke_pitches[1].frequency == tuning.strings[1].pitch.adjust(2).frequency
    assert upstroke_pitches[2].frequency == tuning.strings[2].pitch.adjust(2).frequency
    assert upstroke_pitches[3].frequency == tuning.strings[3].pitch.adjust(1).frequency
    assert upstroke_pitches[4].frequency == tuning.strings[4].pitch.adjust(0).frequency
    assert upstroke_pitches[5].frequency == tuning.strings[5].pitch.adjust(0).frequency


def test_plucked_string_instrument_downstroke_chord() -> None:
    tuning = StringTuning.from_notes("E4", "A3", "D4", "G4", "B3", "E2")
    instrument = PluckedStringInstrument(
        tuning=tuning,
        vibration=Time(1),
    )
    chord = Chord([0, 2, 2, 1, 0, 0])  # A major chord
    downstroke_pitches = instrument.downstroke(chord)

    # Check the resulting downstroke pitches
    assert len(downstroke_pitches) == 6
    assert downstroke_pitches[0].frequency == tuning.strings[5].pitch.adjust(0).frequency
    assert downstroke_pitches[1].frequency == tuning.strings[4].pitch.adjust(0).frequency
    assert downstroke_pitches[2].frequency == tuning.strings[3].pitch.adjust(1).frequency
    assert downstroke_pitches[3].frequency == tuning.strings[2].pitch.adjust(2).frequency
    assert downstroke_pitches[4].frequency == tuning.strings[1].pitch.adjust(2).frequency
    assert downstroke_pitches[5].frequency == tuning.strings[0].pitch.adjust(0).frequency
