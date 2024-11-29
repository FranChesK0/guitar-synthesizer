import pytest

from guitar_synth.temporal import Time
from guitar_synth.synthesis import Synthesis
from guitar_synth.instrument import StringTuning, PluckedStringInstrument


@pytest.fixture(scope="function")
def instrument() -> PluckedStringInstrument:
    tuning = StringTuning.from_notes("E4", "A3", "D4", "G4", "B3", "E2")
    return PluckedStringInstrument(tuning=tuning, vibration=Time(1))


@pytest.fixture(scope="function")
def synthesis(instrument: PluckedStringInstrument) -> Synthesis:
    return Synthesis(instrument=instrument)
