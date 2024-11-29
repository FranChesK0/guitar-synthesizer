import numpy as np
import pytest  # noqa: F401
from numpy.testing import assert_array_equal

from guitar_synth.track import AudioTrack
from guitar_synth.temporal import Time


def test_audio_track_len(audio_track: AudioTrack) -> None:
    assert len(audio_track) == 0

    samples = np.ones(100)
    audio_track.add(samples)

    assert len(audio_track) == 100

    samples = np.ones(200)
    audio_track.add(samples)

    assert len(audio_track) == 300


def test_audio_track_add_samples(audio_track: AudioTrack) -> None:
    samples = np.ones(100)
    audio_track.add(samples)

    assert len(audio_track) == 100
    assert_array_equal(audio_track.samples[:100], samples)


def test_audio_track_duration(audio_track: AudioTrack) -> None:
    samples = np.ones(44100)
    audio_track.add(samples)
    assert audio_track.duration == Time(1)


def test_audio_track_add_at(audio_track: AudioTrack) -> None:
    samples1 = np.ones(100)
    audio_track.add_at(Time(0), samples1)

    assert len(audio_track) == 100
    assert_array_equal(audio_track.samples[:100], samples1)

    samples2 = np.ones(100) * 2
    audio_track.add_at(Time(0.5), samples2)

    assert len(audio_track) == 22150
    assert_array_equal(audio_track.samples[22050:22150], samples2)


def test_audio_track_add_at_with_gap(audio_track: AudioTrack) -> None:
    samples1 = np.ones(100)
    audio_track.add_at(Time(0), samples1)

    samples2 = np.ones(100) * 2
    audio_track.add_at(Time(2), samples2)

    assert len(audio_track) == 88300
    assert_array_equal(audio_track.samples[88200:88300], samples2)


def test_audio_track_add_at_with_padding(audio_track: AudioTrack) -> None:
    samples1 = np.ones(100)
    audio_track.add_at(Time(0), samples1)

    samples2 = np.ones(100) * 3
    audio_track.add_at(Time(5), samples2)

    assert len(audio_track) == 220600
    assert_array_equal(audio_track.samples[220500:220600], samples2)


def test_audio_track_add_at_truncate(audio_track: AudioTrack) -> None:
    samples1 = np.ones(100)
    audio_track.add_at(Time(0), samples1)

    samples2 = np.ones(100) * 4
    audio_track.add_at(Time(0.1), samples2)

    samples3 = np.ones(100) * 5
    audio_track.add_at(Time(0.2), samples3)

    assert_array_equal(audio_track.samples[:100], samples1)
    assert_array_equal(audio_track.samples[4410:4510], samples2)
    assert_array_equal(audio_track.samples[8820:8920], samples3)


def test_audio_track_add_at_truncate_to_end(audio_track: AudioTrack) -> None:
    samples1 = np.ones(100)
    audio_track.add_at(Time(0), samples1)

    samples2 = np.ones(100) * 6
    audio_track.add_at(Time(1), samples2)

    samples3 = np.ones(100) * 7
    audio_track.add_at(Time(2), samples3)

    samples4 = np.ones(100) * 8
    audio_track.add_at(Time(3), samples4)

    assert_array_equal(audio_track.samples[:100], samples1)
    assert_array_equal(audio_track.samples[44100:44200], samples2)
    assert_array_equal(audio_track.samples[88200:88300], samples3)
    assert_array_equal(audio_track.samples[132300:132400], samples4)
