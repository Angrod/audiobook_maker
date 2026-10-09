import os
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from voice_tools import (  # noqa: E402
    load_voice_info,
    median_f0,
    median_f0_of_files,
    save_voice_info,
    suggest_semitones,
)


def test_suggest_semitones():
    assert suggest_semitones(110, 220) == 12
    assert suggest_semitones(220, 110) == -12
    assert suggest_semitones(200, 200) == 0
    assert suggest_semitones(100, 150) == 7  # a fifth up
    assert suggest_semitones(60, 1000) == 16  # clamped to the slider
    assert suggest_semitones(None, 200) is None
    assert suggest_semitones(0, 200) is None


def test_voice_info_round_trip(tmp_path):
    voice_dir = str(tmp_path / "Narrator")
    assert load_voice_info(voice_dir) == {}
    save_voice_info(voice_dir, {"target_median_f0": 180.5})
    info = save_voice_info(voice_dir, {"epochs": 300})
    assert info == {"target_median_f0": 180.5, "epochs": 300}
    assert load_voice_info(voice_dir) == info


def _tone(path, hz, seconds=2.0, sr=22050):
    t = np.arange(int(sr * seconds)) / sr
    # A few harmonics so it looks voice-like to pYIN
    wave = sum(0.3 / k * np.sin(2 * np.pi * hz * k * t) for k in range(1, 4))
    sf.write(path, wave.astype(np.float32), sr)


def test_median_f0_on_tones(tmp_path):
    pytest.importorskip("librosa")
    low, high = str(tmp_path / "low.wav"), str(tmp_path / "high.wav")
    _tone(low, 110)
    _tone(high, 220)
    assert median_f0(low) == pytest.approx(110, rel=0.03)
    assert median_f0(high) == pytest.approx(220, rel=0.03)
    assert suggest_semitones(median_f0(low), median_f0(high)) == 12
    assert median_f0_of_files([low, high, low]) == pytest.approx(110, rel=0.03)


def test_median_f0_silence_is_none(tmp_path):
    pytest.importorskip("librosa")
    path = str(tmp_path / "silence.wav")
    sf.write(path, np.zeros(22050, dtype=np.float32), 22050)
    assert median_f0(path) is None
    assert median_f0_of_files([path]) is None
