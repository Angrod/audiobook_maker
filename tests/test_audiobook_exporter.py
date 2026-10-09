import json
import os
import shutil
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from audiobook_exporter import collect_audio_paths, export_audiobook, export_settings_from_global  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _tone(seconds, rate, channels=1, dtype="int16"):
    t = np.arange(int(seconds * rate)) / rate
    wave = 0.3 * np.sin(2 * np.pi * 220 * t)
    if channels == 2:
        wave = np.stack([wave, wave], axis=1)
    return wave.astype(np.float32) if dtype == "float32" else (wave * 32767).astype(np.int16)


@pytest.fixture
def book(tmp_path):
    book_dir = tmp_path / "My Book"
    book_dir.mkdir()
    # Majority format: 32 kHz mono PCM_16 (GPT-SoVITS); one 24 kHz float clip from another engine
    sf.write(book_dir / "audio_0.wav", _tone(1.0, 32000), 32000, subtype="PCM_16")
    sf.write(book_dir / "audio_1.wav", _tone(1.0, 24000, dtype="float32"), 24000, subtype="FLOAT")
    sf.write(book_dir / "audio_3.wav", _tone(1.0, 32000), 32000, subtype="PCM_16")
    text_audio_map = {
        "0": {"sentence": "a", "audio_path": "audiobooks\\My Book\\audio_0.wav", "generated": True},
        "1": {"sentence": "b", "audio_path": str(book_dir / "audio_1.wav"), "generated": True},
        "2": {"sentence": "c", "audio_path": "", "generated": False},
        "3": {"sentence": "d", "audio_path": "audiobooks/My Book/audio_3.wav", "generated": True},
    }
    (book_dir / "text_audio_map.json").write_text(json.dumps(text_audio_map), encoding="utf-8")
    return book_dir


def test_collect_skips_missing_and_handles_windows_paths(book):
    with open(book / "text_audio_map.json", encoding="utf-8") as f:
        paths, missing = collect_audio_paths(str(book), json.load(f))
    assert [os.path.basename(p) for p in paths] == ["audio_0.wav", "audio_1.wav", "audio_3.wav"]
    assert missing == 1


def test_export_wav_converts_mixed_formats_and_adds_pauses(book):
    name, skipped = export_audiobook(str(book), 0.5, {"format": "wav", "normalize_loudness": False})
    assert skipped == 1
    info = sf.info(str(book / "exported_audiobooks" / name))
    assert (info.samplerate, info.channels) == (32000, 1)
    assert info.duration == pytest.approx(3.0 + 2 * 0.5, abs=0.05)


@pytest.mark.parametrize("export_format", ["mp3", "m4b"])
def test_export_compressed_formats_with_loudnorm(book, export_format):
    name, _ = export_audiobook(str(book), 0, {"format": export_format, "normalize_loudness": True})
    assert name == f"My Book_audiobook_0.{export_format}"
    output = book / "exported_audiobooks" / name
    assert output.stat().st_size > 1000
    # A second export never overwrites the first
    name2, _ = export_audiobook(str(book), 0, {"format": export_format, "normalize_loudness": False})
    assert name2 == f"My Book_audiobook_1.{export_format}"


def test_loudnorm_keeps_sample_rate(book):
    name, _ = export_audiobook(str(book), 0, {"format": "wav", "normalize_loudness": True})
    assert sf.info(str(book / "exported_audiobooks" / name)).samplerate == 32000


def test_no_audio_raises(tmp_path):
    (tmp_path / "text_audio_map.json").write_text(json.dumps({"0": {"sentence": "a", "audio_path": "", "generated": False}}))
    with pytest.raises(ValueError):
        export_audiobook(str(tmp_path), 0, {"format": "mp3", "normalize_loudness": False})


def test_settings_validation():
    assert export_settings_from_global({"export_format": "FLAC"})["format"] == "mp3"
    assert export_settings_from_global({"export_format": "m4b", "export_normalize_loudness": True}) == {
        "format": "m4b", "normalize_loudness": True}
