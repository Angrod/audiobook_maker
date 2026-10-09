import os
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tts_engines  # noqa: E402


class FakeGptSovits:
    def __init__(self):
        self.inputs = None

    def run(self, inputs):
        self.inputs = inputs
        yield 32000, np.zeros(320, dtype=np.float32)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    shutil.copytree(os.path.join(ROOT, "configs"), tmp_path / "configs")
    voice = tmp_path / "voices" / "gpt_sovits" / "narrator"
    voice.mkdir(parents=True)
    (voice / "narrator.wav").write_bytes(b"")
    (voice / "narrator.txt").write_text("Reference transcript.", encoding="utf-8")
    (voice / "extra_1.wav").write_bytes(b"")
    (voice / "notes.txt").write_text("not audio", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


BASE_PARAMETERS = {
    "gpt_sovits_voice": "narrator",
    "gpt_sovits_output_lang": "en",
    "gpt_sovits_ref_lang": "en",
    "gpt_sovits_seed": -1,
    "gpt_sovits_sample_steps": 8,
    "gpt_sovits_temperature": 70,
    "gpt_sovits_top_k": 5,
    "gpt_sovits_top_p": 100,
}


def test_new_parameters_are_passed(workdir):
    engine = FakeGptSovits()
    params = {
        **BASE_PARAMETERS,
        "gpt_sovits_speed": 85,
        "gpt_sovits_repetition_penalty": 160,
        "gpt_sovits_text_split_method": "cut5",
        "gpt_sovits_use_aux_refs": True,
    }
    out = str(workdir / "out.wav")
    assert tts_engines.generate_with_gpt_sovits(engine, "Hello there.", params, out) == out
    assert os.path.exists(out)
    assert engine.inputs["speed_factor"] == 0.85
    assert engine.inputs["repetition_penalty"] == 1.6
    assert engine.inputs["text_split_method"] == "cut5"
    assert [os.path.basename(p) for p in engine.inputs["aux_ref_audio_paths"]] == ["extra_1.wav"]
    assert engine.inputs["temperature"] == 0.7


def test_old_speaker_settings_use_previous_defaults(workdir):
    engine = FakeGptSovits()
    tts_engines.generate_with_gpt_sovits(engine, "Hello there.", dict(BASE_PARAMETERS), str(workdir / "out.wav"))
    assert engine.inputs["speed_factor"] == 1.0
    assert engine.inputs["repetition_penalty"] == 1.35
    assert engine.inputs["text_split_method"] == "cut0"
    assert engine.inputs["aux_ref_audio_paths"] == []
