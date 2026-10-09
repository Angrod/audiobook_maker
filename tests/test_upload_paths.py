import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model import AudiobookModel  # noqa: E402


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return AudiobookModel({})


def _file(tmp_path, name):
    path = tmp_path / "source" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"data")
    return str(path)


def test_rvc_model_and_index_go_in_the_voice_folder(model, tmp_path):
    model.process_upload_items("AI Models", [
        {"name": "Narrator"},
        {"type": "file", "target_path": "engines/rvc", "source_path": _file(tmp_path, "g_200e.pth"), "save_format": "folder"},
        {"type": "file", "target_path": "engines/rvc", "source_path": _file(tmp_path, "added.index"), "save_format": "folder"},
    ])
    # rvc-python and the voice list look for engines/rvc/<voice>/<files>
    assert sorted(os.listdir("engines/rvc")) == ["Narrator"]
    assert sorted(os.listdir("engines/rvc/Narrator")) == ["Narrator.index", "Narrator.pth"]


def test_styletts_model_and_config_go_in_the_model_folder(model, tmp_path):
    model.process_upload_items("AI Models", [
        {"name": "MyModel"},
        {"type": "file", "target_path": "engines/styletts", "source_path": _file(tmp_path, "epoch_2nd.pth"), "save_format": "folder"},
        {"type": "file", "target_path": "engines/styletts", "source_path": _file(tmp_path, "config.yml"), "save_format": "folder"},
    ])
    assert sorted(os.listdir("engines/styletts/MyModel")) == ["MyModel.pth", "MyModel.yml"]


def test_file_format_uploads_are_unchanged(model, tmp_path):
    os.makedirs("engines/gpt_sovits/gpt_models")
    model.process_upload_items("AI Models", [
        {"name": "Voice"},
        {"type": "file", "target_path": "engines/gpt_sovits/gpt_models", "source_path": _file(tmp_path, "x.ckpt"), "save_format": "file"},
    ])
    assert os.listdir("engines/gpt_sovits/gpt_models") == ["Voice.ckpt"]


def test_reference_transcript_creates_its_folder(model):
    model.process_upload_items("Voice Reference File", [
        {"name": "Ref"},
        {"type": "text", "target_path": "voices/gpt_sovits", "source_text": "Hello there.", "save_format": "folder"},
    ])
    with open("voices/gpt_sovits/Ref/Ref.txt") as f:
        assert f.read() == "Hello there."
