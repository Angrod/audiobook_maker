import os
import shutil
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tts_engines  # noqa: E402

ALL_VERSIONS = {"v1": {}, "v2": {}, "v3": {}, "v4": {}, "v2Pro": {}, "v2ProPlus": {}}


class FakeConfig:
    default_configs = {}

    def __init__(self, path, local_files_only=False):
        self.path = path


class FakePipeline:
    instances = []

    def __init__(self, cfg):
        self.calls = []
        FakePipeline.instances.append(self)

    def init_t2s_weights(self, path):
        self.calls.append(("t2s", path))

    def init_vits_weights(self, path, **kwargs):
        self.calls.append(("vits", path, kwargs))


@pytest.fixture
def gpt_sovits(tmp_path, monkeypatch):
    for folder in ("configs", os.path.join("engines", "gpt_sovits")):
        shutil.copytree(os.path.join(ROOT, folder), tmp_path / folder, ignore=shutil.ignore_patterns(".gitignore"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tts_engines, "TTS", FakePipeline, raising=False)
    monkeypatch.setattr(tts_engines, "TTS_Config", FakeConfig, raising=False)
    monkeypatch.setattr(FakeConfig, "default_configs", dict(ALL_VERSIONS))
    FakePipeline.instances.clear()

    detected = {"version": "v2"}
    process_ckpt = types.ModuleType("GPT_SoVITS.process_ckpt")
    process_ckpt.get_sovits_version_from_path_fast = lambda path: ("v2", detected["version"], False)
    monkeypatch.setitem(sys.modules, "GPT_SoVITS", types.ModuleType("GPT_SoVITS"))
    monkeypatch.setitem(sys.modules, "GPT_SoVITS.process_ckpt", process_ckpt)
    return detected


def test_v2proplus_loads_like_v2_without_vocoder(gpt_sovits):
    gpt_sovits["version"] = "v2ProPlus"
    tts_engines.load_with_gpt_sovits(gpt_sovits_version="v2ProPlus")
    calls = FakePipeline.instances[-1].calls
    assert calls[0] == ("t2s", "engines/gpt_sovits/pretrained_models/s1v3.ckpt")
    assert calls[1] == ("vits", "engines/gpt_sovits/pretrained_models/v2Pro/s2Gv2ProPlus.pth", {})
    assert FakeConfig.default_configs["v2ProPlus"]["vits_weights_path"].endswith("v2Pro/s2Gv2ProPlus.pth")


def test_custom_v2pro_model_detected_from_file_even_if_dropdown_says_v4(gpt_sovits):
    gpt_sovits["version"] = "v2Pro"
    tts_engines.load_with_gpt_sovits(gpt_sovits_version="v4", gpt_sovits_model="my.ckpt", gpt_sovits_vits_model="my.pth")
    calls = FakePipeline.instances[-1].calls
    assert calls[1] == ("vits", os.path.join("engines/gpt_sovits/sovits_models", "my.pth"), {})


def test_v4_still_uses_vocoder(gpt_sovits):
    gpt_sovits["version"] = "v4"
    tts_engines.load_with_gpt_sovits(gpt_sovits_version="v4")
    _, path, kwargs = FakePipeline.instances[-1].calls[1]
    assert kwargs == {"vocoder_path": "engines/gpt_sovits/pretrained_models/gsv-v4-pretrained/vocoder.pth",
                      "model_version": "v4"}


def test_old_package_without_v2pro_gives_update_instructions(gpt_sovits, monkeypatch):
    monkeypatch.setattr(FakeConfig, "default_configs", {"v1": {}, "v2": {}, "v3": {}, "v4": {}})
    with pytest.raises(ValueError, match="git submodule sync"):
        tts_engines.load_with_gpt_sovits(gpt_sovits_version="v2ProPlus")
    # Older versions keep working with the old package
    gpt_sovits["version"] = "v2"
    tts_engines.load_with_gpt_sovits(gpt_sovits_version="v2")
    assert "v2ProPlus" not in FakeConfig.default_configs
