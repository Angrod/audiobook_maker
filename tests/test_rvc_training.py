import json
import os
import shutil
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import rvc_training as rt  # noqa: E402


# ------------------------------------------------------------------ pure decisions

@pytest.mark.parametrize("vram,batch", [(4, 4), (6, 4), (8, 6), (12, 8), (24, 12)])
def test_batch_size_by_vram(vram, batch):
    assert rt.recommend_settings(15, vram)["batch_size"] == batch


@pytest.mark.parametrize("minutes,epochs", [(3, 300), (10, 300), (15, 250), (30, 200), (50, 150), (90, 100)])
def test_epochs_by_dataset_length(minutes, epochs):
    settings = rt.recommend_settings(minutes, 8)
    assert settings["total_epoch"] == epochs
    # About six checkpoints to compare, within Applio's 1-100 range
    assert 5 <= epochs // settings["save_every_epoch"] <= 7
    assert 10 <= settings["save_every_epoch"] <= 100


def test_fixed_settings_and_warnings():
    settings = rt.recommend_settings(3, 8, noisy=True)
    assert (settings["sample_rate"], settings["f0_method"], settings["vocoder"], settings["embedder_model"]) == \
        (40000, "rmvpe", "HiFi-GAN", "contentvec")
    assert settings["noise_reduction"] and settings["can_train"]
    assert any("Under 5 minutes" in w for w in settings["warnings"])
    assert any("Over an hour" in w for w in rt.recommend_settings(90, 8)["warnings"])
    no_gpu = rt.recommend_settings(20, None)
    assert not no_gpu["can_train"] and any("No NVIDIA GPU" in w for w in no_gpu["warnings"])
    assert not rt.recommend_settings(0.5, 8)["can_train"]


def test_voice_name_rules():
    assert rt.validate_voice_name("Narrator_v2-1") is None
    for bad in ("", "my voice", "-x", "a/b", "x" * 70):
        assert rt.validate_voice_name(bad)


def test_parse_epoch():
    assert rt.parse_epoch("Voice | epoch=12 | step=340 | time=00:01") == 12
    assert rt.parse_epoch("Loaded pretrained (G)") is None


# ------------------------------------------------------------------ commands

def test_build_steps_exact_applio_commands(tmp_path):
    applio = str(tmp_path)
    settings = rt.recommend_settings(15, 8)
    steps = rt.build_steps(applio, "Narrator", "/data/narrator", settings)
    labels = [label for label, _ in steps]
    assert labels == ["Downloading Applio's base models", "Preparing the dataset",
                      "Extracting pitch and voice features", "Training"]
    argv = {label: args[1:] for label, args in steps}
    assert argv["Downloading Applio's base models"] == ["core.py", "prerequisites"]
    assert argv["Preparing the dataset"] == ["core.py", "preprocess", "--model-name", "Narrator", "--dataset-path",
                                             "/data/narrator", "--sample-rate", "40000", "--cut-preprocess", "Automatic"]
    assert argv["Extracting pitch and voice features"] == [
        "core.py", "extract", "--model-name", "Narrator", "--f0-method", "rmvpe", "--gpu", "0",
        "--sample-rate", "40000", "--embedder-model", "contentvec"]
    assert argv["Training"] == [
        "core.py", "train", "--model-name", "Narrator", "--vocoder", "HiFi-GAN", "--save-every-epoch", "40",
        "--total-epoch", "250", "--sample-rate", "40000", "--batch-size", "6", "--gpu", "0", "--index-algorithm", "Auto"]
    assert all(args[0] == rt.applio_python(applio) for _, args in steps)


def test_build_steps_skips_downloads_and_resume(tmp_path):
    applio = str(tmp_path)
    for path in rt.missing_prerequisites(applio):
        os.makedirs(os.path.dirname(os.path.join(applio, path)), exist_ok=True)
        open(os.path.join(applio, path), "w").close()
    settings = rt.recommend_settings(15, 8, noisy=True)
    steps = rt.build_steps(applio, "N", "/d", settings)
    assert [label for label, _ in steps][0] == "Preparing the dataset"
    assert steps[0][1][-1] == "--noise-reduction"
    assert [label for label, _ in rt.build_steps(applio, "N", "/d", settings, resume=True)] == ["Training"]


def test_build_steps_refuses_vocoders_the_app_cannot_play(tmp_path):
    settings = dict(rt.recommend_settings(15, 8), vocoder="RefineGAN")
    with pytest.raises(ValueError):
        rt.build_steps(str(tmp_path), "N", "/d", settings)


# ------------------------------------------------------------------ results

def _fake_logs(applio, name, epochs):
    folder = rt.experiment_dir(applio, name)
    os.makedirs(folder)
    for e in epochs:
        with open(os.path.join(folder, f"{name}_{e}e_{e * 10}s.pth"), "w") as f:
            f.write(f"weights {e}")
    with open(os.path.join(folder, f"{name}.index"), "w") as f:
        f.write("index")
    open(os.path.join(folder, "G_100.pth"), "w").close()  # training state, not a voice model


def test_find_outputs_sorts_by_epoch(tmp_path):
    _fake_logs(str(tmp_path), "V", [100, 20, 5])
    outputs = rt.find_outputs(str(tmp_path), "V")
    assert [os.path.basename(p) for p in outputs["checkpoints"]] == ["V_5e_50s.pth", "V_20e_200s.pth", "V_100e_1000s.pth"]
    assert outputs["index"].endswith("V.index")
    assert rt.can_resume(str(tmp_path), "V")


def test_import_voice_layout_and_checkpoint_switch(tmp_path):
    applio, voices = str(tmp_path / "applio"), str(tmp_path / "voices")
    _fake_logs(applio, "V", [50, 100])
    voice_dir = rt.import_voice(applio, "V", voices, {"target_median_f0": 180.0})
    # rvc-python takes the first .pth in the folder, so there must be exactly one
    assert sorted(os.listdir(voice_dir)) == ["V.index", "V.pth", "checkpoints", "voice_info.json"]
    with open(os.path.join(voice_dir, "V.pth")) as f:
        assert f.read() == "weights 100"
    assert rt.list_checkpoints(voice_dir) == ["V_50e_500s.pth", "V_100e_1000s.pth"]
    rt.use_checkpoint(voice_dir, "V_50e_500s.pth")
    with open(os.path.join(voice_dir, "V.pth")) as f:
        assert f.read() == "weights 50"
    with open(os.path.join(voice_dir, "voice_info.json")) as f:
        info = json.load(f)
    assert info["target_median_f0"] == 180.0 and info["active_epoch"] == 50 and info["trained_with"].startswith("Applio")


def test_import_voice_reports_missing_outputs(tmp_path):
    os.makedirs(rt.experiment_dir(str(tmp_path), "V"))
    with pytest.raises(RuntimeError, match="without saving"):
        rt.import_voice(str(tmp_path), "V", str(tmp_path / "voices"))


# ------------------------------------------------------------------ dataset

def _wav(path, seconds, hz=150):
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(path, (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32), 16000)


def test_dataset_summary_and_single_file(tmp_path):
    data = tmp_path / "clips"
    (data / "sub").mkdir(parents=True)
    _wav(str(data / "a.wav"), 30)
    _wav(str(data / "sub" / "b.wav"), 90)
    (data / "c.m4a").write_bytes(b"")
    summary = rt.dataset_summary(str(data))
    assert summary["files"] == 2 and summary["minutes"] == pytest.approx(2.0, abs=0.01)
    assert summary["ignored_formats"] == ["c.m4a"]

    applio = str(tmp_path / "applio")
    single = rt.prepare_dataset(applio, "V", str(data / "a.wav"))
    assert os.listdir(single) == ["a.wav"] and single.startswith(os.path.abspath(applio))
    assert rt.prepare_dataset(applio, "V", str(data)) == os.path.abspath(str(data))


# ------------------------------------------------------------------ full pipeline with a fake Applio

@pytest.fixture
def fake_applio(tmp_path, monkeypatch):
    applio = tmp_path / "Applio"
    applio.mkdir()
    shutil.copy(os.path.join(ROOT, "tests", "fake_applio_core.py"), applio / "core.py")
    monkeypatch.setattr(rt, "applio_python", lambda path: sys.executable)
    data = tmp_path / "clips"
    data.mkdir()
    for i in range(3):
        _wav(str(data / f"{i}.wav"), 20, hz=180)
    return {"applio": str(applio), "data": str(data), "voices": str(tmp_path / "voices")}


def _calls(applio):
    with open(os.path.join(applio, "calls.log")) as f:
        return [line.split()[0] for line in f]


def _train(env, settings, should_stop=lambda: False, resume=False, progress=None):
    lines = []
    progress = progress if progress is not None else []
    voice_dir = rt.train_voice(env["applio"], "Narrator", env["data"], settings, lines.append,
                               lambda pct, text: progress.append((pct, text)), should_stop,
                               resume=resume, voices_dir=env["voices"])
    return voice_dir, lines, progress


def test_full_training_installs_the_voice(fake_applio):
    settings = dict(rt.recommend_settings(1, 8), total_epoch=30, save_every_epoch=10)
    voice_dir, lines, progress = _train(fake_applio, settings)
    assert _calls(fake_applio["applio"]) == ["prerequisites", "preprocess", "extract", "train"]
    assert sorted(os.listdir(voice_dir)) == ["Narrator.index", "Narrator.pth", "checkpoints", "voice_info.json"]
    assert rt.list_checkpoints(voice_dir) == ["Narrator_10e_100s.pth", "Narrator_20e_200s.pth", "Narrator_30e_300s.pth"]
    percents = [p for p, _ in progress if p is not None]
    assert percents == sorted(percents) and percents[-1] == 100 and (33, "Training: epoch 10 of 30") in progress
    with open(os.path.join(voice_dir, "voice_info.json")) as f:
        info = json.load(f)
    assert info["dataset_minutes"] == 1.0 and info["total_epoch"] == 30
    try:
        import librosa  # noqa: F401
    except ImportError:
        return
    assert info["target_median_f0"] == pytest.approx(180, rel=0.05)


def test_applio_failure_is_detected_from_missing_files(fake_applio, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "train")  # Applio prints the failure but exits 0
    with pytest.raises(RuntimeError, match="without saving"):
        _train(fake_applio, dict(rt.recommend_settings(1, 8), total_epoch=5, save_every_epoch=5))


def test_failing_step_stops_the_pipeline(fake_applio, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "preprocess")
    with pytest.raises(RuntimeError, match="Preparing the dataset failed"):
        _train(fake_applio, dict(rt.recommend_settings(1, 8), total_epoch=5, save_every_epoch=5))
    assert "train" not in _calls(fake_applio["applio"])


def test_cancel_then_resume(fake_applio, monkeypatch):
    monkeypatch.setenv("FAKE_EPOCH_SECONDS", "0.05")
    settings = dict(rt.recommend_settings(1, 8), total_epoch=40, save_every_epoch=10)
    progress = []
    stop = lambda: any(text == "Training: epoch 12 of 40" for _, text in progress)  # noqa: E731
    started = time.time()
    with pytest.raises(rt.TrainingCancelled):
        _train(fake_applio, settings, should_stop=stop, progress=progress)
    assert time.time() - started < 10
    assert rt.can_resume(fake_applio["applio"], "Narrator")

    voice_dir, lines, _ = _train(fake_applio, settings, resume=True)
    assert _calls(fake_applio["applio"])[-1] == "train" and _calls(fake_applio["applio"]).count("preprocess") == 1
    assert any("epoch=11 " in line for line in lines)  # picked up after the epoch-10 checkpoint
    assert rt.list_checkpoints(voice_dir)[-1] == "Narrator_40e_400s.pth"


def test_cancel_from_another_thread_kills_a_quiet_process(fake_applio, monkeypatch):
    # The worker's Cancel button kills the running process directly, so a long silent epoch
    # does not delay cancelling
    monkeypatch.setenv("FAKE_EPOCH_SECONDS", "30")
    settings = dict(rt.recommend_settings(1, 8), total_epoch=5, save_every_epoch=5)
    running, stopped = [], threading.Event()

    def cancel_soon():
        while not running or running[-1].args[2] != "train":
            time.sleep(0.05)
        time.sleep(0.5)
        stopped.set()
        rt.kill_process_tree(running[-1])

    threading.Thread(target=cancel_soon, daemon=True).start()
    started = time.time()
    with pytest.raises(rt.TrainingCancelled):
        rt.train_voice(fake_applio["applio"], "Narrator", fake_applio["data"], settings, lambda line: None,
                       lambda pct, text: None, stopped.is_set, voices_dir=fake_applio["voices"],
                       on_process=running.append)
    assert time.time() - started < 15


def test_start_over_removes_the_previous_run(fake_applio):
    settings = dict(rt.recommend_settings(1, 8), total_epoch=20, save_every_epoch=10)
    _train(fake_applio, settings)
    settings["total_epoch"] = 10
    voice_dir, lines, _ = _train(fake_applio, settings)
    assert any("Removing the previous training run" in line for line in lines)
    # Neither Applio's logs nor the installed voice keep the first run's checkpoints
    assert [os.path.basename(p) for p in rt.find_outputs(fake_applio["applio"], "Narrator")["checkpoints"]] == \
        ["Narrator_10e_100s.pth"]
    assert rt.list_checkpoints(voice_dir) == ["Narrator_10e_100s.pth"]
