# rvc_training.py
'''
Trains RVC voices with Applio (https://github.com/IAHispano/Applio) without leaving the app.

Applio runs in its own Python environment (installed by install_applio.bat, or an existing
install) and is driven through its command line, core.py:
    prerequisites -> preprocess -> extract -> train (train also builds the .index)

This module holds the decisions and bookkeeping, kept free of Qt so it can be tested:
which settings to use for a dataset and GPU, the exact Applio commands, reading progress
from Applio's output, finding the results, and installing them as an app voice in
engines/rvc/<name>/ (one .pth + .index, other checkpoints in a checkpoints/ subfolder,
because rvc-python loads the first .pth it finds in a voice folder).
'''

import glob
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime

import voice_tools

APPLIO_VERSION = "3.6.5"
DEFAULT_APPLIO_PATH = os.path.join("modules", "Applio")
RVC_VOICES_DIR = os.path.join("engines", "rvc")
DATASET_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")  # what Applio's preprocess reads
SAMPLE_RATE = 40000
# Fixed so rvc-python (the app's RVC player) can load the result: it only knows the
# original HiFi-GAN vocoder and contentvec features.
VOCODER = "HiFi-GAN"
EMBEDDER = "contentvec"
F0_METHOD = "rmvpe"
CHECKPOINT_PATTERN = re.compile(r"_(\d+)e_(\d+)s\.pth$")
PROGRESS_PATTERN = re.compile(r"\bepoch=(\d+)\b")
VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class TrainingCancelled(Exception):
    pass


# ---------------------------------------------------------------- Applio install

def applio_python(applio_path):
    if os.name == "nt":
        return os.path.join(applio_path, "env", "python.exe")
    return os.path.join(applio_path, "env", "bin", "python")


def applio_problems(applio_path):
    """What is missing for training with the Applio install at applio_path ([] if nothing)."""
    problems = []
    if not os.path.isfile(os.path.join(applio_path, "core.py")):
        problems.append(f"Applio was not found in {applio_path}.")
    elif not os.path.isfile(applio_python(applio_path)):
        problems.append(f"Applio in {applio_path} is downloaded but not installed (no env folder). Run its run-install.bat.")
    return problems


def missing_prerequisites(applio_path, sample_rate=SAMPLE_RATE):
    """Model files Applio downloads with `core.py prerequisites`. Without the pretrained ones
    Applio silently trains from scratch, which needs far more data and time."""
    rate = str(sample_rate)[:2]
    required = [
        os.path.join("rvc", "models", "pretraineds", "hifi-gan", f"f0G{rate}k.pth"),
        os.path.join("rvc", "models", "pretraineds", "hifi-gan", f"f0D{rate}k.pth"),
        os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
        os.path.join("rvc", "models", "embedders", "contentvec", "pytorch_model.bin"),
    ]
    return [path for path in required if not os.path.isfile(os.path.join(applio_path, path))]


# ---------------------------------------------------------------- dataset

def validate_voice_name(name):
    if not VALID_NAME.match(name or ""):
        return "Use letters, numbers, - and _ only (no spaces), starting with a letter or number."
    return None


def dataset_files(path):
    if os.path.isfile(path):
        return [path] if path.lower().endswith(DATASET_EXTENSIONS) else []
    files = []
    for root, _, names in os.walk(path):
        files.extend(os.path.join(root, n) for n in names if n.lower().endswith(DATASET_EXTENSIONS))
    return sorted(files)


def audio_seconds(path):
    import soundfile as sf
    try:
        return sf.info(path).duration
    except Exception:
        from pydub import AudioSegment  # formats libsndfile cannot read
        return len(AudioSegment.from_file(path)) / 1000.0


def dataset_summary(path):
    """Minutes of usable audio, number of files, and other files Applio will ignore."""
    files = dataset_files(path)
    seconds = 0.0
    unreadable = []
    for f in files:
        try:
            seconds += audio_seconds(f)
        except Exception:
            unreadable.append(f)
    ignored = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            ignored.extend(n for n in names if n.lower().endswith((".m4a", ".aac", ".opus", ".wma")))
    return {"minutes": seconds / 60.0, "files": len(files) - len(unreadable), "unreadable": unreadable,
            "ignored_formats": ignored}


def prepare_dataset(applio_path, name, path):
    """Applio's preprocess needs a folder; a single long file is copied into one."""
    if os.path.isdir(path):
        return os.path.abspath(path)
    folder = os.path.join(applio_path, "assets", "datasets", name)
    os.makedirs(folder, exist_ok=True)
    shutil.copy2(path, os.path.join(folder, os.path.basename(path)))
    return os.path.abspath(folder)


# ---------------------------------------------------------------- settings

def detect_gpu():
    """{'name', 'vram_gb'} for the first CUDA GPU, or None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return {"name": props.name, "vram_gb": props.total_memory / 1024 ** 3}
    except Exception:
        return None


def recommend_settings(minutes, vram_gb, noisy=False):
    """Starting-point settings from the dataset length and GPU memory (community guidance:
    batch size by VRAM, fewer epochs for longer datasets, pretrained base models, 40 kHz)."""
    warnings = []
    if vram_gb is None:
        warnings.append("No NVIDIA GPU was found. RVC training needs one (6 GB of VRAM or more).")
        batch_size = 4
    elif vram_gb < 7:
        batch_size = 4
    elif vram_gb < 11:
        batch_size = 6
    elif vram_gb < 16:
        batch_size = 8
    else:
        batch_size = 12
    if vram_gb is not None and vram_gb < 5.5:
        warnings.append(f"Your GPU has {vram_gb:.0f} GB of VRAM; training may run out of memory.")

    if minutes <= 10:
        epochs = 300
    elif minutes <= 20:
        epochs = 250
    elif minutes <= 40:
        epochs = 200
    elif minutes <= 60:
        epochs = 150
    else:
        epochs = 100
    if minutes < 1:
        warnings.append("Less than a minute of audio is not enough to train a voice.")
    elif minutes < 5:
        warnings.append("Under 5 minutes of audio gives rough results; 10 to 30 minutes is recommended.")
    elif minutes > 60:
        warnings.append("Over an hour of audio makes training long; about 30 minutes is usually enough.")

    # About six saved checkpoints to compare by ear, in case the last one is over-trained
    save_every = min(100, max(10, int(round(epochs / 6 / 5)) * 5))
    return {
        "batch_size": batch_size,
        "total_epoch": epochs,
        "save_every_epoch": save_every,
        "sample_rate": SAMPLE_RATE,
        "f0_method": F0_METHOD,
        "embedder_model": EMBEDDER,
        "vocoder": VOCODER,
        "noise_reduction": bool(noisy),
        "can_train": vram_gb is not None and minutes >= 1,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- commands and progress

def build_steps(applio_path, name, dataset_dir, settings, resume=False):
    """[(label, argv)] to run with cwd=applio_path. Resuming skips straight to training,
    which picks up from the latest saved checkpoint."""
    if settings["vocoder"] != VOCODER or settings["embedder_model"] != EMBEDDER:
        raise ValueError("The app's RVC player needs the HiFi-GAN vocoder and contentvec embedder.")
    python = applio_python(applio_path)
    rate = str(settings["sample_rate"])
    steps = []
    if missing_prerequisites(applio_path, settings["sample_rate"]):
        steps.append(("Downloading Applio's base models", [python, "core.py", "prerequisites"]))
    if not resume:
        preprocess = [python, "core.py", "preprocess", "--model-name", name, "--dataset-path", dataset_dir,
                      "--sample-rate", rate, "--cut-preprocess", "Automatic"]
        if settings.get("noise_reduction"):
            preprocess.append("--noise-reduction")
        steps.append(("Preparing the dataset", preprocess))
        steps.append(("Extracting pitch and voice features",
                      [python, "core.py", "extract", "--model-name", name, "--f0-method", settings["f0_method"],
                       "--gpu", "0", "--sample-rate", rate, "--embedder-model", settings["embedder_model"]]))
    steps.append(("Training", [
        python, "core.py", "train", "--model-name", name, "--vocoder", settings["vocoder"],
        "--save-every-epoch", str(settings["save_every_epoch"]), "--total-epoch", str(settings["total_epoch"]),
        "--sample-rate", rate, "--batch-size", str(settings["batch_size"]), "--gpu", "0",
        "--index-algorithm", "Auto"]))
    return steps


def parse_epoch(line):
    match = PROGRESS_PATTERN.search(line)
    return int(match.group(1)) if match else None


def run_steps(steps, applio_path, on_line, should_stop, on_process=None):
    """Runs each step's command in Applio's folder, passing every output line to on_line.
    Raises TrainingCancelled when should_stop() turns true, RuntimeError when a step fails."""
    env = dict(os.environ)
    for var in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):  # keep the app's venv out of Applio's Python
        env.pop(var, None)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    for label, argv in steps:
        on_line(f"=== {label} ===")
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(argv, cwd=applio_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs)
        if on_process:
            on_process(process)
        try:
            for line in process.stdout:
                on_line(line.rstrip("\n"))
                if should_stop():
                    raise TrainingCancelled()
            process.wait()
        except TrainingCancelled:
            kill_process_tree(process)
            raise
        if should_stop():
            raise TrainingCancelled()
        if process.returncode != 0:
            raise RuntimeError(f"{label} failed (exit code {process.returncode}). The log above has the details.")


def kill_process_tree(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except Exception:
        process.kill()


# ---------------------------------------------------------------- results

def experiment_dir(applio_path, name):
    return os.path.join(applio_path, "logs", name)


def can_resume(applio_path, name):
    return bool(glob.glob(os.path.join(experiment_dir(applio_path, name), "G_*.pth")))


def checkpoint_epoch(path):
    match = CHECKPOINT_PATTERN.search(os.path.basename(path))
    return int(match.group(1)) if match else -1


def find_outputs(applio_path, name):
    """Saved voice checkpoints (oldest first) and the index Applio built for `name`."""
    folder = experiment_dir(applio_path, name)
    checkpoints = sorted((p for p in glob.glob(os.path.join(folder, f"{glob.escape(name)}_*e_*s.pth"))
                          if checkpoint_epoch(p) >= 0), key=checkpoint_epoch)
    index = os.path.join(folder, f"{name}.index")
    return {"checkpoints": checkpoints, "index": index if os.path.isfile(index) else None}


def import_voice(applio_path, name, voices_dir=RVC_VOICES_DIR, info=None):
    """Installs the trained voice as engines/rvc/<name>/ with the last checkpoint active."""
    outputs = find_outputs(applio_path, name)
    if not outputs["checkpoints"]:
        raise RuntimeError("Training finished without saving a voice model. The log above shows why.")
    if not outputs["index"]:
        raise RuntimeError("Training finished but the .index file was not created. The log above shows why.")
    voice_dir = os.path.join(voices_dir, name)
    checkpoints_dir = os.path.join(voice_dir, "checkpoints")
    if os.path.isdir(checkpoints_dir):  # checkpoints from an earlier training run of this voice
        shutil.rmtree(checkpoints_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)
    for checkpoint in outputs["checkpoints"]:
        shutil.copy2(checkpoint, os.path.join(checkpoints_dir, os.path.basename(checkpoint)))
    shutil.copy2(outputs["index"], os.path.join(voice_dir, f"{name}.index"))
    voice_tools.save_voice_info(voice_dir, dict(info or {}, trained_with=f"Applio {APPLIO_VERSION}",
                                                trained_on=datetime.now().isoformat(timespec="seconds")))
    use_checkpoint(voice_dir, os.path.basename(outputs["checkpoints"][-1]))
    return voice_dir


def list_checkpoints(voice_dir):
    folder = os.path.join(voice_dir, "checkpoints")
    if not os.path.isdir(folder):
        return []
    return sorted((n for n in os.listdir(folder) if checkpoint_epoch(n) >= 0), key=checkpoint_epoch)


def use_checkpoint(voice_dir, checkpoint_name):
    """Makes checkpoints/<checkpoint_name> the voice's model (the only .pth in the voice folder)."""
    source = os.path.join(voice_dir, "checkpoints", checkpoint_name)
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    for old in glob.glob(os.path.join(voice_dir, "*.pth")):
        os.remove(old)
    shutil.copy2(source, os.path.join(voice_dir, f"{os.path.basename(os.path.normpath(voice_dir))}.pth"))
    voice_tools.save_voice_info(voice_dir, {"active_checkpoint": checkpoint_name,
                                            "active_epoch": checkpoint_epoch(checkpoint_name)})


def train_voice(applio_path, name, dataset_path, settings, on_line, on_progress, should_stop,
                resume=False, voices_dir=RVC_VOICES_DIR, on_process=None):
    """The whole pipeline: prepare data, run Applio, install the voice. Returns the voice folder.
    Without resume, a previous run with the same name is deleted first: Applio would otherwise
    continue from its checkpoints and keep its old .index."""
    if not resume and os.path.isdir(experiment_dir(applio_path, name)):
        on_line(f"Removing the previous training run for {name}")
        shutil.rmtree(experiment_dir(applio_path, name))
    dataset_dir = prepare_dataset(applio_path, name, dataset_path)
    steps = build_steps(applio_path, name, dataset_dir, settings, resume=resume)
    total = settings["total_epoch"]
    training_started = [False]

    def handle_line(line):
        on_line(line)
        if line.startswith("=== Training"):
            training_started[0] = True
        epoch = parse_epoch(line)
        if training_started[0] and epoch is not None:
            on_progress(min(99, int(epoch / total * 100)), f"Training: epoch {epoch} of {total}")
        elif line.startswith("=== "):
            on_progress(None, line.strip("= "))

    run_steps(steps, applio_path, handle_line, should_stop, on_process=on_process)
    on_progress(99, "Installing the voice")
    info = {"dataset_minutes": round(dataset_summary(dataset_path)["minutes"], 1), "sample_rate": settings["sample_rate"],
            "total_epoch": total, "batch_size": settings["batch_size"]}
    try:
        info["target_median_f0"] = voice_tools.median_f0_of_files(dataset_files(dataset_path))
    except Exception as e:  # the pitch suggestion is optional
        on_line(f"Could not measure the voice's pitch: {e}")
    voice_dir = import_voice(applio_path, name, voices_dir, info)
    on_progress(100, "Done")
    return voice_dir


if __name__ == "__main__":  # quick check of an Applio install: python src/rvc_training.py [applio_path]
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_APPLIO_PATH
    print(applio_problems(path) or "Applio found", missing_prerequisites(path) or "base models present")
