# audiobook_exporter.py
'''
Joins the generated clips of an audiobook into a single file with ffmpeg.

- Sentences without generated audio are skipped (and counted) instead of breaking the export.
- Clips whose sample rate, channel count or sample format differ from the rest (for example
  when speakers use different engines) are converted first, because ffmpeg's concat demuxer
  silently produces garbled audio when they are mixed.
- Optional loudness normalisation (EBU R128, -18 LUFS, -3 dBTP), roughly what ACX asks for.
- mp3 (default), m4b (audiobook players remember your position) or wav output.
'''

import json
import ntpath
import os
import shutil
import subprocess
import tempfile
from collections import Counter

import numpy as np
import soundfile as sf

EXPORT_FORMATS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128k"],
    "m4b": ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"],
    "wav": ["-c:a", "pcm_s16le"],
}
LOUDNORM_FILTER = "loudnorm=I=-18:TP=-3:LRA=11"


def export_settings_from_global(global_settings):
    global_settings = global_settings or {}
    export_format = str(global_settings.get("export_format", "mp3")).lower()
    if export_format not in EXPORT_FORMATS:
        print(f"Unknown export_format '{export_format}', using mp3")
        export_format = "mp3"
    return {
        "format": export_format,
        "normalize_loudness": bool(global_settings.get("export_normalize_loudness", False)),
    }


def collect_audio_paths(directory_path, text_audio_map):
    '''Audio files in sentence order, plus the number of sentences that have no audio.'''
    paths = []
    missing = 0
    for key in sorted(text_audio_map, key=int):
        entry = text_audio_map[key]
        audio_path = entry.get("audio_path")
        # Clips live next to text_audio_map.json; only the file name is trusted so books
        # that were moved or made on another OS (\\ vs / separators) still export
        path = os.path.join(directory_path, ntpath.basename(audio_path)) if audio_path else ""
        if entry.get("generated") and path and os.path.exists(path):
            paths.append(path)
        else:
            missing += 1
    return paths, missing


def export_audiobook(directory_path, pause_duration, settings, ffmpeg="ffmpeg", run=None):
    '''Returns (output file name, number of sentences skipped because they have no audio).'''
    run = run or _run_ffmpeg
    audio_map_path = os.path.join(directory_path, "text_audio_map.json")
    if not os.path.exists(audio_map_path):
        raise FileNotFoundError("The selected directory is not a valid Audiobook Directory.")
    with open(audio_map_path, "r", encoding="utf-8") as f:
        text_audio_map = json.load(f)

    paths, missing = collect_audio_paths(directory_path, text_audio_map)
    if not paths:
        raise ValueError("No generated audio found in this audiobook yet.")

    exported_dir = os.path.join(directory_path, "exported_audiobooks")
    os.makedirs(exported_dir, exist_ok=True)
    output_path = _next_output_path(exported_dir, os.path.basename(os.path.normpath(directory_path)), settings["format"])

    work_dir = tempfile.mkdtemp(prefix="audiobook_export_")
    try:
        infos = {path: sf.info(path) for path in paths}
        target_rate, target_channels = Counter((info.samplerate, info.channels) for info in infos.values()).most_common(1)[0][0]
        concat_paths = []
        for i, path in enumerate(paths):
            info = infos[path]
            if (info.samplerate, info.channels, info.subtype) == (target_rate, target_channels, "PCM_16"):
                concat_paths.append(path)
            else:
                converted = os.path.join(work_dir, f"converted_{i}.wav")
                run([ffmpeg, "-y", "-v", "error", "-i", path, "-ar", str(target_rate), "-ac", str(target_channels),
                     "-c:a", "pcm_s16le", converted])
                concat_paths.append(converted)

        silence_path = None
        if pause_duration > 0:
            silence_path = os.path.join(work_dir, "silence.wav")
            frames = int(round(pause_duration * target_rate))
            sf.write(silence_path, np.zeros((frames, target_channels), dtype=np.int16), target_rate, subtype="PCM_16")

        file_list_path = os.path.join(work_dir, "file_list.txt")
        with open(file_list_path, "w", encoding="utf-8") as f:
            for i, path in enumerate(concat_paths):
                if i and silence_path:
                    f.write(f"file {_quote(silence_path)}\n")
                f.write(f"file {_quote(path)}\n")

        cmd = [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", file_list_path]
        if settings["normalize_loudness"]:
            # loudnorm resamples to 192 kHz internally, so set the output rate back explicitly
            cmd += ["-af", LOUDNORM_FILTER, "-ar", str(target_rate)]
        cmd += EXPORT_FORMATS[settings["format"]] + [output_path]
        run(cmd)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Combined audiobook saved in {output_path}")
    return os.path.basename(output_path), missing


def _next_output_path(exported_dir, book_name, extension):
    idx = 0
    while True:
        path = os.path.join(exported_dir, f"{book_name}_audiobook_{idx}.{extension}")
        if not os.path.exists(path):
            return path
        idx += 1


def _quote(path):
    # ffmpeg concat list syntax: single quotes, with embedded quotes written as '\''
    return "'" + os.path.abspath(path).replace("'", "'\\''") + "'"


def _run_ffmpeg(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-1000:]}")
