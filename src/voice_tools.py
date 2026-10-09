# voice_tools.py
'''
Small audio measurements used to pick RVC settings automatically.

The main one is the speaker's typical pitch (median f0 over voiced frames). Comparing the
pitch of the TTS voice with the pitch of the voice an RVC model was trained on gives the
pitch shift that keeps the converted voice in its natural range.
'''

import json
import math
import os

VOICE_INFO_FILE = "voice_info.json"
MAX_SEMITONES = 16  # the Voice Pitch slider's range


def median_f0(path, max_seconds=30):
    """Typical pitch in Hz of the speech in an audio file, or None if no voiced speech is found."""
    import librosa  # imported lazily: it ships with the TTS engines, not the base app
    import numpy as np

    audio, sr = librosa.load(path, sr=16000, mono=True, duration=max_seconds)
    if audio.size == 0:
        return None
    f0, voiced, _ = librosa.pyin(audio, fmin=60.0, fmax=1000.0, sr=sr, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    if f0.size == 0:
        return None
    return float(np.median(f0))


def median_f0_of_files(paths, max_files=20, max_seconds_each=15):
    """Typical pitch across several clips (e.g. a training dataset): the median of per-clip medians."""
    values = []
    for path in sorted(paths)[:max_files]:
        try:
            value = median_f0(path, max_seconds=max_seconds_each)
        except Exception as e:
            print(f"Could not measure pitch of {path}: {e}")
            continue
        if value:
            values.append(value)
    if not values:
        return None
    values.sort()
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def suggest_semitones(source_hz, target_hz, limit=MAX_SEMITONES):
    """Pitch shift (semitones) that moves a voice at source_hz to target_hz, clamped to the slider."""
    if not source_hz or not target_hz or source_hz <= 0 or target_hz <= 0:
        return None
    semitones = round(12 * math.log2(target_hz / source_hz))
    return max(-limit, min(limit, semitones))


def load_voice_info(voice_dir):
    path = os.path.join(voice_dir, VOICE_INFO_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"Could not read {path}: {e}")
        return {}


def save_voice_info(voice_dir, updates):
    """Merges updates into the voice folder's voice_info.json."""
    info = load_voice_info(voice_dir)
    info.update(updates)
    os.makedirs(voice_dir, exist_ok=True)
    with open(os.path.join(voice_dir, VOICE_INFO_FILE), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    return info
