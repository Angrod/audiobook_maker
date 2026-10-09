# qa_checker.py
'''
Finds generated clips that probably do not say what the text says, so they can be
regenerated instead of having to listen to the whole audiobook.

Each clip is transcribed with Whisper (faster-whisper) and compared with its chunk text:
    - Error rate: word error rate (character error rate for Chinese/Japanese/Korean).
      Catches repeated, skipped, swapped or invented words.
    - Speaking rate: characters per second compared with the median of the whole book.
      Catches long babbling/silence (too slow) and clips that were cut off (too fast).

Results are cached per clip (audio file modification time + sentence), so re-running the
check after "Regenerate in Bulk" only transcribes the clips that changed.
'''

import os
import re
import statistics
import unicodedata

DEFAULT_QA_SETTINGS = {
    "qa_whisper_model": "small",
    "qa_device": "auto",
    "qa_language": "en",
    "qa_max_error_rate": 0.25,
}

# A clip with fewer errors than this is never flagged on error rate alone, so a single
# misheard word in a three word sentence does not count as a failure
MIN_ERRORS_TO_FLAG = 2
SLOW_FACTOR = 2.0
FAST_FACTOR = 0.5
DEFAULT_CHARS_PER_SECOND = 14.0
MIN_CLIPS_FOR_MEDIAN = 10

CHARACTER_LANGUAGES = {"zh", "ja", "ko", "yue"}

# GPT-SoVITS output language -> Whisper language
GPT_SOVITS_LANGUAGES = {
    "en": "en", "zh": "zh", "all_zh": "zh", "ja": "ja", "all_ja": "ja",
    "ko": "ko", "all_ko": "ko", "yue": "yue", "all_yue": "yue",
}

WORD_EQUIVALENTS = {"mr": "mister", "mrs": "missus", "misses": "missus", "ms": "miz", "dr": "doctor"}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def qa_settings_from_global(global_settings):
    global_settings = global_settings or {}
    settings = {key: global_settings.get(key, default) for key, default in DEFAULT_QA_SETTINGS.items()}
    try:
        settings["qa_max_error_rate"] = float(settings["qa_max_error_rate"])
    except (TypeError, ValueError):
        settings["qa_max_error_rate"] = DEFAULT_QA_SETTINGS["qa_max_error_rate"]
    return settings


def language_for_speaker(speaker_settings, default_language):
    '''Whisper language code for a speaker, or None to let Whisper detect it.'''
    gpt_sovits_language = (speaker_settings or {}).get("gpt_sovits_output_lang")
    if (speaker_settings or {}).get("tts_engine", "").lower() == "gpt_sovits" and gpt_sovits_language in GPT_SOVITS_LANGUAGES:
        return GPT_SOVITS_LANGUAGES[gpt_sovits_language]
    if not default_language or default_language == "auto":
        return None
    return default_language


def number_to_words(number):
    if number < 20:
        return _ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {number_to_words(rest)}" if rest else "")
    if number < 1_000_000:
        thousands, rest = divmod(number, 1000)
        return f"{number_to_words(thousands)} thousand" + (f" {number_to_words(rest)}" if rest else "")
    return str(number)


def normalize_tokens(text, language="en"):
    '''Lowercased words (or characters for CJK languages) with punctuation removed.'''
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"['’‘`]", "", text)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    if language in CHARACTER_LANGUAGES:
        return [ch for ch in text if not ch.isspace()]
    tokens = []
    for word in text.split():
        word = WORD_EQUIVALENTS.get(word, word)
        if language in (None, "en") and word.isdigit() and len(word) <= 6:
            tokens.extend(number_to_words(int(word)).split())
        else:
            tokens.append(word)
    return tokens


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_token in enumerate(hypothesis, start=1):
            current[j] = min(
                previous[j] + 1,                                # deletion
                current[j - 1] + 1,                             # insertion
                previous[j - 1] + (ref_token != hyp_token),     # substitution
            )
        previous = current
    return previous[-1]


def error_rate(reference_text, hypothesis_text, language="en"):
    '''Returns (error_rate, number_of_errors) between what should have been said and what was heard.'''
    reference = normalize_tokens(reference_text, language)
    hypothesis = normalize_tokens(hypothesis_text, language)
    errors = edit_distance(reference, hypothesis)
    return errors / max(1, len(reference)), errors


def median_chars_per_second(durations_by_sentence):
    rates = [len(sentence) / duration for sentence, duration in durations_by_sentence
             if duration > 0 and len(sentence) >= 20]
    if len(rates) < MIN_CLIPS_FOR_MEDIAN:
        return DEFAULT_CHARS_PER_SECOND
    return statistics.median(rates)


def speaking_rate_problem(sentence, duration, median_rate):
    '''Describes a clip that is much longer or shorter than the text explains, else None.'''
    expected = len(sentence) / median_rate
    if duration > expected * SLOW_FACTOR + 1.5:
        return f"too long ({duration:.1f}s, expected about {expected:.1f}s): extra speech or long silence"
    if duration < expected * FAST_FACTOR - 0.5:
        return f"too short ({duration:.1f}s, expected about {expected:.1f}s): words may be missing"
    return None


def evaluate_clip(sentence, transcript, duration, language, median_rate, max_error_rate):
    rate, errors = error_rate(sentence, transcript, language)
    reasons = []
    if rate > max_error_rate and errors >= MIN_ERRORS_TO_FLAG:
        reasons.append(f"{errors} word errors ({rate:.0%})" if language not in CHARACTER_LANGUAGES
                       else f"{errors} character errors ({rate:.0%})")
    rate_problem = speaking_rate_problem(sentence, duration, median_rate)
    if rate_problem:
        reasons.append(rate_problem)
    return {
        "passed": not reasons,
        "error_rate": round(rate, 3),
        "errors": errors,
        "duration": round(duration, 2),
        "transcript": transcript,
        "reasons": reasons,
    }


def audio_duration(path):
    import soundfile as sf
    return sf.info(path).duration


def check_clips(text_audio_map, speakers, settings, transcriber, report_progress=None, should_stop=None,
                get_duration=audio_duration):
    '''
    Checks every generated clip in text_audio_map. Returns {idx: result} for the clips checked
    (or reused from cache) and stores each result under entry["qa"].
    transcriber(path, language) -> text.

    Failed clips get their "regen" box ticked. A clip that passes is only unticked if this
    check is what ticked it, so boxes ticked by hand are never cleared.
    '''
    clips = []
    for idx, entry in sorted(text_audio_map.items(), key=lambda item: int(item[0])):
        path = entry.get("audio_path")
        if entry.get("generated") and path and os.path.exists(path):
            clips.append((idx, entry, path))

    durations = {}
    for idx, _, path in clips:
        try:
            durations[idx] = get_duration(path)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            durations[idx] = None
    median_rate = median_chars_per_second([(entry["sentence"], durations[idx]) for idx, entry, _ in clips
                                           if durations[idx] is not None])

    results = {}
    for count, (idx, entry, path) in enumerate(clips, start=1):
        if should_stop and should_stop():
            break
        mtime = os.path.getmtime(path)
        cached = entry.get("qa")
        speaker_settings = (speakers.get(entry.get("speaker_id", 1)) or {}).get("settings", {})
        language = language_for_speaker(speaker_settings, settings["qa_language"])
        if durations[idx] is None:
            result = {"passed": False, "error_rate": 1.0, "errors": 0, "duration": 0.0, "transcript": "",
                      "reasons": ["audio file could not be read"]}
        else:
            if cached and cached.get("mtime") == mtime and cached.get("sentence") == entry["sentence"] and "transcript" in cached:
                transcript = cached["transcript"]
            else:
                transcript = transcriber(path, language)
            result = evaluate_clip(entry["sentence"], transcript, durations[idx], language,
                                   median_rate, settings["qa_max_error_rate"])
        result["mtime"] = mtime
        result["sentence"] = entry["sentence"]
        ticked_by_previous_check = bool(cached and cached.get("ticked_regen"))
        if not result["passed"]:
            if not entry.get("regen"):
                entry["regen"] = True
                result["ticked_regen"] = True
            else:
                result["ticked_regen"] = ticked_by_previous_check
        elif ticked_by_previous_check and entry.get("regen"):
            entry["regen"] = False
        entry["qa"] = result
        results[idx] = result
        if report_progress:
            report_progress(int(count / len(clips) * 100))
    return results


class WhisperTranscriber:
    '''Loads faster-whisper lazily so the app works without it installed.'''
    def __init__(self, model_size="small", device="auto"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. Install it with:  pip install faster-whisper"
            ) from e
        _use_torch_cuda_libraries()
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        try:
            try:
                self.model = self._load(WhisperModel, model_size, device)
            except Exception as e:
                if device == "cpu":
                    raise
                print(f"Could not load Whisper on {device} ({e}); falling back to CPU")
                self.model = self._load(WhisperModel, model_size, "cpu")
        except Exception as e:
            raise RuntimeError(
                f"Could not load the Whisper model '{model_size}'. The first run downloads it from "
                f"Hugging Face, so check your internet connection, or set qa_whisper_model in "
                f"configs/settings.yaml to a local model folder.\n\nDetails: {e}"
            ) from e

    @staticmethod
    def _load(WhisperModel, model_size, device):
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Loading Whisper '{model_size}' on {device} ({compute_type})")
        return WhisperModel(model_size, device=device, compute_type=compute_type)

    def __call__(self, path, language):
        # condition_on_previous_text=False and no VAD so Whisper reports what is actually in
        # the clip, including babble and repeats, instead of smoothing them over
        segments, _ = self.model.transcribe(
            path, language=language, beam_size=5, condition_on_previous_text=False, vad_filter=False
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def _use_torch_cuda_libraries():
    '''On Windows, let faster-whisper find the cuDNN/cuBLAS DLLs that ship with PyTorch.'''
    if os.name != "nt":
        return
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _cuda_available():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def write_report(directory_path, text_audio_map, results):
    '''Writes qa_report.txt listing flagged clips; returns its path.'''
    report_path = os.path.join(directory_path, "qa_report.txt")
    flagged = [(idx, results[idx]) for idx in sorted(results, key=int) if not results[idx]["passed"]]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Checked {len(results)} clips, flagged {len(flagged)}.\n\n")
        for idx, result in flagged:
            f.write(f"Sentence {int(idx) + 1}: {'; '.join(result['reasons'])}\n")
            f.write(f"  Text:    {text_audio_map[idx]['sentence']}\n")
            f.write(f"  Whisper: {result['transcript']}\n\n")
    return report_path
