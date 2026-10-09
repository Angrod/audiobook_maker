import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import qa_checker  # noqa: E402
from qa_checker import (  # noqa: E402
    check_clips,
    error_rate,
    language_for_speaker,
    normalize_tokens,
    number_to_words,
    qa_settings_from_global,
    speaking_rate_problem,
)


def test_normalize_ignores_case_punctuation_and_apostrophes():
    assert normalize_tokens("“Don’t STOP,” she said—now!") == ["dont", "stop", "she", "said", "now"]


def test_normalize_numbers_and_titles():
    assert normalize_tokens("Mr. Smith paid 1922 dollars") == ["mister", "smith", "paid", "one", "thousand",
                                                               "nine", "hundred", "twenty", "two", "dollars"]
    assert number_to_words(45) == "forty five"


def test_normalize_cjk_uses_characters():
    assert normalize_tokens("こんにちは、世界。", "ja") == list("こんにちは世界")


def test_error_rate_counts_repeats_and_skips():
    rate, errors = error_rate("the cat sat on the mat", "the cat sat on the mat")
    assert (rate, errors) == (0.0, 0)
    rate, errors = error_rate("the cat sat on the mat", "the cat cat cat sat on the mat")
    assert errors == 2
    rate, errors = error_rate("the cat sat on the mat", "the cat")
    assert errors == 4


def test_language_for_speaker():
    assert language_for_speaker({"tts_engine": "GPT_SoVITS", "gpt_sovits_output_lang": "all_ja"}, "en") == "ja"
    assert language_for_speaker({"tts_engine": "F5TTS"}, "en") == "en"
    assert language_for_speaker({}, "auto") is None


def test_speaking_rate_problem():
    sentence = "x" * 140  # about 10 seconds at 14 chars/s
    assert speaking_rate_problem(sentence, 10, 14) is None
    assert "too long" in speaking_rate_problem(sentence, 25, 14)
    assert "too short" in speaking_rate_problem(sentence, 3, 14)


def test_settings_defaults():
    settings = qa_settings_from_global({"qa_max_error_rate": "bad"})
    assert settings["qa_max_error_rate"] == 0.25
    assert settings["qa_whisper_model"] == "small"


def _make_map(tmp_path, sentences):
    text_audio_map = {}
    for idx, sentence in enumerate(sentences):
        path = tmp_path / f"audio_{idx}.wav"
        path.write_bytes(b"")
        text_audio_map[str(idx)] = {"sentence": sentence, "audio_path": str(path), "generated": True,
                                    "speaker_id": 1, "regen": False}
    return text_audio_map


def test_check_clips_flags_reuses_cache_and_unticks_fixed_clips(tmp_path):
    sentences = ["The quick brown fox jumps over the lazy dog.", "A second sentence that is fine.",
                 "Manually ticked sentence here."]
    text_audio_map = _make_map(tmp_path, sentences)
    text_audio_map["2"]["regen"] = True  # ticked by the user
    heard = {
        "audio_0.wav": "The quick quick quick brown fox jumps jumps over the dog.",
        "audio_1.wav": "A second sentence that is fine.",
        "audio_2.wav": "Manually ticked sentence here.",
    }
    calls = []

    def transcriber(path, language):
        calls.append(os.path.basename(path))
        return heard[os.path.basename(path)]

    settings = qa_settings_from_global({})
    speakers = {1: {"settings": {"tts_engine": "GPT_SoVITS", "gpt_sovits_output_lang": "en"}}}
    durations = lambda path: 3.0  # noqa: E731

    results = check_clips(text_audio_map, speakers, settings, transcriber, get_duration=durations)
    assert not results["0"]["passed"] and text_audio_map["0"]["regen"] is True
    assert results["1"]["passed"] and text_audio_map["1"]["regen"] is False
    assert results["2"]["passed"] and text_audio_map["2"]["regen"] is True  # user's tick is kept
    assert len(calls) == 3

    # Unchanged clips are not transcribed again
    calls.clear()
    check_clips(text_audio_map, speakers, settings, transcriber, get_duration=durations)
    assert calls == []

    # Clip 0 is regenerated and now passes: the tick this check added is removed
    os.utime(text_audio_map["0"]["audio_path"], (1, 1))
    heard["audio_0.wav"] = sentences[0]
    results = check_clips(text_audio_map, speakers, settings, transcriber, get_duration=durations)
    assert calls == ["audio_0.wav"]
    assert results["0"]["passed"] and text_audio_map["0"]["regen"] is False


def test_unreadable_clip_is_flagged_without_transcribing(tmp_path):
    text_audio_map = _make_map(tmp_path, ["A clip that was never written properly."])
    calls = []

    def transcriber(path, language):
        calls.append(path)
        return ""

    # Real soundfile on a zero-byte .wav raises; it must flag the clip instead of aborting the check
    results = check_clips(text_audio_map, {}, qa_settings_from_global({}), transcriber)
    assert results["0"]["reasons"] == ["audio file could not be read"]
    assert text_audio_map["0"]["regen"] is True
    assert calls == []


def test_write_report(tmp_path):
    text_audio_map = _make_map(tmp_path, ["Hello world again and again."])
    results = {"0": {"passed": False, "reasons": ["3 word errors (60%)"], "transcript": "Hello"}}
    path = qa_checker.write_report(str(tmp_path), text_audio_map, results)
    report = open(path, encoding="utf-8").read()
    assert "flagged 1" in report and "Sentence 1: 3 word errors" in report and "Whisper: Hello" in report
