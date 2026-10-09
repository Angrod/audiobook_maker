import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from text_chunker import (  # noqa: E402
    chunk_line,
    chunk_settings_from_global,
    chunk_text,
    split_long_sentence,
    split_sentences,
)

SAMPLE_FILES = [
    os.path.join(ROOT, "input_text_files", name)
    for name in ("text.txt", "text_small.txt", "text_small_test.txt", "text_test1.txt")
]


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]


def test_split_sentences_keeps_abbreviations_and_initials():
    text = "Mr. Gatsby met Dr. Smith. F. Scott wrote it. I saw him, said I. Then we left."
    assert split_sentences(text) == [
        "Mr. Gatsby met Dr. Smith.",
        "F. Scott wrote it.",
        "I saw him, said I.",
        "Then we left.",
    ]


def test_split_sentences_handles_quotes():
    text = '"Stop!" she cried. "Why?" He turned. “Fine.” And so on.'
    assert split_sentences(text) == ['"Stop!" she cried.', '"Why?"', "He turned.", "“Fine.”", "And so on."]


def test_split_sentences_ignores_decimals():
    assert split_sentences("It cost 3.50 dollars. Cheap.") == ["It cost 3.50 dollars.", "Cheap."]


def test_short_sentences_are_packed_up_to_max_chars():
    line = "Yes. No. Maybe. " * 3
    chunks = chunk_line(line, 40)
    assert all(len(c) <= 40 for c in chunks)
    assert " ".join(chunks) == " ".join(line.split())


def test_long_sentence_split_prefers_clause_boundary():
    sentence = ("word " * 20).strip() + ", " + ("more " * 20).strip() + "."
    pieces = split_long_sentence(sentence, 120)
    assert len(pieces) == 2
    assert pieces[0].endswith(",")
    assert all(len(p) <= 120 for p in pieces)


def test_unbreakable_word_is_left_alone():
    word = "x" * 300
    assert chunk_line(word, 100) == [word]


def test_line_mode_matches_legacy_behaviour():
    text = "First line.\nSecond line that is long. With two sentences.\n\n***\n\nThird."
    assert chunk_text(text, {"mode": "line"}) == [
        "First line.",
        "Second line that is long. With two sentences.",
        "Third.",
    ]


def test_join_wrapped_lines():
    text = "This sentence was hard\nwrapped by the publisher. Another\nsentence.\n\nNew paragraph."
    chunks = chunk_text(text, {"mode": "sentence", "max_chars": 30, "join_wrapped_lines": True})
    assert chunks == ["This sentence was hard", "wrapped by the publisher.", "Another sentence.", "New paragraph."]


def test_settings_from_global_reads_flat_keys_and_validates():
    settings = chunk_settings_from_global({"chunk_mode": "bogus", "chunk_max_chars": "5"})
    assert settings["mode"] == "sentence"
    assert settings["max_chars"] == 20


@pytest.mark.parametrize("path", SAMPLE_FILES)
@pytest.mark.parametrize("max_chars", [80, 150, 250, 400])
@pytest.mark.parametrize("join", [False, True])
def test_chunking_is_idempotent_and_bounded(path, max_chars, join):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    settings = {"mode": "sentence", "max_chars": max_chars, "join_wrapped_lines": join}
    chunks = chunk_text(text, settings)
    # book_text.txt is written as chunks separated by blank lines and re-read on generation
    assert chunk_text("\n\n".join(chunks) + "\n\n", settings) == chunks
    assert all(len(c) <= max_chars or " " not in c for c in chunks)
    # No text is lost or added (chunks split at an unspaced dash have no space between them)
    kept_lines = [line for line in text.split("\n") if any(ch.isalpha() for ch in line)]
    assert "".join("".join(chunks).split()) == "".join("".join(kept_lines).split())
