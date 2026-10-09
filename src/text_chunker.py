# text_chunker.py
'''
Splits book text into chunks that are short enough for TTS models to read without
hallucinating (repeating, skipping or inventing words), while keeping whole sentences
together whenever possible.

Two modes:
    "line"     - Legacy behaviour: every non-empty line is one chunk, regardless of length.
    "sentence" - Each line is split into sentences, then neighbouring sentences are packed
                 together up to `max_chars`.  A single sentence longer than `max_chars` is
                 split at a clause boundary (comma, semicolon, dash...) or, failing that,
                 at the space closest to its middle.

Chunking is idempotent: chunking the "\n\n"-joined output again returns the same chunks.
This matters because book_text.txt is written that way and is re-read on generation.
'''

import re

DEFAULT_CHUNK_SETTINGS = {
    "mode": "sentence",
    "max_chars": 250,
    "join_wrapped_lines": False,
}

VALID_MODES = ("sentence", "line")

# Words that end in a period but do not end a sentence ("Mr. Gatsby", "e.g. this")
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev", "gen", "col", "capt", "lt",
    "sgt", "mt", "ft", "vs", "etc", "e.g", "i.e", "approx", "dept", "fig", "vol", "ch", "pp",
}

# Sentence end: terminal punctuation, optional closing quotes/brackets, then whitespace
_SENTENCE_END = re.compile(r'[.!?…]+["\'”’»)\]]*(?=\s)')
# What the next sentence is allowed to start with
_SENTENCE_START = re.compile(r'\s+["\'“‘«(\[]*[A-Z0-9―—–-]')
# Clause boundaries that are good places to break a sentence that is too long
_CLAUSE_BREAK = re.compile(r'[,;:]["\'”’)]*(?=\s)|[—–―]')


def chunk_settings_from_global(global_settings):
    '''Builds chunk settings from the flat keys in configs/settings.yaml.'''
    global_settings = global_settings or {}
    settings = dict(DEFAULT_CHUNK_SETTINGS)
    settings["mode"] = global_settings.get("chunk_mode", settings["mode"])
    settings["max_chars"] = global_settings.get("chunk_max_chars", settings["max_chars"])
    settings["join_wrapped_lines"] = global_settings.get("chunk_join_wrapped_lines", settings["join_wrapped_lines"])
    return normalize_chunk_settings(settings)


def normalize_chunk_settings(settings):
    settings = {**DEFAULT_CHUNK_SETTINGS, **(settings or {})}
    if settings["mode"] not in VALID_MODES:
        print(f"Unknown chunk mode '{settings['mode']}', falling back to 'sentence'")
        settings["mode"] = "sentence"
    try:
        settings["max_chars"] = max(20, int(settings["max_chars"]))
    except (TypeError, ValueError):
        settings["max_chars"] = DEFAULT_CHUNK_SETTINGS["max_chars"]
    settings["join_wrapped_lines"] = bool(settings["join_wrapped_lines"])
    return settings


def chunk_text(text, settings=None):
    '''Splits the full text of a book into a list of chunks.'''
    settings = normalize_chunk_settings(settings)
    chunks = []
    for paragraph in text.split('\n\n'):
        if settings["mode"] == "line":
            chunks.extend(_filter_lines(paragraph))
            continue
        if settings["join_wrapped_lines"]:
            lines = [' '.join(_filter_lines(paragraph))]
        else:
            lines = _filter_lines(paragraph)
        for line in lines:
            if line:
                chunks.extend(chunk_line(line, settings["max_chars"]))
    return chunks


def chunk_line(line, max_chars):
    '''Splits one line of text into chunks of at most max_chars, keeping sentences whole when possible.'''
    line = ' '.join(line.split())
    # Each piece remembers the separator that preceded it in the original text, so that
    # re-merging "horizon—" and "for" gives back "horizon—for" rather than "horizon— for"
    pieces = []
    for sentence in split_sentences(line):
        pieces.extend(_split_long_sentence(sentence, max_chars, " "))

    chunks = []
    current = ""
    for separator, piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(separator) + len(piece) <= max_chars:
            current = f"{current}{separator}{piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def split_sentences(text):
    '''Splits text into sentences on terminal punctuation, ignoring common abbreviations and initials.'''
    sentences = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if not _SENTENCE_START.match(text, end):
            continue
        if match.group().startswith('.') and _is_abbreviation(text[start:match.start()]):
            continue
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def split_long_sentence(sentence, max_chars):
    '''Recursively halves a sentence that is longer than max_chars, preferring clause boundaries.'''
    return [piece for _, piece in _split_long_sentence(sentence, max_chars, " ")]


def _split_long_sentence(sentence, max_chars, separator):
    if len(sentence) <= max_chars:
        return [(separator, sentence)]
    split_at = _find_split_point(sentence)
    if split_at is None:
        # A single "word" longer than max_chars (e.g. a URL); nothing sensible to do
        return [(separator, sentence)]
    left = sentence[:split_at].strip()
    right = sentence[split_at:].strip()
    if not left or not right:
        return [(separator, sentence)]
    right_separator = " " if sentence[split_at:split_at + 1].isspace() else ""
    return (_split_long_sentence(left, max_chars, separator)
            + _split_long_sentence(right, max_chars, right_separator))


def _find_split_point(sentence):
    length = len(sentence)
    middle = length / 2
    # Clause boundaries in the middle 40% of the sentence give balanced halves
    clause_points = [m.end() for m in _CLAUSE_BREAK.finditer(sentence)
                     if 0.3 * length <= m.end() <= 0.7 * length]
    if clause_points:
        return min(clause_points, key=lambda p: abs(p - middle))
    space_points = [m.start() for m in re.finditer(r'\s', sentence)]
    if space_points:
        return min(space_points, key=lambda p: abs(p - middle))
    return None


def _is_abbreviation(text_before_period):
    words = text_before_period.split()
    if not words:
        return False
    word = words[-1].lstrip('"\'“‘(«[').lower()
    if word in ABBREVIATIONS:
        return True
    # Single-letter initials such as "F. Scott Fitzgerald" (but not the pronoun "I")
    return len(word) == 1 and word.isalpha() and word != "i"


def _filter_lines(paragraph):
    '''Same filtering the app has always used: stripped, non-empty lines containing a letter.'''
    lines = []
    for line in paragraph.split('\n'):
        line = line.strip()
        if line and any(c.isalpha() for c in line):
            lines.append(line)
    return lines
