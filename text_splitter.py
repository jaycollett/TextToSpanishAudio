import re

# Upper bound on the characters handed to a single tts_to_file() call.
#
# VITS has no hard input-length limit (its text encoder uses windowed relative
# attention), but memory scales with the length of one call, because the
# HiFi-GAN decoder materialises the whole waveform at 32+ channels per sample
# before returning. Measured peak memory for a single call with this model
# (CPU, fp32, model alone about 0.8 GB):
#     400 chars ->  2.4 GB      3,200 chars ->  9.9 GB
#   1,000 chars ->  5.4 GB      6,500 chars -> 16.4 GB
# GPU activation memory grows the same way, so the 3,000-6,500 character
# run-ons that failed were asking a shared GPU for many gigabytes. The model
# speaks about 20 characters per second, so 400 characters is about 20 seconds
# of audio: two or three ordinary sentences per segment, with a small, flat
# memory ceiling however long the transcript is.
MAX_SEGMENT_CHARS = 400

_CLOSERS = "\"'”’»)\\]"

# Boundaries, from most to least preferred. Each pattern matches the whitespace
# where a split may happen, so punctuation always stays with its own text.
#   1. after . ? ! ; or an ellipsis (optionally followed by a closing quote or
#      bracket), or before an opening ¿ or ¡ (optionally after an opening quote)
#   2. after a comma, colon or em dash
#   3. any whitespace, so no word is ever cut in half
_BOUNDARIES = [
    re.compile(
        rf"(?:(?<=[.?!;…])|(?<=[.?!;…][{_CLOSERS}]))\s+"
        r"|\s+(?=[\"'“«(]?[¿¡])"
    ),
    re.compile(rf"(?:(?<=[,:—])|(?<=[,:—][{_CLOSERS}]))\s+"),
    re.compile(r"\s+"),
]


def split_text(text, max_chars=MAX_SEGMENT_CHARS):
    """
    Split text into segments of at most max_chars characters for synthesis.

    Segments break at sentence punctuation where possible, then at commas,
    then between words. Whitespace is collapsed, and segments with nothing
    speakable in them (empty, or punctuation only) are dropped, so the result
    is safe to feed straight to tts_to_file(). Returns an empty list when the
    input has no speakable text.
    """
    text = " ".join((text or "").split())
    if not text:
        return []
    return [s for s in _split(text, max_chars, 0) if any(c.isalnum() for c in s)]


def _split(text, max_chars, level):
    if len(text) <= max_chars:
        return [text]
    if level == len(_BOUNDARIES):
        # A single "word" longer than max_chars (a pasted URL or similar).
        # Cutting it is the only way to keep the segment within bounds.
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    segments = []
    current = ""
    for piece in _BOUNDARIES[level].split(text):
        if not piece:
            continue
        if len(piece) > max_chars:
            if current:
                segments.append(current)
                current = ""
            segments.extend(_split(piece, max_chars, level + 1))
            continue
        candidate = f"{current} {piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            segments.append(current)
            current = piece
    if current:
        segments.append(current)
    return segments
