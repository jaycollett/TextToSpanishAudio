import itertools
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

# Segments shorter than this are folded into a neighbour. A lone "a" or "Amén."
# synthesizes fine, but as its own clip it gets a sentence-final pause and a
# reset intonation in the middle of running speech.
MIN_SEGMENT_CHARS = 40

_CLOSERS = "\"'”’»)\\]"

# Abbreviations whose period does not end a sentence, mostly titles and the
# Bible book abbreviations that turn up in sermon transcripts ("Jn. 3:16").
_ABBREVIATIONS = [
    "Sr", "Sra", "Srta", "Dr", "Dra", "Lic", "Cor", "Gn", "Ex", "Mt", "Mc", "Lc",
    "Jn", "Rom", "Ef", "Fil", "Col", "Ap", "vs", "cap", "núm", "p",
]
_NOT_ABBREVIATION = "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS)

# Boundaries, from most to least preferred. Each pattern matches the whitespace
# where a split may happen, so punctuation always stays with its own text.
#   1. after . ? ! ; or an ellipsis (optionally followed by a closing quote or
#      bracket), unless that is an abbreviation or the next word is a number
#      ("capítulo 22. 3"), or before an opening ¿ or ¡ (optionally after an
#      opening quote)
#   2. after a comma, colon, em or en dash, or a spaced hyphen
#   3. any whitespace, so no word is ever cut in half
_BOUNDARIES = [
    re.compile(
        rf"(?:(?<=[.?!;…])|(?<=[.?!;…][{_CLOSERS}])){_NOT_ABBREVIATION}\s+(?!\d)"
        r"|\s+(?=[\"'“«(]?[¿¡])"
    ),
    re.compile(rf"(?:(?<=[,:—–])|(?<=[,:—–][{_CLOSERS}])|(?<=\s-))\s+"),
    re.compile(r"\s+"),
]


def split_text(text, max_chars=MAX_SEGMENT_CHARS, min_chars=MIN_SEGMENT_CHARS):
    """
    Split text into segments of at most max_chars characters for synthesis.

    Segments break at sentence punctuation where possible, then at commas,
    then between words. Within those rules the text is cut into as few
    segments as possible, sized as evenly as that count allows, and a segment
    under min_chars is folded into a neighbour (unless the whole text is that
    short). Whitespace is collapsed, and segments with nothing speakable in
    them (empty, or punctuation only) are dropped, so the result is safe to
    feed straight to tts_to_file(). Returns an empty list when the input has
    no speakable text.
    """
    text = " ".join((text or "").split())
    if not text:
        return []
    segments = _fold_short(_split(text, max_chars, 0), max_chars, min_chars)
    return [s for s in segments if any(c.isalnum() for c in s)]


def _split(text, max_chars, level):
    if len(text) <= max_chars:
        return [text]
    if level == len(_BOUNDARIES):
        # A single "word" longer than max_chars (a pasted URL or similar).
        # Cutting it is the only way to keep the segment within bounds.
        count = -(-len(text) // max_chars)
        size, extra = divmod(len(text), count)
        cuts = [i * size + min(i, extra) for i in range(count + 1)]
        return [text[a:b] for a, b in itertools.pairwise(cuts)]

    pieces = []
    for piece in _BOUNDARIES[level].split(text):
        if piece:
            pieces.extend(_split(piece, max_chars, level + 1))
    return _pack(pieces, max_chars)


def _pack(pieces, max_chars):
    """
    Join pieces, in order, into the fewest segments of at most max_chars.

    Greedy filling gives the fewest segments but leaves whatever is left over
    in the last one, which can be a single word. Searching for the smallest
    limit that still gives that count spreads the text evenly instead.
    """
    fewest = len(_fill(pieces, max_chars))
    low, high = max(map(len, pieces)), max_chars
    while low < high:
        mid = (low + high) // 2
        if len(_fill(pieces, mid)) <= fewest:
            high = mid
        else:
            low = mid + 1
    return _fill(pieces, low)


def _fill(pieces, limit):
    segments = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if len(candidate) <= limit:
            current = candidate
        else:
            segments.append(current)
            current = piece
    segments.append(current)
    return segments


def _fold_short(segments, max_chars, min_chars):
    """
    Merge each segment under min_chars into its neighbour.

    This catches a short sentence sitting between two that are each nearly
    max_chars long. If the merged text is too long for one segment it is
    re-split evenly between words, which moves one break inside a sentence
    rather than leaving a fragment.
    """
    segments = list(segments)
    i = 0
    while i < len(segments):
        if len(segments) == 1 or len(segments[i]) >= min_chars:
            i += 1
            continue
        neighbours = [n for n in (i - 1, i + 1) if 0 <= n < len(segments)]
        fitting = [n for n in neighbours if len(segments[n]) + 1 + len(segments[i]) <= max_chars]
        n = min(fitting or neighbours, key=lambda n: len(segments[n]))
        first, last = min(i, n), max(i, n)
        merged = f"{segments[first]} {segments[last]}"
        if len(merged) <= max_chars:
            # The merged segment may itself still be short, so look at it again.
            segments[first:last + 1] = [merged]
            i = first
        else:
            replacement = _pack(merged.split(" "), max_chars)
            segments[first:last + 1] = replacement
            i = first + len(replacement)
    return segments
