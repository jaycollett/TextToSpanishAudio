import pytest
from helpers import run_on

from text_splitter import MAX_SEGMENT_CHARS, MIN_SEGMENT_CHARS, split_text


def assert_well_formed(segments, text, max_chars=MAX_SEGMENT_CHARS):
    assert segments
    assert all(s.strip() == s and s for s in segments)
    assert all(len(s) <= max_chars for s in segments)
    # Splits only ever happen at whitespace, so rejoining with single spaces
    # gives back the whitespace-normalised input: no word was cut or dropped.
    assert " ".join(segments) == " ".join(text.split())


def test_split_text_run_on_without_punctuation_stays_under_max():
    text = run_on(6526)
    segments = split_text(text)
    assert_well_formed(segments, text)
    assert len(segments) >= 6526 // MAX_SEGMENT_CHARS


def test_split_text_leading_long_sentence_emits_no_empty_segment():
    # The old splitter's failure: a first "sentence" over the limit flushed
    # an empty current segment, which VITS cannot synthesize.
    text = run_on(1107) + ". Hola a todos. Espero que estén bien."
    segments = split_text(text, max_chars=1000)
    assert_well_formed(segments, text, max_chars=1000)
    assert segments[-1].endswith("Espero que estén bien.")


def test_split_text_short_text_is_one_segment():
    assert split_text("Hola a todos.") == ["Hola a todos."]


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n", None])
def test_split_text_empty_or_whitespace_returns_nothing(text):
    assert split_text(text) == []


@pytest.mark.parametrize("text", ["...", " . . ! ", "— ¿? ¡!"])
def test_split_text_punctuation_only_returns_nothing(text):
    assert split_text(text) == []


def test_split_text_collapses_whitespace():
    assert split_text("Hola\n\n  a   todos.\t") == ["Hola a todos."]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Primera frase. Segunda frase", ["Primera frase.", "Segunda frase"]),
        ("¿Cómo entraste aquí? Y él calló", ["¿Cómo entraste aquí?", "Y él calló"]),
        ("¡Qué alegría! Vamos ya", ["¡Qué alegría!", "Vamos ya"]),
        ("primero esto; luego aquello", ["primero esto;", "luego aquello"]),
        ("Dijo: «Venid al banquete.» Y fueron", ["Dijo: «Venid al banquete.»", "Y fueron"]),
        # Run-on transcripts often open a question with no period before it.
        ("y entonces dijo ¿cómo entraste aquí", ["y entonces dijo", "¿cómo entraste aquí"]),
        ("y gritó ¡qué alegría tan grande", ["y gritó", "¡qué alegría tan grande"]),
    ],
)
def test_split_text_spanish_sentence_boundaries(text, expected):
    # A max just under the full length forces exactly one split, which must
    # land on the sentence boundary.
    assert split_text(text, max_chars=len(text) - 1, min_chars=0) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("el Sr. García habló, y luego todos cantaron", ["el Sr. García habló,", "y luego todos cantaron"]),
        ("como dice Jn. 3:16, con mucha claridad", ["como dice Jn. 3:16,", "con mucha claridad"]),
        ("en el capítulo 22. 3 personas vinieron, y cantaron", ["en el capítulo 22. 3 personas vinieron,", "y cantaron"]),
    ],
)
def test_split_text_abbreviation_or_number_is_not_a_sentence_end(text, expected):
    # Each case has one comma. Had the period counted as a sentence end, the
    # split would land there instead.
    assert split_text(text, max_chars=len(text) - 1, min_chars=0) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("primero esto – luego aquello", ["primero esto –", "luego aquello"]),
        ("primero esto - luego aquello", ["primero esto -", "luego aquello"]),
        ("un niño bien-educado y luego aquello", ["un niño bien-educado", "y luego aquello"]),
    ],
)
def test_split_text_dashes_are_clause_boundaries(text, expected):
    assert split_text(text, max_chars=len(text) - 1, min_chars=0) == expected


def test_split_text_prefers_sentence_over_comma_over_word():
    text = "uno dos, tres cuatro. cinco seis, siete ocho"
    # Room for one sentence per segment: break at the period, not the commas.
    assert split_text(text, max_chars=22, min_chars=0) == ["uno dos, tres cuatro.", "cinco seis, siete ocho"]
    # No sentence fits: fall back to the commas.
    assert split_text(text, max_chars=14, min_chars=0) == ["uno dos,", "tres cuatro.", "cinco seis,", "siete ocho"]


def test_split_text_comma_run_on_splits_at_commas():
    clause = "el rey envió a sus siervos a llamar a los invitados"
    text = ", ".join([clause] * 40)
    segments = split_text(text)
    assert_well_formed(segments, text)
    assert all(s.endswith(",") for s in segments[:-1])


def test_split_text_uses_fewest_evenly_sized_segments():
    sentence = "Hola a todos."
    text = " ".join([sentence] * 100)  # 1,399 characters
    segments = split_text(text)
    assert_well_formed(segments, text)
    assert len(segments) == 4
    assert max(map(len, segments)) - min(map(len, segments)) <= len(sentence) + 1


@pytest.mark.parametrize(
    "text",
    [
        run_on(401),
        run_on(805) + ". Amén. " + run_on(410) + ", y dijo. Hola a todos.",
        run_on(1107) + ". Hola a todos. Espero que estén bien.",
        "Hola a todos, hermanos y hermanas. " * 200,
    ],
)
def test_split_text_leaves_no_short_fragments(text):
    segments = split_text(text)
    assert_well_formed(segments, text)
    assert min(map(len, segments)) >= MIN_SEGMENT_CHARS, [len(s) for s in segments]


def test_split_text_short_sentence_between_two_full_ones_is_folded_in():
    long_sentence = run_on(380) + "."
    text = f"{long_sentence} Sí. {long_sentence}"
    max_chars = len(long_sentence) + 2  # neither neighbour has room for "Sí."
    segments = split_text(text, max_chars=max_chars)
    assert_well_formed(segments, text, max_chars=max_chars)
    assert min(map(len, segments)) >= MIN_SEGMENT_CHARS


def test_split_text_short_input_stays_whole():
    assert split_text("Amén.") == ["Amén."]


def test_split_text_never_splits_mid_word():
    text = run_on(6526)
    words = set(text.split())
    for segment in split_text(text, max_chars=97):
        assert set(segment.split()) <= words


def test_split_text_overlong_single_word_is_cut_to_max():
    word = "a" * 950
    segments = split_text(f"Hola {word} adiós", max_chars=400)
    assert all(0 < len(s) <= 400 for s in segments)
    assert "".join(segments).replace(" ", "") == f"Hola{word}adiós"
