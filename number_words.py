import re
import unicodedata

from num2words import num2words

# The Spanish VITS model (tts_models/es/css10/vits) has no digits in its
# character vocabulary, and multilingual_cleaners lowercases text without
# expanding numbers, so every digit is silently discarded: "Génesis capítulo
# 1" is spoken as "génesis capítulo". Sermon transcripts are full of Bible
# references, years and counts, so everything numeric is written out in
# Spanish words here before the text reaches the model.
#
# Rules run from most to least specific, and each one only replaces what it
# matched, so a later, more general rule never sees text an earlier one has
# already read. Whatever none of them recognises is still read as a plain
# cardinal: no ASCII digit survives normalize_numbers().

# Bible book names and the abbreviations that turn up in transcripts, compared
# lowercased and without accents. A number after one of these is a reference
# ("Juan 3.16" is chapter 3, verse 16, not 3.16).
_BOOKS = {
    "genesis", "exodo", "levitico", "numeros", "deuteronomio", "josue",
    "jueces", "rut", "samuel", "reyes", "cronicas", "esdras", "nehemias",
    "ester", "job", "salmo", "salmos", "proverbios", "eclesiastes", "cantares",
    "isaias", "jeremias", "lamentaciones", "ezequiel", "daniel", "oseas",
    "joel", "amos", "abdias", "jonas", "miqueas", "nahum", "habacuc",
    "sofonias", "hageo", "zacarias", "malaquias", "mateo", "marcos", "lucas",
    "juan", "hechos", "romanos", "corintios", "galatas", "efesios",
    "filipenses", "colosenses", "tesalonicenses", "timoteo", "tito",
    "filemon", "hebreos", "santiago", "pedro", "judas", "apocalipsis",
    "gn", "ex", "lv", "dt", "sal", "pr", "is", "jer", "ez", "dn", "mt", "mc",
    "lc", "jn", "hch", "ro", "rom", "co", "cor", "gá", "ga", "gal", "ef",
    "fil", "col", "ts", "tes", "ti", "tim", "heb", "stg", "ap", "apoc",
}

# Books that take a number ("1 Corintios"). Spanish reads the epistles as
# "primera de Corintios" (epístola is feminine) and the history books as
# "primero de Reyes" (libro is masculine). Abbreviations are expanded to the
# full name, since the model would otherwise spell "Co" out as a word.
_NUMBERED_BOOKS = {
    "corintios": ("Corintios", "f"), "co": ("Corintios", "f"), "cor": ("Corintios", "f"),
    "tesalonicenses": ("Tesalonicenses", "f"), "ts": ("Tesalonicenses", "f"),
    "tes": ("Tesalonicenses", "f"),
    "timoteo": ("Timoteo", "f"), "ti": ("Timoteo", "f"), "tim": ("Timoteo", "f"),
    "pedro": ("Pedro", "f"), "pe": ("Pedro", "f"), "ped": ("Pedro", "f"),
    "juan": ("Juan", "f"), "jn": ("Juan", "f"),
    "samuel": ("Samuel", "m"), "sam": ("Samuel", "m"), "sa": ("Samuel", "m"),
    "reyes": ("Reyes", "m"), "re": ("Reyes", "m"), "rey": ("Reyes", "m"),
    "cronicas": ("Crónicas", "m"), "cr": ("Crónicas", "m"), "cro": ("Crónicas", "m"),
    "cron": ("Crónicas", "m"),
}
_LOWERCASE_BOOKS = {"corintios", "tesalonicenses", "reyes", "cronicas"}
_BOOK_ORDINALS = {
    "1": 1, "2": 2, "3": 3, "i": 1, "ii": 2, "iii": 3,
}
_BOOK_ORDINAL_WORDS = {
    (1, "f"): "primera", (2, "f"): "segunda", (3, "f"): "tercera",
    (1, "m"): "primero", (2, "m"): "segundo", (3, "m"): "tercero",
}

# A few common nouns, so "1 vez" reads "una vez" and "21 años" reads "veintiún
# años" rather than "uno vez" and "veintiuno años". Only nouns sermons use a
# lot; anything else gets the plain cardinal, which is still understandable.
_FEMININE_NOUNS = {
    "vez", "veces", "persona", "personas", "semana", "semanas", "hora", "horas",
    "mujer", "mujeres", "cosa", "cosas", "palabra", "palabras", "iglesia",
    "iglesias", "parte", "partes", "tribu", "tribus", "noche", "noches",
    "piedra", "piedras", "oveja", "ovejas", "moneda", "monedas", "ciudad",
    "ciudades", "generación", "generaciones", "carta", "cartas", "virgen",
    "vírgenes", "hija", "hijas", "familia", "familias", "razón", "razones",
    "pregunta", "preguntas", "manera", "maneras", "copa", "copas", "plaga",
    "plagas", "lámpara", "lámparas", "mañana", "mañanas", "tarde", "tardes",
}
_MASCULINE_NOUNS = {
    "año", "años", "día", "días", "hombre", "hombres", "mes", "meses",
    "libro", "libros", "hijo", "hijos", "discípulo", "discípulos", "minuto",
    "minutos", "siglo", "siglos", "segundo", "segundos", "apóstol",
    "apóstoles", "pan", "panes", "pez", "peces", "talento", "talentos",
    "denario", "denarios", "dólar", "dólares", "versículo", "versículos",
    "capítulo", "capítulos", "punto", "puntos", "mandamiento", "mandamientos",
    "sello", "sellos", "ángel", "ángeles", "anciano", "ancianos", "niño",
    "niños", "kilómetro", "kilómetros", "metro", "metros", "siervo", "siervos",
    # Not nouns of their own, but uno shortens before them the same way
    # ("21 mil" is "veintiún mil").
    "mil", "millón", "millones", "billón", "billones",
}

# Words before a bare number that make "1 Juan" a verse or chapter number
# followed by the apostle's name, not the epistle.
_COUNTING_WORDS = {
    "capitulo", "capitulos", "cap", "versiculo", "versiculos", "vs", "v", "vv",
}

_TIME_FOLLOWS = re.compile(
    r"\s*(?:h\b|hrs?\b|horas\b|de la (?:mañana|tarde|noche)\b|[ap]\.?\s?m\b\.?)",
    re.IGNORECASE,
)

# A numeric token: thousands grouped with "." (Spanish) or "," (English, since
# the transcripts are translated from English), with an optional decimal part
# in the other separator; a decimal with one separator; or a plain integer.
_NUMBER = (
    r"(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?(?![.,]?\d)"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![.,]?\d)"
    r"|\d+[.,]\d+(?![.,]?\d)"
    r"|\d+)"
)

# Plain integers with more digits than this (phone or account numbers) are read
# digit by digit; nobody says them as one quantity.
_MAX_CARDINAL_DIGITS = 12


def normalize_numbers(text):
    """
    Return text with every number written out in Spanish words.

    Handles numbered Bible books ("1 Corintios" -> "primera de Corintios"),
    references ("Juan 3:16-18" -> "Juan tres, del dieciséis al dieciocho"),
    clock times after "las" or before "de la mañana", ordinals ("2.ª"),
    percentages, dollar amounts, ranges ("15-18" -> "del quince al
    dieciocho"), grouped thousands ("2.000" and "2,000"), decimals ("3,5"
    -> "tres coma cinco") and plain cardinals. Anything else made of digits
    is read as a cardinal, so the result never contains an ASCII digit.
    """
    if not text:
        return text
    text = _sub(_NUMBERED_BOOK, _numbered_book, text)
    # A sign settles what the number is, so these go before references:
    # "2.5%" is a decimal, not chapter 2, verse 5.
    text = _sub(_PERCENT, lambda m, t: f"{_read_number(m.group(1))} por ciento", text)
    text = _sub(_DOLLARS, _dollars, text)
    text = _sub(_CROSS_CHAPTER_REF, _cross_chapter_ref, text)
    text = _sub(_REFERENCE, _reference, text)
    text = _sub(_ORDINAL, _ordinal, text)
    text = _sub(_RANGE, _range, text)
    text = _sub(_PLAIN_NUMBER, _plain_number, text)
    # Anything a rule above left behind (a stray digit next to other digits,
    # say). The rules should leave nothing, but a dropped digit is silent, so
    # this is what guarantees none reaches the model.
    return _sub(re.compile(r"\d+"), lambda m, t: _read_number(m.group()), text)


def _sub(pattern, replace, text):
    """
    pattern.sub(), with each replacement given the whole text (for context)
    and padded with a space wherever it would otherwise run into a letter
    ("3er" -> "tercer", but "10x" -> "diez x").
    """
    def wrapped(match):
        words = replace(match, text)
        if words == match.group():
            return words
        start, end = match.span()
        if start > 0 and text[start - 1].isalnum():
            words = " " + words
        if end < len(text) and text[end].isalnum():
            words += " "
        return words

    return pattern.sub(wrapped, text)


def _plain(word):
    """Lowercase and drop accents, for comparing against the word lists."""
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _word_before(text, index):
    match = re.search(r"(\w+)\.?\s*$", text[max(0, index - 40):index])
    return _plain(match.group(1)) if match else ""


_NEXT_WORD = re.compile(r"\s*([^\W\d_]+)")


def _word_after(text, index):
    match = _NEXT_WORD.match(text, index)
    return match.group(1).lower() if match else ""


def _cardinal(value):
    words = num2words(value, lang="es")
    # num2words writes "veintiuno mil" for 21000; Spanish shortens uno before
    # mil, millón and millones ("veintiún mil", "un millón").
    return re.sub(
        r"(\w*)uno (?=mil\b|millón\b|millones\b|billón\b|billones\b)",
        lambda m: ("veintiún " if m.group(1) == "veinti" else m.group(1) + "un "),
        words,
    )


def _ordinal(match, text):
    value = int(match.group(1))
    suffix = (match.group(2) or match.group(3)).lower()
    words = num2words(value, lang="es", to="ordinal")
    for wrong, right in _ORDINAL_SPELLINGS.items():
        words = words.replace(wrong, right)
    if suffix in ("ª", "ra", "da", "ta", "va", "ma", "na"):
        # num2words only writes the masculine form; every word of a Spanish
        # ordinal ends in -o and agrees in gender ("vigésima primera").
        words = re.sub(r"o\b", "a", words)
    elif suffix == "er" or _word_after(text, match.end()) not in _NOT_NOUNS:
        # Before a noun, primero and tercero shorten ("1.º lugar" is "primer
        # lugar"; "1.º de enero" stays "primero de enero").
        words = re.sub(r"(primer|tercer)o$", r"\1", words)
    return words


# num2words misspells a few ordinals; these are the RAE forms.
_ORDINAL_SPELLINGS = {
    "décimoprimero": "undécimo",
    "décimosegundo": "duodécimo",
    "quadragésimo": "cuadragésimo",
    "cuadrigentésimo": "cuadringentésimo",
    "septigentésimo": "septingentésimo",
    "octigentésimo": "octingentésimo",
}

# Words that can follow a masculine ordinal without being the noun it
# describes. An ordinal at the end of a sentence has no word after it and
# keeps its full form too.
_NOT_NOUNS = {
    "", "de", "del", "y", "e", "o", "u", "en", "a", "al", "que", "con", "por",
    "para", "es", "fue", "era", "se",
}


def _read_number(token):
    """Spanish words for one numeric token as matched by _NUMBER."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", token):
        whole, _, fraction = token.replace(".", "").partition(",")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
        whole, _, fraction = token.replace(",", "").partition(".")
    elif re.fullmatch(r"\d+[.,]\d+", token):
        whole, fraction = re.split(r"[.,]", token)
    else:
        whole, fraction = token, ""

    if (len(whole) > 1 and whole.startswith("0")) or len(whole) > _MAX_CARDINAL_DIGITS:
        words = _digits(whole)
    else:
        words = _cardinal(int(whole))
    if fraction:
        # "3,05" is "tres coma cero cinco": leading zeros are read one by one,
        # then the rest as a number.
        zeros = len(fraction) - len(fraction.lstrip("0"))
        rest = fraction[zeros:]
        parts = ["cero"] * zeros
        if rest:
            parts.append(_digits(rest) if len(rest) > _MAX_CARDINAL_DIGITS else _cardinal(int(rest)))
        words += " coma " + " ".join(parts)
    return words


def _digits(digits):
    return " ".join(_cardinal(int(d)) for d in digits)


def _agree(words, noun):
    """Make a cardinal agree with the noun after it, for the nouns we know."""
    if noun in _FEMININE_NOUNS:
        words = re.sub(r"ientos\b", "ientas", words)
        words = re.sub(r"uno$", "una", words)
    elif noun in _MASCULINE_NOUNS:
        words = re.sub(r"veintiuno$", "veintiún", words)
        words = re.sub(r"uno$", "un", words)
    return words


# ---- Numbered books: "1 Corintios", "2ª Timoteo", "II Reyes", "1 Co." ----

# Roman numerals are matched case-sensitively even though the pattern is not:
# a lowercase "i" is not a book number.
_BOOK_NUMBER = r"(?:[123]|(?-i:III|II|I))(?:\.?\s?[ºª°]|ra|ro|er|da|do|a)?"
_NUMBERED_BOOK = re.compile(
    # "1 y 2 Samuel", "1, 2 y 3 Juan": the earlier numbers are part of the
    # same name, so they are matched along with it.
    rf"(?<![\w:.,])(?P<list>(?:{_BOOK_NUMBER}\s*,\s*)*{_BOOK_NUMBER}\s+[yoe]\s+)?"
    r"(?P<n>[123]|(?-i:III|II|I))(?:\.?\s?[ºª°]|ra|ro|er|da|do|a)?\s+"
    # The keys are unaccented, so the accented spellings are added here.
    r"(?P<book>"
    + "|".join(sorted({*_NUMBERED_BOOKS, "crónicas", "crón"}, key=len, reverse=True))
    + r")\b\.?",
    re.IGNORECASE,
)


def _numbered_book(match, text):
    # "versículo 1 Juan dice" is verse one, then the apostle.
    if _word_before(text, match.start()) in _COUNTING_WORDS:
        return match.group()
    number, name = match.group("n"), match.group("book")
    if number.isalpha() and not number.isupper():
        return match.group()
    # Lowercase "2 ti" or "1 juan" is a pronoun or a person, not a book. Only
    # the names that mean nothing else are trusted in lowercase.
    if name[0].islower() and _plain(name) not in _LOWERCASE_BOOKS:
        return match.group()
    entry = _NUMBERED_BOOKS.get(_plain(name))
    earlier = re.findall(r"[123]|III|II|I", match.group("list") or "")
    positions = [_BOOK_ORDINALS.get(n.lower()) for n in [*earlier, number]]
    if entry is None or None in positions:
        # IGNORECASE also matches "ſ" for "s" and "İ" for "I", which the
        # lookups do not know.
        return match.group()
    book, gender = entry
    ordinals = [_BOOK_ORDINAL_WORDS[(p, gender)] for p in positions]
    if len(ordinals) > 1:
        # The pattern ignores case, so "1 Y 2 Samuel" gets here with a capital.
        conjunction = re.search(r"\s([yoe])\s+$", match.group("list"), re.IGNORECASE).group(1).lower()
        # "e" is "y" before an i sound ("I e II"); spoken as ordinals, the
        # next word starts with p, s or t, so it is "y" again.
        conjunction = "o" if conjunction == "o" else "y"
        ordinals = [", ".join(ordinals[:-1]) + f" {conjunction} " + ordinals[-1]]
    # The period after an abbreviation is dropped along with it: "1 Co. 13"
    # is not the end of a sentence. After the full name ("en 1 Juan. Y")
    # the period ends the sentence and stays.
    period = "." if match.group().endswith(".") and _plain(name) == _plain(book) else ""
    return f"{ordinals[0]} de {book}{period}"


# ---- References and clock times: "3:16", "3:16-18", "3:16,18", "3:16-4:2" ----
#
# A period works as the separator too, with or without a book in front. In
# the stored transcripts "N.N" with one or two digits after the period is
# almost always a reference ("en 16.7, dice", "Proverbios 4.23") or a time
# ("las 6.30 de la mañana"); decimals are written with a comma ("2,5%").

_CROSS_CHAPTER_REF = re.compile(
    r"(?<![\d.,:])(\d{1,3}):(\d{1,3})\s?[-–]\s?(\d{1,3}):(\d{1,3})(?!\d|:\d)"
)

_VERSE = r"\d{1,3}(?:\s?[-–]\s?\d{1,3})?"
_REFERENCE = re.compile(
    rf"(?<![\d.,:])(\d{{1,3}})([:.])({_VERSE}(?:,{_VERSE})*)(?!\d|[:.,]\d)"
)


def _cross_chapter_ref(match, text):
    first, verse, last, last_verse = (_cardinal(int(g)) for g in match.groups())
    return f"del {first}, {verse} al {last}, {last_verse}"


def _reference(match, text):
    chapter, separator, verses = match.groups()
    after_book = _word_before(text, match.start()) in _BOOKS
    if separator == "." and not after_book and re.match(r"\d{3}(?!\d)", verses):
        # "2.000" and "1.234,56" are grouped thousands; leave them for
        # _PLAIN_NUMBER. After a book
        # name ("Salmos 119.105") three digits are still a verse.
        return match.group()
    if not after_book and re.fullmatch(r"\d{2}", verses):
        hour, minute = int(chapter), int(verses)
        before = _word_before(text, match.start())
        if hour <= 24 and minute < 60 and (
            before in ("la", "las") or _TIME_FOLLOWS.match(text, match.end())
        ):
            return _clock(hour, minute)
    # A comma list is only read as verses in the tight form "3:16,18"; with
    # a space after the comma it could just as well be the sentence going on
    # with another number, so _REFERENCE stops before it.
    items = [_verse_span(v) for v in verses.split(",")]
    if len(items) > 1:
        verses = ", ".join(items[:-1]) + " y " + items[-1]
    else:
        verses = items[0]
    return f"{_cardinal(int(chapter))}, {verses}"


def _verse_span(verses):
    numbers = re.split(r"\s?[-–]\s?", verses)
    if len(numbers) == 2:
        return f"del {_cardinal(int(numbers[0]))} al {_cardinal(int(numbers[1]))}"
    return _cardinal(int(numbers[0]))


def _clock(hour, minute):
    words = "una" if hour == 1 else _cardinal(hour)
    if minute == 0:
        return f"{words} en punto"
    if minute == 15:
        return f"{words} y cuarto"
    if minute == 30:
        return f"{words} y media"
    return f"{words} y {_cardinal(minute)}"


# ---- Ordinals: "1.º", "2.ª", "3º", "1er", "2do", "3ra" ----

_ORDINAL = re.compile(
    r"(?<![\d.,])(\d{1,3})(?:\.?([ºª°])(?!\s?[CF]\b)|(er|ra|ro|da|do|ta|to|va|vo|ma|mo|na|no)\b)"
)


# ---- Percentages, dollar amounts and ranges ----

_PERCENT = re.compile(rf"(?<![\d.,])({_NUMBER})\s?%")
_DOLLARS = re.compile(
    rf"\$\s?({_NUMBER})(?![\d])(\s+(?:mil millones|millones|millón|mil)\b)?"
)


def _dollars(match, text):
    words = _read_number(match.group(1))
    if match.group(2):
        # "$2 millones" is "dos millones de dólares", with the currency after
        # the whole amount.
        scale = match.group(2).strip()
        # Mil, millón and millones are all masculine, like dólares.
        words = _agree(words, "dólares")
        de = " de" if scale != "mil" else ""
        return f"{words} {scale}{de} dólares"
    if words == "uno":
        return "un dólar"
    return f"{_agree(words, 'dólares')} dólares"


_RANGE = re.compile(r"(?<![\d.,:])(\d{1,4})(?:-|\s?–\s?)(\d{1,4})(?![\d]|[.,:]\d)")


def _range(match, text):
    first, last = (_cardinal(int(g)) for g in match.groups())
    before = _word_before(text, match.start())
    # Keep the sentence grammatical when it already has the preposition:
    # "de 15-18 años", "entre 1995-2000".
    if before == "de":
        return f"{first} a {last}"
    if before == "del":
        return f"{first} al {last}"
    if before == "entre":
        return f"{first} y {last}"
    return f"del {first} al {last}"


# ---- Everything else that is a number ----

_PLAIN_NUMBER = re.compile(rf"(?<![\d.,])({_NUMBER})")


def _plain_number(match, text):
    words = _read_number(match.group(1))
    if "coma" in words:
        return words
    return _agree(words, _word_after(text, match.end()))
