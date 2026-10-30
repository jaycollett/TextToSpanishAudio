import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from number_words import normalize_numbers


@pytest.mark.parametrize(
    "text, expected",
    [
        # The production failures: each number here was being dropped.
        ("en génesis capítulo 1.", "en génesis capítulo uno."),
        ("abrir en deuteronomio 18", "abrir en deuteronomio dieciocho"),
        ("en el 18, empezando en el 15.", "en el dieciocho, empezando en el quince."),
        ("tenía 7 hijos", "tenía siete hijos"),
        ("en 1995 nació", "en mil novecientos noventa y cinco nació"),
        ("en el año 2024", "en el año dos mil veinticuatro"),
        ("cerca de 100 ovejas", "cerca de cien ovejas"),
        ("un millón: 1000000", "un millón: un millón"),
    ],
)
def test_normalize_numbers_plain_cardinals(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Juan 3:16", "Juan tres, dieciséis"),
        ("Génesis 1:1", "Génesis uno, uno"),
        ("Éxodo 20:3", "Éxodo veinte, tres"),
        ("Salmo 23:1", "Salmo veintitrés, uno"),
        ("Salmos 119:105", "Salmos ciento diecinueve, ciento cinco"),
        ("Isaías 53:5", "Isaías cincuenta y tres, cinco"),
        ("Romanos 8:28", "Romanos ocho, veintiocho"),
        ("Apocalipsis 21:4", "Apocalipsis veintiuno, cuatro"),
        ("Jn. 3:16", "Jn. tres, dieciséis"),
        # A period works as the separator too, with or without a book.
        ("Juan 3.16", "Juan tres, dieciséis"),
        ("Mt. 5.9", "Mt. cinco, nueve"),
        ("Salmos 119.105", "Salmos ciento diecinueve, ciento cinco"),
        ("Así que en 16.7, dice", "Así que en dieciséis, siete, dice"),
        # A colon after the reference introduces the quotation.
        ("Oseas 2:14: Por tanto", "Oseas dos, catorce: Por tanto"),
        # Ranges and lists of verses.
        ("Juan 3:16-18", "Juan tres, del dieciséis al dieciocho"),
        ("Juan 3:16–18", "Juan tres, del dieciséis al dieciocho"),
        ("Juan 3:16,18", "Juan tres, dieciséis y dieciocho"),
        ("Hechos 2:1-4,38", "Hechos dos, del uno al cuatro y treinta y ocho"),
        ("Mateo 5:3-7:29", "Mateo del cinco, tres al siete, veintinueve"),
        # With a space after the comma it is the sentence going on.
        ("Juan 3:16, 18 personas", "Juan tres, dieciséis, dieciocho personas"),
        # No book name, but no other sensible reading either.
        ("en el versículo 3:16", "en el versículo tres, dieciséis"),
    ],
)
def test_normalize_numbers_bible_references(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1 Corintios 13", "primera de Corintios trece"),
        ("2 Corintios 5:17", "segunda de Corintios cinco, diecisiete"),
        ("1 Tesalonicenses 5:16", "primera de Tesalonicenses cinco, dieciséis"),
        ("2 Timoteo 3:16", "segunda de Timoteo tres, dieciséis"),
        ("1 Pedro 5:7", "primera de Pedro cinco, siete"),
        ("1 Juan 4:8", "primera de Juan cuatro, ocho"),
        ("3 Juan 1:2", "tercera de Juan uno, dos"),
        ("1 Samuel 17", "primero de Samuel diecisiete"),
        ("2 Reyes 5", "segundo de Reyes cinco"),
        ("2 Crónicas 7:14", "segundo de Crónicas siete, catorce"),
        ("1ª Corintios", "primera de Corintios"),
        ("1.ª Corintios", "primera de Corintios"),
        ("I Corintios 13", "primera de Corintios trece"),
        ("II Reyes 2", "segundo de Reyes dos"),
        ("III Juan", "tercera de Juan"),
        ("1 Co. 13:4 dice", "primera de Corintios trece, cuatro dice"),
        ("2 Tim. 1:7", "segunda de Timoteo uno, siete"),
        ("leemos en 1 corintios", "leemos en primera de Corintios"),
    ],
)
def test_normalize_numbers_numbered_books(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # A verse number followed by the apostle's name, not the epistle.
        ("versículo 1 Juan dice", "versículo uno Juan dice"),
        ("Juan 3:1 Juan dice", "Juan tres, uno Juan dice"),
        # Lowercase names and pronouns are not book abbreviations.
        ("eran 2 ti", "eran dos ti"),
        ("vinieron 2 juan y pedro", "vinieron dos juan y pedro"),
    ],
)
def test_normalize_numbers_not_numbered_books(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("había 2.000 personas", "había dos mil personas"),
        ("había 2,000 personas", "había dos mil personas"),
        ("unos 12.500.000 habitantes", "unos doce millones quinientos mil habitantes"),
        ("unos 21,000 hombres", "unos veintiún mil hombres"),
        ("3,5 kilos", "tres coma cinco kilos"),
        # In the transcripts a period before one or two digits is a
        # reference, so only the comma is a decimal there.
        ("3.5 kilos", "tres, cinco kilos"),
        ("3.14159", "tres coma catorce mil ciento cincuenta y nueve"),
        ("3,05 metros", "tres coma cero cinco metros"),
        ("1.234,56", "mil doscientos treinta y cuatro coma cincuenta y seis"),
        ("1,234.56", "mil doscientos treinta y cuatro coma cincuenta y seis"),
        # A sentence that ends on a number keeps its period.
        ("nació en 1995. Luego", "nació en mil novecientos noventa y cinco. Luego"),
    ],
)
def test_normalize_numbers_thousands_and_decimals(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("los versículos 15-18", "los versículos del quince al dieciocho"),
        ("los versículos 15–18", "los versículos del quince al dieciocho"),
        ("de 15-18 años", "de quince a dieciocho años"),
        ("entre 1995-2000", "entre mil novecientos noventa y cinco y dos mil"),
        ("desde el 1939 – 1945", "desde el del mil novecientos treinta y nueve al mil novecientos cuarenta y cinco"),
    ],
)
def test_normalize_numbers_ranges(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("el 1.º lugar", "el primer lugar"),
        ("el 1º de enero", "el primero de enero"),
        ("quedó 2.º", "quedó segundo"),
        ("la 2.ª carta", "la segunda carta"),
        ("la 3ª vez", "la tercera vez"),
        ("el 3er día", "el tercer día"),
        ("el 2do día", "el segundo día"),
        ("la 1ra vez", "la primera vez"),
        ("el 21.º siglo", "el vigésimo primer siglo"),
        ("la 21.ª", "la vigésima primera"),
    ],
)
def test_normalize_numbers_ordinals(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("el 50% de la gente", "el cincuenta por ciento de la gente"),
        ("un 100 % seguro", "un cien por ciento seguro"),
        ("el 2,5%", "el dos coma cinco por ciento"),
        ("cuesta $100", "cuesta cien dólares"),
        ("cuesta $1", "cuesta un dólar"),
        ("cuesta $21", "cuesta veintiún dólares"),
        ("cuesta $2,000", "cuesta dos mil dólares"),
        ("unos $2 millones.", "unos dos millones de dólares."),
        ("$1 millón", "un millón de dólares"),
        ("$21 millones", "veintiún millones de dólares"),
        ("$21 mil", "veintiún mil dólares"),
        ("$5 mil millones", "cinco mil millones de dólares"),
    ],
)
def test_normalize_numbers_percent_and_dollars(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a las 10:30", "a las diez y media"),
        ("a la 1:15", "a la una y cuarto"),
        ("las 7:00", "las siete en punto"),
        ("10:45 de la mañana", "diez y cuarenta y cinco de la mañana"),
        ("a las 6:30 p.m.", "a las seis y media p.m."),
        ("a las 6.30 de la mañana", "a las seis y media de la mañana"),
        # After a book it is a reference, whatever it looks like.
        ("Juan 10:30", "Juan diez, treinta"),
    ],
)
def test_normalize_numbers_clock_times(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1 vez", "una vez"),
        ("21 personas", "veintiuna personas"),
        ("500 personas", "quinientas personas"),
        ("21000 personas", "veintiún mil personas"),
        ("1 hombre", "un hombre"),
        ("21 años", "veintiún años"),
        ("31 días", "treinta y un días"),
        # An unknown noun gets the plain cardinal.
        ("21 camellos", "veintiuno camellos"),
    ],
)
def test_normalize_numbers_gender_agreement(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5551234567890", "cinco cinco cinco uno dos tres cuatro cinco seis siete ocho nueve cero"),
        ("el agente 007", "el agente cero cero siete"),
        ("COVID-19", "COVID-diecinueve"),
        ("1,2,3", "uno,dos,tres"),
        ("10x", "diez x"),
        ("ver 3:16:20", "ver tres:dieciséis:veinte"),
    ],
)
def test_normalize_numbers_odd_input_still_read(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize("text", ["", None, "Hola a todos.", "¿Cómo están? ¡Amén!"])
def test_normalize_numbers_text_without_digits_unchanged(text):
    assert normalize_numbers(text) == text


# Sentences from the stored Spanish transcripts, covering the shapes that
# turn up most there.
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hola a todos, hoy vamos a ver Mateo 21, versículos 28 al 32.",
         "Hola a todos, hoy vamos a ver Mateo veintiuno, versículos veintiocho al treinta y dos."),
        ("Juan 13:1-21. Antes de la fiesta de la Pascua",
         "Juan trece, del uno al veintiuno. Antes de la fiesta de la Pascua"),
        ("En 1 Juan 1:8-9, dice que si decimos que no tenemos pecado",
         "En primera de Juan uno, del ocho al nueve, dice que si decimos que no tenemos pecado"),
        ("Romanos 5.1-5 dice, Por tanto, habiendo sido justificados por la fe",
         "Romanos cinco, del uno al cinco dice, Por tanto, habiendo sido justificados por la fe"),
        ("Tal vez eran 120 los que quedaron. Eso es como el 2,5% de la multitud.",
         "Tal vez eran ciento veinte los que quedaron. Eso es como el dos coma cinco por ciento de la multitud."),
        ("y estimo que vale lo que sea, $5,000 o $6,000.",
         "y estimo que vale lo que sea, cinco mil dólares o seis mil dólares."),
        ("Los cristianos han enseñado durante 2.000 años",
         "Los cristianos han enseñado durante dos mil años"),
        ("Así que estamos hablando más o menos de $2 millones.",
         "Así que estamos hablando más o menos de dos millones de dólares."),
        ("En los años 80, había unos 500 cristianos en Irán. Hoy hay alrededor de 100,000.",
         "En los años ochenta, había unos quinientos cristianos en Irán. Hoy hay alrededor de cien mil."),
        ("que es Mateo 20, 17-28. Y resucitará al tercer día.",
         "que es Mateo veinte, del diecisiete al veintiocho. Y resucitará al tercer día."),
        ("En 1984, Ronald Reagan obtuvo 525 votos electorales. Ganó en 49 estados y el 58,8% del voto popular.",
         ("En mil novecientos ochenta y cuatro, Ronald Reagan obtuvo quinientos veinticinco votos electorales. "
          "Ganó en cuarenta y nueve estados y el cincuenta y ocho coma ocho por ciento del voto popular.")),
        ("El Salmo 138.6 dice: Porque el Señor es excelso",
         "El Salmo ciento treinta y ocho, seis dice: Porque el Señor es excelso"),
        ("llega a un punto crítico en 10.37-38, donde Jesús dice",
         "llega a un punto crítico en diez, del treinta y siete al treinta y ocho, donde Jesús dice"),
        ("eran aproximadamente $25.", "eran aproximadamente veinticinco dólares."),
        ("Sí, lo que dice Daniel 7:13-14 habla de mí.",
         "Sí, lo que dice Daniel siete, del trece al catorce habla de mí."),
        ("En Éxodo 22, 2-3, si un ladrón es sorprendido.",
         "En Éxodo veintidós, del dos al tres, si un ladrón es sorprendido."),
        ("está cerca» (Mateo 4:17). Y la implicación",
         "está cerca» (Mateo cuatro, diecisiete). Y la implicación"),
        ("nuestro 30º presidente", "nuestro trigésimo presidente"),
        ("¿10 para las 8? Bien, perfecto.", "¿diez para las ocho? Bien, perfecto."),
    ],
)
def test_normalize_numbers_stored_transcript_sentences(text, expected):
    assert normalize_numbers(text) == expected


SAMPLE_SERMON = """
Buenos días, iglesia. Abran sus Biblias en Génesis capítulo 1. Hoy vamos a
leer Juan 3:16-18, Romanos 8:28 y 1 Corintios 13:4-7. En 1995 empecé a
predicar, y desde entonces he visto a más de 2.000 personas, unas 1,500
familias, venir a Cristo. El 50% de ellas tenía entre 15-18 años. Pablo
escribió 2 Timoteo 3:16 alrededor del año 67 d.C., y II Reyes 5 cuenta
la historia de Naamán, que se lavó 7 veces en el Jordán. La 1.ª carta de
Pedro, 1 Pe. 5:7, nos dice que echemos toda ansiedad sobre Él. Nos vemos
el domingo a las 10:30 de la mañana; la ofrenda de $250 fue para 3 familias.
Deuteronomio 18. 3 personas preguntaron por el capítulo 22. Mateo 5:3-7:29
es el Sermón del Monte, y el Salmo 119.105 dice: lámpara es a mis pies tu
palabra. Lean Hechos 2:1-4,38 y 1 Juan 1:9 esta semana. ¡Amén! 1º, 2º, 3º.
"""


def test_normalize_numbers_sample_sermon_leaves_no_ascii_digit():
    assert not re.search(r"[0-9]", normalize_numbers(SAMPLE_SERMON))


def test_normalize_numbers_every_integer_leaves_no_ascii_digit():
    # Every number from 0 to 2100 in each position a sermon might put it.
    for n in range(2101):
        for text in (f"{n}", f"el {n}.", f"Juan {n}:{n % 50}", f"{n}%", f"{n}-{n + 1}", f"{n}º"):
            assert not re.search(r"[0-9]", normalize_numbers(text)), text


@pytest.mark.parametrize(
    "text",
    [
        "1.2.3.4", "1,2.3,4", "12:34:56", "1:2-3:4-5", "$", "%5", "5%%", "1-2-3",
        "1–", "–1", "0.0", "00:00", "9" * 40, "1.000.000.000.000.000.000.000.000.000",
        "3:16,", "3:16-", "1ª", "I 1 II 2", "1er2do", "3,", ",3", ".5", "5.",
    ],
)
def test_normalize_numbers_awkward_input_leaves_no_ascii_digit(text):
    assert not re.search(r"[0-9]", normalize_numbers(text))


@pytest.mark.parametrize(
    "text, expected",
    [
        # The full name keeps a sentence-ending period; an abbreviation's
        # period goes with the abbreviation.
        ("en 1 Juan. Y luego", "en primera de Juan. Y luego"),
        ("en 2 Crónicas. Y luego", "en segundo de Crónicas. Y luego"),
        ("1 Co. 13 dice", "primera de Corintios trece dice"),
        # Every number in a list of books takes the book's gender.
        ("1 y 2 Samuel", "primero y segundo de Samuel"),
        ("1 y 2 Timoteo", "primera y segunda de Timoteo"),
        ("1, 2 y 3 Juan", "primera, segunda y tercera de Juan"),
        ("1 o 2 Corintios", "primera o segunda de Corintios"),
        ("I y II Reyes.", "primero y segundo de Reyes."),
        # The list pattern ignores case.
        ("1 Y 2 Samuel", "primero y segundo de Samuel"),
        ("1, 2 O 3 Juan", "primera, segunda o tercera de Juan"),
        ("I e II Reyes", "primero y segundo de Reyes"),
        # A lowercase "i" is not a Roman numeral.
        ("i y 2 Samuel", "i y segundo de Samuel"),
        ("los versículos 1 y 2 Juan dice", "los versículos uno y dos Juan dice"),
    ],
)
def test_normalize_numbers_numbered_book_periods_and_lists(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize(
    "text",
    # Under IGNORECASE these match the book pattern ("ſ" as "s", "ı" and "İ"
    # as "i") but are not in the lookup tables.
    ["1 Tesalonicenſes", "2 Cronicaſ", "1 Tımoteo", "İ Corintios", "İI Reyes"],
)
def test_normalize_numbers_odd_case_letters_do_not_raise(text):
    assert not re.search(r"[0-9]", normalize_numbers(text))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("el 2.5%", "el dos coma cinco por ciento"),
        ("cuesta $1.50", "cuesta uno coma cincuenta dólares"),
        ("unos $1.5 millones", "unos uno coma cinco millones de dólares"),
        ("unos 21 mil hombres", "unos veintiún mil hombres"),
        ("1 millón", "un millón"),
    ],
)
def test_normalize_numbers_signs_before_references_and_apocope(text, expected):
    out = normalize_numbers(text)
    assert out == expected
    assert "$" not in out and "%" not in out


@pytest.mark.parametrize(
    "text, expected",
    [
        ("el 11.º", "el undécimo"),
        ("la 12.ª", "la duodécima"),
        ("el 40.º", "el cuadragésimo"),
        ("la 45.ª", "la cuadragésima quinta"),
        ("el 400.º", "el cuadringentésimo"),
        ("el 700.º", "el septingentésimo"),
        ("el 800.º", "el octingentésimo"),
    ],
)
def test_normalize_numbers_ordinal_spellings(text, expected):
    assert normalize_numbers(text) == expected


_TOKENS = st.sampled_from(
    list("0123456789.,:;-–—$%ºª°/ \n\tIVXaeiouyACJTSRPſıİ¹½٣３")
    + [" Y ", " O ", " e ", "i ", "Juan ", "Samuel ", "Corintios ", "Co. ", "Reyes ", "Crónicas", "Tesalonicenſes", "las ", "la ",
       "de la mañana", "p.m.", "millones", "mil ", " personas", " años", "er", "da", " y ", " o "]
)


@settings(max_examples=2000, deadline=None, suppress_health_check=list(HealthCheck))
@given(st.one_of(st.lists(_TOKENS, max_size=30).map("".join), st.text(max_size=60)))
def test_normalize_numbers_never_raises_or_leaves_a_digit(text):
    assert not re.search(r"[0-9]", normalize_numbers(text))
