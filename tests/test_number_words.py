import re

import pytest

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
        # A period between chapter and verse is only a reference after a book.
        ("Juan 3.16", "Juan tres, dieciséis"),
        ("Mt. 5.9", "Mt. cinco, nueve"),
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
        ("3.5 kilos", "tres coma cinco kilos"),
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
