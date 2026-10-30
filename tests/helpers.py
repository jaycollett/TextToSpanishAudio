WORDS = [
    "y",
    "entonces",
    "el",
    "pueblo",
    "de",
    "Dios",
    "se",
    "resistió",
    "continuamente",
    "a",
    "la",
    "invitación",
    "del",
    "rey",
    "porque",
    "no",
    "quería",
    "renunciar",
    "a",
    "sus",
    "negocios",
    "ni",
    "a",
    "sus",
    "granjas",
]


def run_on(length):
    """Spanish words with no punctuation at all, at least `length` characters."""
    words = []
    while len(" ".join(words)) < length:
        words.append(WORDS[len(words) % len(WORDS)])
    return " ".join(words)
