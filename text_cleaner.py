"""Shared deterministic text cleaner used throughout the project."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

NUMBER_KEYWORDS = (
    "درجه",
    "سطح",
    "رتبه",
    "رده",
    "پایه",
    "کلاس",
    "گروه",
    "مرحله",
    "فاز",
)

_URL_RE = re.compile(r"(?:https?|ftp)://\S+|www\.\S+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[0-9]+")
_PROTECTED_NUMBER_RE = re.compile(
    rf"({'|'.join(map(re.escape, NUMBER_KEYWORDS))})(?:[\W_]*)([0-9]+)"
)
_DIGIT_MARKERS = {str(number): chr(0xE000 + number) for number in range(10)}
_MARKER_DIGITS = {marker: number for number, marker in _DIGIT_MARKERS.items()}

_CHAR_TRANSLATION = str.maketrans(
    {
        # Arabic/Urdu variants -> Persian.
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ۃ": "ه",
        "ھ": "ه",
        "ؤ": "و",
        "ۇ": "و",
        "ۈ": "و",
        "آ": "ا",  # alef with madda
        "ﺁ": "ا",
        "ﺂ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ٲ": "ا",
        "ٳ": "ا",
        "ٹ": "ت",
        "ں": "ن",
        "ڪ": "ک",
        "ې": "ی",
        "ۍ": "ی",
        "ێ": "ی",
        "ء": "",
        # Kashida counts as a letter in Unicode, so drop it explicitly.
        "\u0640": "",
        # Persian and Arabic digits -> ASCII.
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


def _collapse_repeated_letters(text: str) -> str:
    """Collapse runs of three or more identical letters to one letter."""
    output: list[str] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and text[end] == text[index]:
            end += 1
        run_length = end - index
        if text[index].isalpha() and run_length >= 3:
            output.append(text[index])
        else:
            output.append(text[index:end])
        index = end
    return "".join(output)


_ZWNJ = "\u200c"  # نیم‌فاصله
_MULTI_SPACE_RE = re.compile(r" {2,}")


def normalize_spacing(value: Any) -> str:
    """
    Keep the readable text. Only turn نیم‌فاصله into a normal space and
    collapse runs of more than one space into a single space.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    text = str(value)
    if not text or text.strip().lower() in {"nan", "none", "nat"}:
        return ""

    text = text.replace(_ZWNJ, " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def clean_text(value: Any) -> str:
    """
    Return a compact comparison form of Persian/English text.

    The result contains letters plus only those ASCII digits that immediately
    follow one of NUMBER_KEYWORDS. All spacing, punctuation, symbols, URLs,
    diacritics and other numbers are removed. English letters are lowercase.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return ""

    # NFKC folds Arabic presentation forms and compatibility characters first.
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub("", text)
    text = text.translate(_CHAR_TRANSLATION).lower()

    # Remove every combining mark (Arabic/Persian harakat and Latin accents).
    text = "".join(char for char in text if unicodedata.category(char)[0] != "M")

    # Temporarily protect numbers with an allowed keyword immediately before.
    def protect_number(match: re.Match[str]) -> str:
        marked_digits = "".join(_DIGIT_MARKERS[digit] for digit in match.group(2))
        return match.group(1) + marked_digits

    text = _PROTECTED_NUMBER_RE.sub(protect_number, text)
    text = _NUMBER_RE.sub("", text)

    # Keep letters and protected digit markers only. This removes all spaces,
    # half-spaces, punctuation, symbols, emoji and invisible controls.
    text = "".join(
        char for char in text if char.isalpha() or char in _MARKER_DIGITS
    )
    text = _collapse_repeated_letters(text)
    return "".join(_MARKER_DIGITS.get(char, char) for char in text)


_READABLE_PROTECTED_NUMBER_RE = re.compile(
    rf"({'|'.join(map(re.escape, NUMBER_KEYWORDS))})([\W_]*)([0-9]+)"
)


def clean_text_readable(value: Any) -> str:
    """
    Same cleaning as clean_text, but the result stays readable.

    نیم‌فاصله becomes a normal space, and runs of spaces collapse to one.
    Existing spaces are kept. Punctuation, symbols, URLs, diacritics and
    numbers outside NUMBER_KEYWORDS are still removed.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_ZWNJ, " ")
    text = _URL_RE.sub(" ", text)
    text = text.translate(_CHAR_TRANSLATION).lower()
    text = "".join(char for char in text if unicodedata.category(char)[0] != "M")

    def protect_number(match: re.Match[str]) -> str:
        keyword, between, digits = match.group(1), match.group(2), match.group(3)
        marked_digits = "".join(_DIGIT_MARKERS[digit] for digit in digits)
        separator = " " if any(char.isspace() for char in between) else ""
        return keyword + separator + marked_digits

    text = _READABLE_PROTECTED_NUMBER_RE.sub(protect_number, text)
    text = _NUMBER_RE.sub("", text)
    kept: list[str] = []
    for char in text:
        if char.isalpha() or char in _MARKER_DIGITS:
            kept.append(char)
        elif char.isspace():
            kept.append(" ")
    text = "".join(kept)
    text = _collapse_repeated_letters(text)
    text = "".join(_MARKER_DIGITS.get(char, char) for char in text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()
