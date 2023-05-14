"""Module for string handliing utilities
"""
import re
from difflib import (
    SequenceMatcher
)


def remove_special_characters_from_text(text) -> str:
    """Remove special characters (non words nor space from text using regexp r'[^\w\s]'.
    re module is unicode aware and can handle non english

    TODO: Move to common library

    Args:
        text: text to remove the special characters from
    Returns: test with special characters being removed
    """
    return re.sub(r'[^\w\s]', '', text.strip())


def string_similarity_score(a: str, b: str):
    """Calculate similarity score between two strings
    https://docs.python.org/3/library/difflib.html
    https://stackoverflow.com/a/31236578/4281353

    Args:
        a: text to compare
        b: text to compare
    Returns: similarity score between 0 and 1
    """
    return SequenceMatcher(None, a, b).ratio()


