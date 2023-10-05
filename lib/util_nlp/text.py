"""
Module for NLP utilities
"""
import re
import string

import nltk
nltk.download('words')
nltk.download('wordnet')

from nltk.stem import (
    WordNetLemmatizer
)
from nltk.corpus import (
    wordnet,
    words
)
import textacy.preprocessing


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
SPACE: str = " "
NLTK_ENGLISH_WORDS = set(words.words())
RE_NOISE_CHARACTERS = re.compile(r'[&#<>{}\[\]\\]')


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def redact_special_characters(text: str, replacement: str = '') -> str:
    """
    Remove special characters (non words nor space from text using regexp.
    re module is unicode aware and can handle non english

    TODO: Move to common library

    Args:
        text: text to remove the special characters from
        replacement: string to replace with
    Returns: test with special characters being removed
    """
    return re.sub(r'[^\w\s]', replacement, text.strip())


def decontracted(text: str) -> str:
    """Restore the contracted words"""
    # specific
    text = re.sub(r"won\'t", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"can\'t", "can not", text, flags=re.IGNORECASE)
    # general
    text = re.sub(r"n\'t", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"\'re", " are", text, flags=re.IGNORECASE)
    text = re.sub(r"\'s", " is", text, flags=re.IGNORECASE)
    text = re.sub(r"\'d", " would", text, flags=re.IGNORECASE)
    text = re.sub(r"\'ll", " will", text, flags=re.IGNORECASE)
    text = re.sub(r"\'t", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"\'ve", " have", text, flags=re.IGNORECASE)
    text = re.sub(r"\'m", " am", text, flags=re.IGNORECASE)
    return text


def redact_email_addresses(text: str, replacement: str = "<EMAIL>") -> str:
    """Redact email address with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the email
    Return: redacted text
    """
    # return re.sub(
    #     r"[\w.+-]+@\w+.[a-zA-Z]{2,3}",
    #     replacement,
    #     text
    # )
    return textacy.preprocessing.replace.emails(text=text, repl=replacement)


def redact_urls(text: str, replacement: str = "<URL>") -> str:
    """Redact URLs with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the URL
    Return: redacted text
    """
    # return re.sub(
    #     r"[\w.+-]+@\w+.[a-zA-Z]{2,3}",
    #     replacement,
    #     text
    # )
    return textacy.preprocessing.replace.urls(text=text, repl=replacement)


def redact_phone_numbers(text: str, replacement: str = "<PHONE_NUMBER>") -> str:
    """Redact phone with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the phone number
    Return: redacted text
    """
    # return re.sub(
    #     r"[\w.+-]+@\w+.[a-zA-Z]{2,3}",
    #     replacement,
    #     text
    # )
    return textacy.preprocessing.replace.phone_numbers(text=text, repl=replacement)


def redact_emojis(text: str, replacement: str = "") -> str:
    """Redact emoji with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the emoji
    Return: redacted text
    """
    # return re.sub(
    #     r"[\w.+-]+@\w+.[a-zA-Z]{2,3}",
    #     replacement,
    #     text
    # )
    return textacy.preprocessing.replace.emojis(text=text, repl=replacement)


def redact_noise(
        text: str,
        regexp=rf'([{string.punctuation}]){{2,}}.*',
        replacement="UNK"
) -> str:
    """Redact noise characters defined by regexp with the replacement
    Args:
        text: text to run the redaction
        regexp: pattern to identify noise character sequence
        replacement: replacement for the noise
    Return: redacted text
    """
    return re.sub(
        pattern=regexp,
        repl=replacement,
        string=text
    )


def is_english_word(lemma: str) -> bool:
    """Check if the word is English
    Args:
        lemma: legitimatized word
    Returns: bool
    """
    _word: str = lemma.strip().lower()
    return (
            _word in NLTK_ENGLISH_WORDS or
            len(wordnet.synsets(_word)) > 0
    )


def redact_non_english_words(text: str, replacement="<UNK>") -> str:
    """Redact non English words with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the non English characters
    Return: redacted text
    """
    return SPACE.join([
        _word for _word in text.split() if is_english_word(_word)
    ])


def noise_character_ratio_in_text(text: str, min_length: int = 10) -> float:
    """percentage of noise characters in the text
    Args:
        text: string to check
        min_length: length of the text below which return 0.0
    """
    assert text is not None and min_length > 0
    text = ''.join(text.split())
    length: int = len(text)
    if length >= min_length:
        return len(RE_NOISE_CHARACTERS.findall(text)) / length
    else:
        return 0.0


def normalize(text: str):
    text = textacy.preprocessing.normalize.unicode(text)
    text = textacy.preprocessing.normalize.whitespace(text)
    text = textacy.preprocessing.remove.accents(text)
    text = textacy.preprocessing.normalize.bullet_points(text)
    text = textacy.preprocessing.normalize.hyphenated_words(text)
    text = textacy.preprocessing.normalize.quotation_marks(text)

    text = redact_emojis(text)
    text = redact_phone_numbers(text)

    return text


