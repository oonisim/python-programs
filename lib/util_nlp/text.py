"""
Module for NLP text manipulation utilities
"""
import string
import unicodedata
from typing import (
    List,
    Set
)

import nltk
import regex as re
import textacy.preprocessing
import unidecode

nltk.download('words')
nltk.download('wordnet')

from nltk.stem import (
    WordNetLemmatizer
)
from nltk.corpus import (
    wordnet,
    words
)

# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
SPACE: str = " "
NLTK_ENGLISH_WORDS = set(words.words())

TAG_AUSTRALIAN_BUSINESS_NUMBER: str = " TAG_ABN "
TAG_AUSTRALIAN_PHONE_NUMBER: str = " TAG_PHONE_NUMBER"
TAG_EMAIL: str = " TAG_EMAIL "
TAG_URL: str = " TAG_URL "


# --------------------------------------------------------------------------------
# Regex
# --------------------------------------------------------------------------------
RE_AUSTRALIAN_BUSINESS_NUMBER: re.Pattern = re.compile(
    pattern=r"ABN[\s:]*\d{2}\s*\d{3}\s*\d{3}\s*\d{3}", flags=re.I
)
RE_AUSTRALIAN_PHONE_NUMBER: re.Pattern = re.compile(
    # Allow 'at 046911112222', 'at 046911112222.', '@046911112222,'
    # patten=r'(?<!\S)(\+?\(?61\)?)?[-\s]*(\(?0?[2-57-8]\)?)[-\s]*(\d\d([- ]' \
    # '(?=\d{3})|(?!\d\d[- ]?\d[- ]))\d\d[- ]?\d[- ]?\d{3})(?!\S)'
    pattern=r'(?<!\S)'\
            '(#|@|at|on)?'\
            '(\+?\(?61\)?)?[-\s]*'\
            '(\(?0?[2-57-8]\)?)[-\s]*'\
            '(\d\d([- ]' \
            '(?=\d{3})|(?!\d\d[- ]?\d[- ]))\d\d[- ]?\d[- ]?\d{3})'\
            '([.,]?)'\
            '(?!\S)',
    flags=re.IGNORECASE
)


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def normalize_typographical_unicode_characters(text: str) -> str:
    """Convert non-ascii typographical characters for punctuations and spaces
    into ascii equivalents.
    Unicode has variants of Ascii characters, e.g. EM-Dash (U+2014), EN-Dash (U+2013).
    for Ascii hyphen-minus (U+002D). Convert them into Ascii equivalent.
    """
    normalized: str = unicodedata.normalize("NFC", text)
    _buf: List[str] = []
    for c in normalized:
        category: str = unicodedata.category(c)
        # --------------------------------------------------------------------------------
        # Punctuation (e.g. Pc, Pd) or Separate space, etc
        # --------------------------------------------------------------------------------
        if category[0] == 'P' or category[0] == 'Z':
            # Removing repeating punctuations as EM-dash is decoded to '--'.
            _buf.append(
                re.sub(
                    pattern=r'([[:punct:][:blank:]])\1+',
                    repl='\\1',
                    string=unidecode.unidecode(c)
                )
            )
        else:
            _buf.append(c)

    return ''.join(_buf)


def redact_non_word_characters(
        text: str, replacement: str = ''
) -> str:
    """
    Redact characters that cannot be used in a word
    re module is unicode aware and can handle non english

    Args:
        text: text to remove the special characters from
        replacement: string to replace with
    Returns: test with special non word characters being removed
    """
    return re.sub(pattern=r'[^\w\s]', repl=replacement, string=text.strip())


def redact_non_english_characters(text: str, replacement: str = '') -> str:
    """Redact non English characters
    Args:
        text: text to remove the special characters from
        replacement: string to replace with
    Returns: test with non EEnglish characters being removed
    """
    regexp: str = rf"[^{string.ascii_letters + string.punctuation + string.whitespace + string.digits}]"
    return re.sub(
        pattern=regexp,
        repl=replacement,
        string=text
    )


def restore_contracted(text: str) -> str:
    """Restore the contracted expression
    Args:
        text: text to run the redaction
    Return: restored text
    """
    text = re.sub(r"won\'t", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"can\'t", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"n\'t", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"\'re", " are", text, flags=re.IGNORECASE)
    text = re.sub(r"\'s", " is", text, flags=re.IGNORECASE)
    text = re.sub(r"\'d", " would", text, flags=re.IGNORECASE)
    text = re.sub(r"\'ll", " will", text, flags=re.IGNORECASE)
    text = re.sub(r"\'t", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"\'ve", " have", text, flags=re.IGNORECASE)
    text = re.sub(r"\'m", " am", text, flags=re.IGNORECASE)
    return text


def redact_email_addresses(text: str, replacement: str = TAG_EMAIL) -> str:
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


def redact_urls(text: str, replacement: str = TAG_URL) -> str:
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


def redact_abn(
        text: str,
        replacement: str = TAG_AUSTRALIAN_BUSINESS_NUMBER
) -> str:
    """Redact ABN with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the ABN
    Return: redacted text
    """
    text = re.sub(
        pattern=RE_AUSTRALIAN_BUSINESS_NUMBER,
        repl=replacement,
        string=text
    )
    return text


def redact_phone_numbers(
        text: str,
        country: str = "AU",
        replacement: str = TAG_AUSTRALIAN_PHONE_NUMBER
) -> str:
    """Redact phone with the replacement
    See https://github.com/google/libphonenumber
    To validate, see https://github.com/daviddrysdale/python-phonenumbers

    Args:
        text: text to run the redaction
        country: phone number country. Only AU is supported for now.
        replacement: replacement for the phone number

    Return: redacted text
    """
    if country == "AU":
        # TODO: handle '+(610) 455 562 400' as invalid
        text = re.sub(
            pattern=RE_AUSTRALIAN_PHONE_NUMBER,
            repl=f'\\1{replacement}\\6',
            string=text
        )
        return text
    else:
        raise NotImplementedError("Currently AU only")


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
        replacement=""
) -> str:
    """Redact noise characters defined by regexp with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the noise

        Replace if repeat more than 3 times (must be 2+)
    Return: redacted text
    """
    # Remove repeating '.'
    text = re.sub(pattern=r"\.{2,}", repl='.', string=text)

    # Remove prepending punctuations and spaces.
    text = re.sub(pattern=r"^[[:punct:][:space:]]*", repl='', string=text)

    # Replace repetition of the same punctuation character with single one.
    # NOTE: There can be valid repetition e.g. Unix ".." as parent directory.
    text = re.sub(
        pattern=r'([[:punct:]])\1+',
        repl='\\1',
        string=text
    )

    # Remove repeating punctuations but not brace or parenthesis
    # e.g. '(...).' or '(...):'
    text = re.sub(
        # Does not work. '^' causes a problem of matching '.' or any.
        # pattern=rf"([{string.punctuation.replace('.', '')}]){{2,}}",

        # Do not remove '(...).' as it has valid meaning.
        pattern='[\\!"#$%&\'\\*\\+,\\-/:;<=>?@\\^_`|~]{2,}',
        repl=replacement,
        string=text
    )

    # Remove trailing specific punctuations and spaces.
    # Some punctuations e.g. '.', "'", '"" can be at the end of the document.
    text = re.sub(
        pattern='[[:space:]\\#$%&\\*\\+,\\-/:;<=>@\\^_`|~]$',
        repl='',
        string=text
    )

    return text.strip()


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
    """Redact non-English words with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the non English word
    Return: redacted text
    """
    return SPACE.join([
        _word for _word in text.split() if is_english_word(_word)
    ])


def redact_white_spaces(text: str, replacement: str = SPACE) -> str:
    """Redact repetition of white spaces with the replacement.
    Args:
        text: text to run the redaction
        replacement: replacement for the white characters
    Return: redacted text
    """
    regexp: str = rf"[{string.whitespace}]+"
    return re.sub(pattern=regexp, repl=replacement, string=text)


def normalize(text: str):
    text = normalize_typographical_unicode_characters(text)
    text = redact_non_english_characters(text)
    text = restore_contracted(text)

    text = textacy.preprocessing.normalize.unicode(text)
    text = textacy.preprocessing.remove.accents(text)
    text = textacy.preprocessing.normalize.bullet_points(text)
    text = textacy.preprocessing.normalize.hyphenated_words(text)
    text = textacy.preprocessing.normalize.quotation_marks(text)
    text = redact_emojis(text)

    text = redact_abn(text)
    text = redact_phone_numbers(text)
    text = redact_urls(text)
    text = redact_email_addresses(text)

    text = redact_noise(text)
    text = redact_white_spaces(text)

    return SPACE.join(text.split())
