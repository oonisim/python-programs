"""
Module for NLP text manipulation utilities
"""
import string
import unicodedata
from typing import (
    List,
    Set,
    Tuple,
    Iterable,
    Generator,
    Union,
)

import spacy
import regex as re
import tldextract
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
RE_EMAIL: re.Pattern = re.compile(
    # --------------------------------------------------------------------------------
    # Valid email format follow the "local-part@domain" format.
    # The local-part of an email address can contain any of the following ASCII characters:
    # * Uppercase and lowercase Latin letters A to Z and a to z
    # * Digits 0 to 9
    # * The following printable characters: !#$%&'*+-/=?^_`{|}~
    #
    # The following guidelines apply to the local-part of a valid email address:
    # * The dot (.) character is allowed but cannot be the first or last character and cannot appear consecutively.
    # * Spaces are not allowed.
    #
    # The domain of an email address can contain any of the following ASCII characters:
    # * Uppercase and lowercase Latin letters A to Z and a to z
    # * Digits 0 to 9
    #
    # The following guidelines apply to the domain of a valid email address:
    # * The domain must match the requirements for a hostname, and include a list of dot (.) separated DNS labels.
    # * The dot (.) character is allowed but cannot be the first or last character and cannot appear consecutively.
    # * No digits are allowed in the top-level domain (TLD). The TLD is the portion of the domain after the dot (.).
    # * The TLD must contain a minimum of 2 and a maximum of 9 characters.
    # * Spaces are not allowed.
    # --------------------------------------------------------------------------------
    pattern=r"""(
        ([^\]'><",):[;\\(@.'\s](\.(?!\.))?)*?           # local name that may include single dot.
        [^\]'><",):[;\\(@.'\s]                          # local name last part 
        @([-a-z0-9]{2,9}(\.(?!\.))?){1,}(\.[a-z]{2,9})  # domain
    )
    """,
    flags=re.IGNORECASE | re.UNICODE | re.VERBOSE
)

RE_URL: re.Pattern = re.compile(
    pattern=r"""
    (?:^|(?<![\w/.]))
    (?:(?:https?://|ftp://|www\d{0,3}\.))
    (?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)
    (?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})
    (?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}
    (?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9])
    (?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9])*
    (?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/\S*)?(?:$|(?![\w?!+&/]))
    """,
    flags=re.IGNORECASE | re.UNICODE | re.VERBOSE
)


# TO allow "ABN: 861-2222-1111"
# RE_AUSTRALIAN_BUSINESS_NUMBER: re.Pattern = re.compile(
#    pattern=r"ABN[\s:]*\d{2}\s*\d{3}\s*\d{3}\s*\d{3}", flags=re.I
# )
RE_AUSTRALIAN_BUSINESS_NUMBER: re.Pattern = re.compile(
    pattern=r"""
    (?<!\S)
    (?:\W*)                      
    (ABN[\s:#]*(\d[-\s]*){11})  # group(1), and group(2) as (\d[-\s]*) 
    (?:\W*)
    (?!\S)
    """,
    flags=re.IGNORECASE | re.VERBOSE
)

RE_AUSTRALIAN_PHONE_NUMBER: re.Pattern = re.compile(
    # Allow 'at 046911112222', 'at 046911112222.', '@046911112222,'
    # patten=r'(?<!\S)(\+?\(?61\)?)?[-\s]*(\(?0?[2-57-8]\)?)[-\s]*(\d\d([- ]' \
    # '(?=\d{3})|(?!\d\d[- ]?\d[- ]))\d\d[- ]?\d[- ]?\d{3})(?!\S)'
    pattern=r'(?<!\S)'\
            '(#|@|at|on)?'\
            '(\+?\(?61\)?)?[-   \s]*'\
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
def regex_match_generator(
        text: str, regexp: Union[str, re.Pattern]
) -> Generator[Tuple[int, re.Match], None, None]:
    redacted: str = ""  # buffer to store ABN redacted text
    cursor: int = 0
    matches: Iterable[re.Match] = re.finditer(
        pattern=regexp, string=text
    )
    for match in matches:
        yield cursor, match
        cursor = match.end(0)

    # Collect the rest in the text
    yield cursor, None


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
    """Redact non-English characters
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


def redact_email_address(text: str, replacement: str = TAG_EMAIL) -> str:
    """Redact email address with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the email
    Return: redacted text
    """
    # return re.sub(
    #     pattern=RE_EMAIL,
    #     string=text,
    #     repl=replacement
    # )

    redacted: str = ""
    for cursor, match in regex_match_generator(text=text, regexp=RE_EMAIL):
        if match:
            email: str = match.group(0)
            # Redact only when the email domain is valid.
            if tldextract.extract(email).registered_domain:
                redacted += text[cursor:match.start(0)] + " TAG_EMAIL "
            else:
                redacted += text[cursor:match.start(0)] + match.group(0)
        else:
            redacted += text[cursor:]

    return redacted


def redact_url(text: str, replacement: str = TAG_URL) -> str:
    """Redact URLs with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the URL
    Return: redacted text
    """
    return re.sub(
        pattern=RE_URL,
        string=text,
        repl=replacement
    )


def is_valid_abn(abn: str) -> bool:
    if abn is None:
        return False

    abn = abn.replace(" ","")
    weight = [10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    weightedSum = 0

    if len(abn) != 11:
        return False
    if not abn.isnumeric():
        return False

    i = 0
    for number in abn:
        weightedSum += int(number) * weight[i]
        i += 1

    # This is the same as subtracting 1 from the first digit.
    weightedSum -= 10

    result = weightedSum % 89 == 0
    return result


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
    # text = re.sub(
    #     pattern=RE_AUSTRALIAN_BUSINESS_NUMBER,
    #     repl=replacement,
    #     string=text
    # )
    # return text

    redacted: str = ""  # buffer to store ABN redacted text
    cursor: int = 0
    matches: Iterable[re.Match] = re.finditer(
        pattern=RE_AUSTRALIAN_BUSINESS_NUMBER, string=text
    )
    for match in matches:
        _abn: str = re.sub(pattern=r'[^\d]', string=match.group(1), repl='')
        if is_valid_abn(_abn):
            redacted += text[cursor:match.start(1)] + replacement
        else:
            redacted += text[cursor:match.start(1)] + match.group(1)

        cursor = match.end(1)

    # Collect the rest in the text
    redacted += text[cursor:]
    return redacted


def redact_phone_number(
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


def redact_emoji(text: str, replacement: str = "") -> str:
    """Redact emoji with the replacement
    Args:
        text: text to run the redaction
        replacement: replacement for the emoji
    Return: redacted text
    """
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

    # Remove repeating punctuations but not brace, parenthesis, quotations.
    # e.g. '(...).' or '(...):'
    text = re.sub(
        # Does not work. '^' causes a problem of matching '.' or any.
        # Do not remove '(...).' as it has valid meaning.
        # Do not remove quotations "..." or '...'.
        # pattern=rf"([{string.punctuation.replace('.', '')}]){{2,}}",
        pattern='[\\!#$%&\\*\\+,\\-/:;<=>?@\\^_`|~]{2,}',
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


def redact_non_english_word(text: str, replacement="<UNK>") -> str:
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
    text = redact_emoji(text)

    text = redact_abn(text)
    text = redact_phone_number(text)
    text = redact_url(text)
    text = redact_email_address(text)

    text = redact_noise(text)
    text = redact_white_spaces(text)

    return SPACE.join(text.split())
