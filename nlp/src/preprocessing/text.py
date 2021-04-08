from typing import(
    Dict,
    List
)
import re
import string
import numpy as np
from preprocessing.constant import (
    DELIMITER,
    NIL
)

PAD_MODE_PREPEND = 'prepend'
PAD_MODE_APPEND = 'append'
PAD_MODE_SQUEEZE = 'squeeze'


def pad_text(
    corpus: str,
    mode: str = PAD_MODE_PREPEND,
    delimiter: str = DELIMITER,
    padding: str = NIL,
    length: int = 1
) -> str:
    """Prepend and append size times of the padding word to the corpus
    Args:
        corpus:
        mode: how to pad. PREPEND|APPEND|SQUEEZE
        delimiter: delimiter between words
        padding: a word to pad
        length: length of words to pad
    """
    assert corpus and len(corpus) > 0 and isinstance(corpus, str)

    if mode == PAD_MODE_SQUEEZE:
        padded = delimiter.join(
            [padding] * length + [corpus] + [padding] * length
        )
    elif mode == PAD_MODE_PREPEND:
        padded = delimiter.join(
            [padding] * length + [corpus]
        )
    elif mode == PAD_MODE_APPEND:
        padded = delimiter.join(
            [corpus] + [padding] * length
        )
    else:
        raise ValueError("unexpected mode {}".format(mode))
    """
    padded = sum(
        [ 
            [ NIL ] * STRIDE, 
            corpus.split(' '),
            [ NIL ] * STRIDE
        ],
        start=[]
    )
    """
    return padded


def replace_punctuations(corpus: str, replacement: str) -> str:
    """Replace punctuations in the corpus
    Args:
        corpus: sequence of words
        replacement: char to replace
    Returns
        corpus: corpus with punctuation being replaced with replacement.
    """
    assert len(replacement) == 1, \
        f"replacement {replacement} must be length 1 character."

    table = str.maketrans(string.punctuation, replacement * len(string.punctuation))
    corpus.translate(table)

    assert len(corpus) > 0, "corpus needs words other than punctuations."
    return corpus


def standardize(text: str) -> str:
    """Standardize the text
    1. Lower the string
    2. Remove punctuation
    3. Remove white space, new lines, carriage returns
    Args:
        text: sequence of words
    Returns
        standardized: standardized text
    """
    assert isinstance(text, str) and len(text) > 0
    replacement = " "
    pattern: str = '[%s%s]+' % (re.escape(string.punctuation), r"\s")

    standardized: str = re.compile(pattern).sub(repl=replacement, string=text).lower().strip()
    assert len(standardized) > 0, f"Text [{text}] needs words other than punctuations."
    return standardized


def word_indexing(corpus: str):
    """Generate word indices
    Args:
        corpus: A string including sentences to process.
    Returns:
        vocabulary: unique words in the corpus
        id_to_word: word index to word mapping
        word_to_id: word to word index mapping
    """
    words = standardize(corpus).split()
    vocabulary = ['UNK'] + list(set(words))
    id_to_word: Dict[int, str] = dict(enumerate(vocabulary))
    word_to_id: Dict[str, int] = dict(zip(id_to_word.values(), id_to_word.keys()))

    return words, vocabulary, id_to_word, word_to_id


def text_to_sequence(
        corpus,
        word_to_id: dict
) -> List[str]:
    """Generate integer sequence word
    Args:
        corpus: A string including sentences to process.
        word_to_id: word to integer index mapping
    Returns:
        sequence:
            word indices to every word in the originlal corpus as as they appear in it.
            The objective of sequence is to preserve the original corpus but as numerical indices.
    """
    return [word_to_id.get(w, 0) for w in standardize(corpus).split()]


if __name__ == "__main__":
    print("'{}'".format(pad_text("tako ika bin")))
    assert pad_text("tako ika bin") == NIL + DELIMITER + "tako ika bin"
