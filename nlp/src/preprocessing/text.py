from typing import(
    Dict,
    List
)
import re
import string
import collections
import numpy as np
from preprocessing.constant import (
    DELIMITER,
    NIL
)
from common.constant import (
    TYPE_FLOAT
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
        index_to_word: word index to word mapping
        word_to_index: word to word index mapping
        probabilities: word occurrence probabilities
    """
    words = standardize(corpus).split()

    total = len(words)
    counts = collections.Counter(words)
    probabilities = {word: (count / total) for (word, count) in counts.items()}

    vocabulary = ['UNK'] + list(set(words))
    index_to_word: Dict[int, str] = dict(enumerate(vocabulary))
    word_to_index: Dict[str, int] = dict(zip(index_to_word.values(), index_to_word.keys()))

    del words, total, counts
    return vocabulary, index_to_word, word_to_index, probabilities


def sub_sampling(corpus: str, power: TYPE_FLOAT = TYPE_FLOAT(0.75)):
    total = len(words := standardize(corpus).split())
    counts = collections.Counter(words)
    probabilities = {word: np.power((count / total), power) for (word, count) in counts.items()}

    sigma = np.sum(list(probabilities.values()))
    probabilities = {word: (p/sigma) for (word, p) in probabilities.items()}
    assert (1.0 - np.sum(list(probabilities.values()))) < 1e-5

    del words, total, counts, sigma
    return probabilities


def text_to_sequence(
        corpus,
        word_to_index: dict
) -> List[List[str]]:
    """Generate integer sequence word
    Args:
        corpus: A string including sentences to process.
        word_to_index: word to integer index mapping
    Returns:
        sequence:
            word indices to every word in the originlal corpus as as they appear in it.
            The objective of sequence is to preserve the original corpus but as numerical indices.
    """
    lines = []
    for line in corpus.split("\n"):
        lines.append([word_to_index.get(w, 0) for w in standardize(line).split()])

    return lines


if __name__ == "__main__":
    print("'{}'".format(pad_text("tako ika bin")))
    assert pad_text("tako ika bin") == NIL + DELIMITER + "tako ika bin"
