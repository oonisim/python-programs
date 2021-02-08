from typing import(
    Dict,
    List
)
import numpy as np
from .constant import (
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


import re


def text_to_sequence(corpus) -> (np.ndarray, Dict[str, int], Dict[int, str], int):
    """Generate integer word indices for the words in the corpus and return
    Args:
        corpus: A string including sentences to process.
    Returns:
        sequence:
            A numpy array of word indices to every word in the originlal corpus as as they appear in it.
            The objective of sequence is to preserve the original corpus but as numerical indices.
        word_to_id: A dictionary to map a word to a word index
        id_to_word: A dictionary to map a word index to a word
        vocabulary_size: Number of words identified in the corpus
    """
    words = re.compile(r'[\s\t]+').split(corpus.lower())

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # Word index starts with 0. Total words = max(word index) + 1
    vocabulary_size = new_id + 1
    assert vocabulary_size == (max(word_to_id.values()) + 1)

    sequence = np.array([word_to_id[w] for w in words])
    return sequence, word_to_id, id_to_word, vocabulary_size


if __name__ == "__main__":
    print("'{}'".format(pad_text("tako ika bin")))
    assert pad_text("tako ika bin") == NIL + DELIMITER + "tako ika bin"
