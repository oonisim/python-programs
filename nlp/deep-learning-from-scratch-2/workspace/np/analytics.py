from typing import Dict
import numpy as np


def debug_print_context(sequence: np.ndarray, index: int, stride: int, id_to_word: Dict[int, str], flag: bool = False):
    """Print the context window (N-Gram)
    Args:
        sequence: integer word index sequence
        index: position in the sequence to center the context window
        stride: context windows size = (stride * 2 + 1)
        id_to_word: word index to word dictionary
        flag: Flag to turn on/off the debug print
    """
    if not flag:
        return
    n = len(sequence)
    print("word is {} and context is {}".format(
        id_to_word[sequence[index]],
        [id_to_word[i] for i in sequence[max(0, (index - stride)): min((index + stride) + 1, n)]]
    ))


def create_cooccurrence_matrix(sequence: np.ndarray, co_occurrence_vector_size: int, context_size: int):
    """Create a co-occurrence matrix from the integer word index sequence.
    Args:
        sequence: word index sequence of the original corpus text
        co_occurrence_vector_size:
        context_size: context (N-gram size N) within to check co-occurrences.
    Returns:
        co_occurrence matrix
    """
    assert int(context_size % 2) == 1

    n = sequence_size = len(sequence)
    co_occurrence_matrix = np.zeros(
        (co_occurrence_vector_size, co_occurrence_vector_size),
        dtype=np.int32
    )

    stride = int((context_size - 1) / 2)
    assert (n > stride), "sequence_size {} is less than/equal to stride {}".format(
        n, stride
    )

    for position in range(stride, (n - 1) - stride + 1):
        # --------------------------------------------------------------------------------
        # Consider counting a word multiple time, and the word itself for performance.
        # e.g. stride=2
        # |W|W|W|W|W| If co-occurrences are all same word W at the position, need +4 for W
        # |X|X|W|X|X| If co-occurrences are all same word X, need +4 for X
        # |X|X|W|Y|Y| If co-occurrences X x 2, Y x 2, then need +2 for X and Y respectively.
        # --------------------------------------------------------------------------------
        np.add.at(
            co_occurrence_matrix,
            (
                sequence[position],  # word_id
                sequence[position - stride: (position + stride) + 1]  # indices to co-occurrence words
            ),
            1
        )

    # --------------------------------------------------------------------------------
    # Compensate the +1 self count of a word at each occurrence.
    # F(w) (frequency/occurrences of a word in the sequence) has been extra added besides
    # the expected (2 * stride) * F(w) times, resulting in (context_size) * F(w).
    # --------------------------------------------------------------------------------
    np.fill_diagonal(
        co_occurrence_matrix,
        (co_occurrence_matrix.diagonal() - co_occurrence_matrix.sum(axis=1) / context_size)
    )

    return co_occurrence_matrix


def cooccurrence_words(co_occurrence_matrix: np.ndarray, word: str, word_to_id: Dict[str, int], id_to_word: Dict[int, str]):
    """Provide the co-occurred words for the word"""
    return [(id_to_word[i], count) for i, count in enumerate(co_occurrence_matrix[word_to_id[word]])]


def word_frequency(co_occurrence_matrix: np.ndarray, word: str, word_to_id: Dict[str, int], context_size: int):
    """Number of times when the word occurred in the sequene"""
    assert int(context_size % 2) == 1

    # Each time the word occurrs in the sequence, it will see (CONTEXT_SIZE -1) words.
    co_occurrence_matrix[
        word_to_id[word]
    ].sum() / (context_size - 1)


def total_frequencies(co_occurrence_matrix: np.ndarray, word_to_id: Dict[str, int], context_size: int, padding: str):
    """Sum of all word occurrence except the padding words e.g. NIL (same with vocabulary size excluding paddings)"""
    assert int(context_size % 2) == 1
    return (co_occurrence_matrix.sum() - co_occurrence_matrix[word_to_id[padding]].sum()) / (context_size - 1)
