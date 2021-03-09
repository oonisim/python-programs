from typing import Dict
import numpy as np
import logging


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
    """Create a co-occurrence matrix from the integer index sequence.
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


# ================================================================================
# PMI
# ================================================================================
def pmi(co_occurrence_matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """ Compute PMI (Pointwise Mutual Information) PMI(x, y) = P(x, y) / ( P(x) * P(y) ) from a co occurrence matrix.
    When P(x) is the probability of the word x to occur in a sequence and P(x) for P(y).
    If P(x), P(y) are independent, expected co-occurrence probability P(x, y) is P(x) * P(y).
    When PMI(x, y) > 1 then x and y may have a co-relation.
    Args:
        co_occurrence_matrix:
            including NIL so as to get F(w) which is occurrence of the word w.
        eps:
            small value to avoid log2() -> inf, not for divide by zero.
    """
    c = co_occurrence_matrix

    assert(c.sum() > 0)

    m = mutual_occurrence_probability = np.divide(c, c.sum())
    s = single_occurrence_probability = np.divide(c.sum(axis=1), c.sum())
    # --------------------------------------------------------------------------------
    # Because c is symmetric, s and t are the same in 1D form. Hence no need to get t.
    # To shape s * t into c.shape for (m / i) operation, use numpy.ix_() to broadcast.
    # --------------------------------------------------------------------------------
    # t = target_occurrence_probability = np.divide(c.T.sum(axis=0), c.sum())
    t = s
    sx, tx = np.ix_(s, t)

    # --------------------------------------------------------------------------------
    # NIL word padding gives F(w) via Z.sum(axis=1), and c has a row for NIL whose
    # co-occurrences with other words are all zero because NIL will not see other words.
    # "See" is an analogy. W co-occur with X -> W sees X.
    # To avoid divide by zero for NIL word, add eps.
    # --------------------------------------------------------------------------------
    i = independent_cooccurrence_probability = sx * tx + eps

    # print("p(x) is {} shape {}".format(sx, sx.shape))
    # print("p(y) is {} shape {}".format(tx, tx.shape))
    # print("p(s @ t) is {} shape {}\n".format(i, i.shape))
    with np.errstate(divide='ignore'):
        # Add eps to avoid log2() -> inf
        _pmi = np.log2(m / i + eps)

    return _pmi


def ppmi(co_occurrence_matrix: np.ndarray, eps: float = 1e-8):
    """Positive PMI
    Elements in co_occurrence_matrix is zero between words that have no co-occurrence.
    Then PMI value log2(0) -> -inf. Replace them with zero, which leave only positives.
    """
    _pmi = pmi(co_occurrence_matrix, eps)
    _pmi[_pmi < 0] = 0
    return _pmi


def create_context_set(sequence: np.ndarray, context_size: int) -> (np.ndarray, np.ndarray):
    """Create a set of context and its label from a integer sequence.
    When there is a sequence [1, 2, 3, 4, 5, 6 7] with context_size = 5:
        contexts = [[1,2,4,5], [2,3,5,6],[3,4,6,7]]
        labels   = [3, 4, 5]
    Args:
        sequence: index sequence of the original corpus
        context_size: context (N-gram size N) within to check co-occurrences.
    Returns:
        contexts, labels
    """
    n = sequence_size = len(sequence)
    assert n >= context_size
    assert int(context_size % 2) == 1

    stride = int((context_size - 1) / 2)
    contexts = np.array([]).reshape(0, context_size-1)
    labels = np.array([]).reshape(0,)

    # The first position of the last context in the sequence is (n-1) - context_size +1.
    # For range(), needs +1 to include the start position of the last context.
    # [(n-1) - context_size +1] + 1
    for position in range(0, (n - 1) - context_size + 2):
        # --------------------------------------------------------------------------------
        # Consider counting a word multiple time, and the word itself for performance.
        # e.g. stride=2
        # |W|W|W|W|W| If co-occurrences are all same word W at the position, need +4 for W
        # |X|X|W|X|X| If co-occurrences are all same word X, need +4 for X
        # |X|X|W|Y|Y| If co-occurrences X x 2, Y x 2, then need +2 for X and Y respectively.
        # --------------------------------------------------------------------------------
        contexts = np.vstack([
            contexts,
            sequence[np.r_[
                position: (position+stride),                         # left stride
                (position + stride) + 1 : (position + context_size)  # right stride
            ]]
        ])
        labels = np.append(labels, sequence[position + stride])

    return contexts.astype(int), labels.astype(int)
