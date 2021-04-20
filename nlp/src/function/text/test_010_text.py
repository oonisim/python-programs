import string
from . text import Function
from common.constant import (
    UNK,
    NIL,
    SPACE,
    TYPE_FLOAT
)


def _must_fail_word_indexing(corpus, msg: str):
    try:
        Function.word_indexing(corpus=corpus)
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _must_succeed_word_indexing(corpus, msg: str):
    try:
        return Function.word_indexing(corpus=corpus)
    except Exception as e:
        raise RuntimeError(msg)


valid_corpus = """
mr. <unk> is chairman of <unk> n.v. the dutch publishing group 
rudolph <unk> N years old and former chairman of consolidated gold fields plc was named a nonexecutive director of this british industrial conglomerate 
a form of asbestos once used to make kent cigarette filters has caused a high percentage of cancer deaths among a group of workers exposed to it more than N years ago researchers reported 
"""


def test_010_text_function_word_indexing_to_fail():
    """
    Objective:
        Verify word_indexing fails with invalid input
    """

    corpus = ''
    msg = "word_indexing must fail with empty string"
    _must_fail_word_indexing(corpus=corpus, msg=msg)

    msg = f"word_indexing must fail with corpus including NIL {NIL.lower()} words."
    corpus = msg
    _must_fail_word_indexing(corpus=corpus, msg=msg)

    corpus = """

    """
    expression = f"Function.word_indexing(corpus='{corpus}')"
    msg = "word_indexing must fail with empty lines"
    _must_fail_word_indexing(corpus=corpus, msg=msg)

    corpus = """

      %s
    """ % string.punctuation
    msg = "word_indexing must fail with lines of non-words"
    _must_fail_word_indexing(corpus=corpus, msg=msg)


def test_010_text_function_word_indexing_to_succeed():
    """
    Objective:
        Verify word_indexing provides valid instances:

    """
    msg = "word_indexing must succeed with \n[%s]" % valid_corpus
    word_to_index, index_to_word, vocabulary, probabilities = \
        _must_succeed_word_indexing(corpus=valid_corpus, msg=msg)

    assert word_to_index[NIL.lower()] == 0, "The index for %s must be 0" % NIL.lower()
    assert word_to_index[UNK.lower()] == 1, "The index for %s must be 1" % UNK.lower()
    assert index_to_word[word_to_index[NIL.lower()]] == NIL.lower()
    assert index_to_word[word_to_index[UNK.lower()]] == UNK.lower()
    assert probabilities[NIL.lower()] == TYPE_FLOAT(0), "The probability of NIL is 0"
    assert (1.0 - sum(probabilities.values())) < 1e-5, \
        f"Sum of probabilities must be 0 but {sum(probabilities.values())}"
    assert len(word_to_index) == len(index_to_word) == len(vocabulary) == len(probabilities)


def test_010_text_function_sentence_to_sequence_to_handles_invalid_inputs():
    """
    Objective:
        Verify sentence_to_sequence fails with invalid input

    Constraints:
        1. Unknown words in inputs are mapped to NIL
    """
    msg = "word_indexing must succeed with \n[%s]" % valid_corpus
    word_to_index, index_to_word, vocabulary, probabilities = \
        _must_succeed_word_indexing(corpus=valid_corpus, msg=msg)

    sentence = "sushi very yummy food from japan"
    expected = "<nil> <nil> <nil> <nil> <nil> <nil>"

    # Constraints
    # 1. Unknown words in inputs are mapped to NIL
    sequences = Function.sentence_to_sequence(sentences=sentence, word_to_index=word_to_index)
    actual = SPACE.join([index_to_word[index] for index in sequences[0]])
    assert expected == actual, \
        "For sentence [%s]\nsequence is \n%s\nExpected:\n%s\nActual\n%s\n" \
        % (sentence, sequences, expected, actual)


def test_010_text_function_sentence_to_sequence_to_succeed():
    """
    Objective:
        Verify sentence_to_sequence works as expected.

    Constraints:
        1. UNK meta word is preserved.
        2. index_to_word reverse the result of word_to_index
           sentence -> word_to_index -> index_to_word -> sentence
    """
    msg = "word_indexing must succeed with \n[%s]" % valid_corpus
    word_to_index, index_to_word, vocabulary, probabilities = \
        _must_succeed_word_indexing(corpus=valid_corpus, msg=msg)

    sentence = "mr. <unk> is chairman of <unk> n.v. the dutch publishing group "
    expected = "mr <unk> is chairman of <unk> n v the dutch publishing group"

    # Constraints
    # 1. UNK meta word is preserved.
    # 2. index_to_word reverse the result of word_to_index
    sequences = Function.sentence_to_sequence(sentences=sentence, word_to_index=word_to_index)
    actual = SPACE.join([index_to_word[index] for index in sequences[0]])
    assert expected == actual, "Expected:\n%s\nActual\n%s\n" % (expected, actual)
