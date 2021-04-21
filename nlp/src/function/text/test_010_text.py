import string
from . text import Function
from common.constant import (
    EVENT_UNK,
    EVENT_NIL,
    SPACE,
    SPACE,
    TYPE_FLOAT
)


def _must_fail_event_indexing(corpus, msg: str):
    try:
        Function.event_indexing(corpus=corpus)
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _must_succeed_event_indexing(corpus, msg: str):
    try:
        return Function.event_indexing(corpus=corpus)
    except Exception as e:
        raise RuntimeError(msg)


valid_corpus = """
mr. <unk> is chairman of <unk> n.v. the dutch publishing group 
rudolph <unk> N years old and former chairman of consolidated gold fields plc was named a nonexecutive director of this british industrial conglomerate 
a form of asbestos once used to make kent cigarette filters has caused a high percentage of cancer deaths among a group of workers exposed to it more than N years ago researchers reported 
"""


def test_010_text_function_event_indexing_to_fail():
    """
    Objective:
        Verify event_indexing fails with invalid input
    """

    corpus = ''
    msg = "event_indexing must fail with empty string"
    _must_fail_event_indexing(corpus=corpus, msg=msg)

    msg = f"event_indexing must fail with corpus including EVENT_NIL {EVENT_NIL.lower()} words."
    corpus = msg
    _must_fail_event_indexing(corpus=corpus, msg=msg)

    corpus = """

    """
    expression = f"Function.event_indexing(corpus='{corpus}')"
    msg = "event_indexing must fail with empty lines"
    _must_fail_event_indexing(corpus=corpus, msg=msg)

    corpus = """

      %s
    """ % string.punctuation
    msg = "event_indexing must fail with lines of non-events"
    _must_fail_event_indexing(corpus=corpus, msg=msg)


def test_010_text_function_event_indexing_to_succeed():
    """
    Objective:
        Verify event_indexing provides valid instances:

    """
    msg = "event_indexing must succeed with \n[%s]" % valid_corpus
    event_to_index, index_to_event, vocabulary, probabilities = \
        _must_succeed_event_indexing(corpus=valid_corpus, msg=msg)

    assert event_to_index[EVENT_NIL.lower()] == 0, "The index for %s must be 0" % EVENT_NIL.lower()
    assert event_to_index[EVENT_UNK.lower()] == 1, "The index for %s must be 1" % EVENT_UNK.lower()
    assert index_to_event[event_to_index[EVENT_NIL.lower()]] == EVENT_NIL.lower()
    assert index_to_event[event_to_index[EVENT_UNK.lower()]] == EVENT_UNK.lower()
    assert probabilities[EVENT_NIL.lower()] == TYPE_FLOAT(0), "The probability of EVENT_NIL is 0"
    assert (1.0 - sum(probabilities.values())) < 1e-5, \
        f"Sum of probabilities must be 0 but {sum(probabilities.values())}"
    assert len(event_to_index) == len(index_to_event) == len(vocabulary) == len(probabilities)


def test_010_text_function_sentence_to_sequence_to_handles_invalid_inputs():
    """
    Objective:
        Verify sentence_to_sequence fails with invalid input

    Constraints:
        1. Unknown events in inputs are mapped to EVENT_NIL
    """
    msg = "event_indexing must succeed with \n[%s]" % valid_corpus
    event_to_index, index_to_event, vocabulary, probabilities = \
        _must_succeed_event_indexing(corpus=valid_corpus, msg=msg)

    sentence = "sushi very yummy food from japan"
    expected = SPACE.join([EVENT_UNK] * len(sentence.split()))

    # Constraints
    # 1. Unknown events in inputs are mapped to EVENT_NIL
    sequences = Function.sentence_to_sequence(sentences=sentence, event_to_index=event_to_index)
    actual = SPACE.join([index_to_event[index] for index in sequences[0]])
    assert expected == actual, \
        "For sentence [%s]\nsequence is \n%s\nExpected:\n%s\nActual\n%s\n" \
        % (sentence, sequences, expected, actual)


def test_010_text_function_sentence_to_sequence_to_succeed():
    """
    Objective:
        Verify sentence_to_sequence works as expected.

    Constraints:
        1. EVENT_UNK meta event is preserved.
        2. index_to_event reverse the result of event_to_index
           sentence -> event_to_index -> index_to_event -> sentence
    """
    msg = "event_indexing must succeed with \n[%s]" % valid_corpus
    event_to_index, index_to_event, vocabulary, probabilities = \
        _must_succeed_event_indexing(corpus=valid_corpus, msg=msg)

    sentence = "mr. <unk> is chairman of <unk> n.v. the dutch publishing group "
    expected = "mr <unk> is chairman of <unk> n v the dutch publishing group"

    # Constraints
    # 1. EVENT_UNK meta event is preserved.
    # 2. index_to_event reverse the result of event_to_index
    sequences = Function.sentence_to_sequence(sentences=sentence, event_to_index=event_to_index)
    actual = SPACE.join([index_to_event[index] for index in sequences[0]])
    assert expected == actual, "Expected:\n%s\nActual\n%s\n" % (expected, actual)
