import cProfile
import logging
import os
import tempfile
from typing import (
    List,
    Iterable
)

import numpy as np
import tensorflow as tf

from common.constant import (
    TYPE_INT,
    EVENT_NIL,
    EVENT_UNK
)
import function.fileio as fileio
import function.utility as utility
from layer.constants import (
    _NAME,
    _NUM_NODES,
    _LOG_LEVEL
)
from layer.preprocessing import (
    EventIndexing
)
from testing.config import (
    NUM_MAX_TEST_TIMES
)
import testing.layer

Logger = logging.getLogger(__name__)


def download_file(target='shakespeare.txt') -> str:

    if not fileio.Function.is_file(f"~/.keras/datasets/{target}"):
        path_to_file = tf.keras.utils.get_file(
            target,
            f'https://storage.googleapis.com/download.tensorflow.org/data/{target}'
        )
        return path_to_file
    else:
        return f"~/.keras/datasets/{target}"


def _instantiate(name: str, num_nodes: int, path_to_corpus: str, log_level: int = logging.ERROR):
    event_indexing = EventIndexing.build({
        _NAME: name,
        _NUM_NODES: num_nodes,
        "path_to_corpus": path_to_corpus,
        _LOG_LEVEL: log_level
    })
    return event_indexing


def _must_fail(name: str, num_nodes: int, path_to_corpus: str, msg: str):
    try:
        _instantiate(name, num_nodes=num_nodes, path_to_corpus=path_to_corpus)
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _must_succeed(name: str, num_nodes: int, path_to_corpus: str, msg: str, log_level=logging.ERROR):
    try:
        if np.random.uniform() > 0.5:
            return _instantiate(name, num_nodes=num_nodes, path_to_corpus=path_to_corpus, log_level=log_level)
        else:
            corpus = fileio.Function.read_file(path_to_corpus)
            return EventIndexing(name=name, num_nodes=num_nodes, corpus=corpus)
    except Exception as e:
        raise RuntimeError(msg)


def test_020_event_indexing_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_020_event_indexing_instantiation_to_fail"
    path_to_corpus: str = download_file()

    for _ in range(NUM_MAX_TEST_TIMES):
        msg = "Name is string with length > 0."
        _must_fail(name="", num_nodes=1, path_to_corpus=path_to_corpus, msg=msg)

        msg = "path_to_corpus must exist"
        _must_fail(name=name, num_nodes=1, path_to_corpus="", msg=msg)

        msg = "corpus file must not be empty"
        fd, path_to_corpus = tempfile.mkstemp()
        _must_fail(name=name, num_nodes=1, path_to_corpus=path_to_corpus, msg=msg)
        os.unlink(path_to_corpus)


def test_020_event_indexing_instance_properties(caplog):
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    caplog.set_level(logging.DEBUG)
    path_to_corpus: str = download_file()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):

        name = utility.Function.random_string(np.random.randint(1, 10))
        event_indexing = _must_succeed(name=name, num_nodes=1, path_to_corpus=path_to_corpus, msg="need success")

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not event_indexing.name == name:
                raise RuntimeError("event_indexing.name == name should be true")
        except AssertionError as e:
            raise RuntimeError("Access to name should be allowed as already initialized.") from e

        try:
            if not isinstance(event_indexing.logger, logging.Logger):
                raise RuntimeError("isinstance(event_indexing.logger, logging.Logger) should be true")
        except AssertionError as e:
            raise RuntimeError("Access to logger should be allowed as already initialized.") from e

        msg = "EventIndexing vocabulary is available"
        assert event_indexing.is_tensor(event_indexing.vocabulary), msg
        assert event_indexing.vocabulary_size > 3    # at least EVENT_UNK and EVENT_NIL and some words
        assert \
            event_indexing.list_events([0]) == EVENT_NIL.lower(), \
            f"{EVENT_NIL.lower()} expected for vocabulary[0] but {event_indexing.vocabulary[0]}"
        assert \
            event_indexing.list_events([1]) == EVENT_UNK.lower(), \
            f"{EVENT_UNK.lower()} expected for vocabulary[0] but {event_indexing.vocabulary[1]}"

        msg = "event_to_index[vocabulary[index]] must be index"
        length = event_indexing.vocabulary_size
        index = np.random.randint(0, length)
        word, *_ = list(event_indexing.list_events([index]))
        assert event_indexing.list_event_indices([word]) == [index], msg

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_event_indexing_function_multi_lines(caplog):
    """
    Objective:
        Verify the wordindexing function can handle multi line sentences
    Expected:
    """
    sentences = """
    the asbestos fiber <unk> is unusually <unk> once it enters the <unk> 
    with even brief exposures to it causing symptoms that show up decades later researchers said
    """
    caplog.set_level(logging.DEBUG)

    caplog.set_level(logging.DEBUG)
    path_to_corpus: str = download_file()

    name = utility.Function.random_string(np.random.randint(1, 10))
    event_indexing = _must_succeed(
        name=name,
        num_nodes=1,
        path_to_corpus=path_to_corpus,
        msg="need success",
        log_level=logging.DEBUG
    )

    event_indexing.function(sentences)


def test_020_event_indexing_save_load(caplog):
    """
    Objective:
        Verify the save/load function of EventIndexing
    Expected:
    """
    sentences = """
    the asbestos fiber <unk> is unusually <unk> once it enters the <unk> 
    with even brief exposures to it causing symptoms that show up decades later researchers said
    """
    caplog.set_level(logging.DEBUG)

    caplog.set_level(logging.DEBUG)
    path_to_corpus: str = download_file()

    name = utility.Function.random_string(np.random.randint(1, 10))
    event_indexing = _must_succeed(
        name=name,
        num_nodes=1,
        path_to_corpus=path_to_corpus,
        msg="need success",
        log_level=logging.DEBUG
    )
    # --------------------------------------------------------------------------------
    # Run methods and save the results to compare later
    # --------------------------------------------------------------------------------
    indices: List[TYPE_INT] = list(np.random.randint(0, event_indexing.vocabulary_size, 5).tolist())
    sequence = event_indexing.function(sentences)
    probabilities = event_indexing.list_probabilities(['asbestos', 'brief'])
    events = event_indexing.list_events(indices)

    # --------------------------------------------------------------------------------
    # Save the layer state and invalidate the state variables
    # --------------------------------------------------------------------------------
    tester = testing.layer.Function(instance=event_indexing)
    tester.test_save()
    event_indexing._vocabulary = "hoge"
    event_indexing._event_to_index = "hoge"
    event_indexing._probabilities = "hoge"

    # --------------------------------------------------------------------------------
    # Confirm the layer does not function anymore
    # --------------------------------------------------------------------------------
    try:
        event_indexing.function(sentences)
        raise RuntimeError("Must fail with state deleted")
    except Exception as e:
        pass

    try:
        event_indexing.list_probabilities(['asbestos', 'brief'])
        raise RuntimeError("Must fail with state deleted")
    except Exception as e:
        pass

    try:
        event_indexing.list_events(indices)
        raise RuntimeError("Must fail with state deleted")
    except Exception as e:
        pass

    # --------------------------------------------------------------------------------
    # Restore the state and confirm the layer functions as expected.
    # --------------------------------------------------------------------------------
    try:
        # Constraint:
        #   Layer works after state reloaded
        tester.test_load()
        assert np.array_equal(event_indexing.function(sentences), sequence)
        assert event_indexing.list_probabilities(['asbestos', 'brief']) == probabilities
        assert set(event_indexing.list_events(indices)) == set(events)

    except Exception as e:
        raise e
    finally:
        tester.clean()
