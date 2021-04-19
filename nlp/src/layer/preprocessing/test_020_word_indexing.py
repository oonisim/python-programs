import tempfile
import cProfile
import os
import copy
import pathlib
import logging
from typing import (
    Union
)

import numpy as np
import tensorflow as tf
from common.constant import (
    TYPE_FLOAT,
    NIL,
    UNK
)
import common.weights as weights
from common.function import (
    numerical_jacobian,
)
from common.utility import (
    random_string
)
import function.fileio as fileio
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS
)
from layer.preprocessing import (
    WordIndexing
)
from test.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES,
    GRADIENT_DIFF_ACCEPTANCE_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO
)


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


def _instantiate(name: str, num_nodes: int, path_to_corpus: str):
    word_indexing = WordIndexing.build({
        _NAME: name,
        _NUM_NODES: num_nodes,
        "path_to_corpus": path_to_corpus
    })
    return word_indexing


def _must_fail(name: str, num_nodes: int, path_to_corpus: str, msg: str):
    try:
        _instantiate(name, num_nodes=num_nodes, path_to_corpus=path_to_corpus)
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _must_succeed(name: str, num_nodes: int, path_to_corpus: str, msg: str):
    try:
        if np.random.uniform() > 0.5:
            return _instantiate(name, num_nodes=num_nodes, path_to_corpus=path_to_corpus)
        else:
            corpus = fileio.Function.read_file(path_to_corpus)
            return WordIndexing(name=name, num_nodes=num_nodes, corpus=corpus)
    except Exception as e:
        raise RuntimeError(msg)


def test_020_word_indexing_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_020_word_indexing_instantiation_to_fail"
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


def test_020_word_indexing_instance_properties(caplog):
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    caplog.set_level(logging.DEBUG)
    msg = "Accessing uninitialized property of the layer must fail."
    path_to_corpus: str = download_file()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):

        name = random_string(np.random.randint(1, 10))
        word_indexing = _must_succeed(name=name, num_nodes=1, path_to_corpus=path_to_corpus, msg="need success")

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not word_indexing.name == name:
                raise RuntimeError("word_indexing.name == name should be true")
        except AssertionError as e:
            raise RuntimeError("Access to name should be allowed as already initialized.") from e

        try:
            if not isinstance(word_indexing.logger, logging.Logger):
                raise RuntimeError("isinstance(word_indexing.logger, logging.Logger) should be true")
        except AssertionError as e:
            raise RuntimeError("Access to logger should be allowed as already initialized.") from e

        msg = "WordIndexing vocabulary is available"
        assert word_indexing.is_tensor(word_indexing.vocabulary), msg
        assert word_indexing.vocabulary.size > 3    # at least UNK and NIL and some words
        assert \
            word_indexing.vocabulary[0] == NIL.lower(), \
            f"{NIL.lower()} expected for vocabulary[0] but {word_indexing.vocabulary[0]}"
        assert \
            word_indexing.vocabulary[1] == UNK.lower(), \
            f"{UNK.lower()} expected for vocabulary[0] but {word_indexing.vocabulary[1]}"
        assert \
            isinstance(word_indexing.vocabulary[2], str) and \
            len(word_indexing.vocabulary[2]) > 0, \
            f"word expected for vocabulary[2] but {word_indexing.vocabulary[2]}"

        msg = "word_to_index[vocabulary[index]] must be index"
        length = len(word_indexing.vocabulary)
        index = np.random.randint(0, length)
        assert word_indexing.word_to_index[word_indexing.vocabulary[index]] == index, msg

    profiler.disable()
    profiler.print_stats(sort="cumtime")
