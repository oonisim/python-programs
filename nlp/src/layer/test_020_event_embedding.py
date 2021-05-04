import cProfile
import copy
import logging
import os
import random
from typing import (
    Tuple
)

import numpy as np
import tensorflow as tf

import testing.layer
from common.constant import (
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_TENSOR,
    EVENT_META_ENTITIES
)
from function.nn import (
    Function
)
from layer.preprocessing import (
    EventIndexing,
    EventContext
)
from layer.preprocessing.test_020_event_context import (
    _instantiate as _instantiate_event_context
)
from layer.preprocessing.test_020_event_indexing import (
    download_file,
    _instantiate as ___instantiate_event_indexing
)
from optimizer import (
    SGD
)
from testing.config import (
    NUM_MAX_TEST_TIMES
)
from .embedding import (
    Embedding
)

Logger = logging.getLogger(__name__)
# Not creating XLA devices, tf_xla_enable_xla_devices not set
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def _instantiate_event_indexing(
        name: str = __name__,
        log_level: int = logging.ERROR
):
    path_to_corpus: str = download_file()
    indexing: EventIndexing = ___instantiate_event_indexing(
        name=name,
        num_nodes=1,
        path_to_corpus=path_to_corpus,
        log_level=log_level
    )
    return indexing


def _instantiate(
        name: str = __name__,
        num_nodes: TYPE_INT = 1,
        target_size: TYPE_INT = 1,
        context_size: TYPE_INT = 4,
        negative_sample_size: TYPE_INT = 10,
        event_vector_size: TYPE_INT = 20,
        dictionary: EventIndexing = None,
        W: TYPE_TENSOR = None,
        log_level: int = logging.ERROR
) -> Tuple[Embedding, EventContext]:

    event_context: EventContext = _instantiate_event_context(
        name=name,
        num_nodes=TYPE_INT(1),
        window_size=(context_size+target_size),
        event_size=target_size
    )

    embedding: Embedding = Embedding(
        name=name,
        num_nodes=num_nodes,
        target_size=target_size,
        context_size=context_size,
        negative_sample_size=negative_sample_size,
        event_vector_size=event_vector_size,
        dictionary=dictionary,
        W=W,
        log_level=log_level
    )

    return embedding, event_context


def _must_fail(
        name: str = __name__,
        num_nodes: TYPE_INT = 1,
        target_size: TYPE_INT = 1,
        context_size: TYPE_INT = 4,
        negative_sample_size: TYPE_INT = 10,
        event_vector_size: TYPE_INT = 20,
        dictionary: EventIndexing = None,
        W: TYPE_TENSOR = None,
        log_level: int = logging.ERROR,
        msg: str = "must fail"
):
    try:
        _instantiate(
            name=name,
            num_nodes=num_nodes,
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=log_level
        )
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _must_succeed(
        name: str = __name__,
        num_nodes: TYPE_INT = 1,
        target_size: TYPE_INT = 1,
        context_size: TYPE_INT = 4,
        negative_sample_size: TYPE_INT = 10,
        event_vector_size: TYPE_INT = 20,
        dictionary: EventIndexing = None,
        W: TYPE_TENSOR = None,
        log_level: int = logging.ERROR,
        msg: str = "must succeed"
) -> Tuple[Embedding, EventContext]:
    try:
        return _instantiate(
            name=name,
            num_nodes=num_nodes,
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=log_level
        )
    except Exception as e:
        raise RuntimeError(msg)


def test_020_embedding_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_020_embedding_instantiation_to_fail"

    # First validate the correct configuration, then change parameter one by one.
    dictionary: EventIndexing = _instantiate_event_indexing()
    target_size = TYPE_INT(np.random.randint(1, 10))
    context_size = TYPE_INT(2 * np.random.randint(1, 10))
    negative_sample_size = TYPE_INT(np.random.randint(5, 20))
    event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 100))
    W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

    _must_succeed(
        name=name,
        num_nodes=TYPE_INT(1),
        target_size=target_size,
        context_size=context_size,
        negative_sample_size=negative_sample_size,
        event_vector_size=event_vector_size,
        dictionary=dictionary,
        W=W,
        log_level=logging.DEBUG,
        msg="must succeed"
    )

    _must_succeed(
        name=name,
        num_nodes=(1+negative_sample_size),
        target_size=target_size,
        context_size=context_size,
        negative_sample_size=negative_sample_size,
        event_vector_size=event_vector_size,
        dictionary=dictionary,
        log_level=logging.DEBUG,
        msg="must succeed without W"
    )

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(NUM_MAX_TEST_TIMES):
        target_size = TYPE_INT(np.random.randint(1, 10))
        context_size = TYPE_INT(2 * np.random.randint(1, 10))
        negative_sample_size = TYPE_INT(np.random.randint(5, 20))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 100))

        msg = "Name is string with length > 0."
        _must_fail(
            name="",
            num_nodes=(1 + negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            log_level=logging.DEBUG,
            msg=msg
        )

        msg = "num_nodes must > 0."
        _must_fail(
            name=name,
            num_nodes=TYPE_INT(0),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            log_level=logging.DEBUG,
            msg=msg
        )

        # msg = "num_nodes is 1+negative_sample_size but does not has to be enforced"
        # _must_fail(
        #     name=name,
        #     num_nodes=TYPE_INT(1+negative_sample_size),
        #     target_size=target_size,
        #     context_size=context_size,
        #     negative_sample_size=negative_sample_size,
        #     event_vector_size=event_vector_size,
        #     dictionary=dictionary,
        #     log_level=logging.DEBUG,
        #     msg=msg
        # )

        msg = "target size must be >0."
        _must_fail(
            name=name,
            num_nodes=(1 + negative_sample_size),
            target_size=TYPE_INT(0),
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            log_level=logging.DEBUG,
            msg=msg
        )

        msg = "context_size must be >0."
        _must_fail(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=TYPE_INT(0),
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            log_level=logging.DEBUG,
            msg=msg
        )

        msg = "negative_sample_size must be >0."
        _must_fail(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=TYPE_INT(0),
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            log_level=logging.DEBUG,
            msg=msg
        )

        msg = "event_vector_size must be >0."
        _must_fail(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=TYPE_INT(0),
            dictionary=dictionary,
            log_level=logging.DEBUG,
            msg=msg
        )

        msg = "dictionary must be of type EventIndexing"
        _must_fail(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=["hoge", None][np.random.randint(0, 2)],
            log_level=logging.DEBUG,
            msg=msg
        )

        msg = "vocabulary_size must >= " \
              "(context_size + target_size) + " \
              "negative_sample_size + " \
              "len(EVENT_META_ENTITIES)"
        length = (context_size + target_size) + negative_sample_size
        corpus = " ".join(str(i) for i in range(length))
        _indexing_dummy = EventIndexing(
            name=__name__,
            num_nodes=1,
            corpus=corpus
        )
        assert _indexing_dummy.vocabulary_size == length + len(EVENT_META_ENTITIES)
        _must_succeed(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=TYPE_INT(target_size),
            context_size=TYPE_INT(context_size),
            negative_sample_size=TYPE_INT(negative_sample_size),
            event_vector_size=TYPE_INT(event_vector_size),
            dictionary=_indexing_dummy,
            log_level=logging.DEBUG,
            msg=msg
        )
        _must_fail(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=TYPE_INT(target_size) + 1,
            context_size=TYPE_INT(context_size),
            negative_sample_size=TYPE_INT(negative_sample_size),
            event_vector_size=TYPE_INT(event_vector_size),
            dictionary=_indexing_dummy,
            log_level=logging.DEBUG,
            msg=msg
        )

        msg = "vocabulary_size must be > W.shape[0]"
        w = np.random.randn(
            dictionary.vocabulary_size - np.random.randint(1, dictionary.vocabulary_size),
            event_vector_size
        )
        _must_fail(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=TYPE_INT(target_size),
            context_size=TYPE_INT(context_size),
            negative_sample_size=TYPE_INT(negative_sample_size),
            event_vector_size=TYPE_INT(event_vector_size),
            dictionary=dictionary,
            W=w,
            log_level=logging.DEBUG,
            msg=msg
        )

        msg = "event_vector_size must be W.shape[1]"
        offset = np.random.randint(1, event_vector_size) * (1 if random.random() < 0.5 else -1)
        assert (event_vector_size + offset) > 0, \
            "%s %s %s" % (event_vector_size, offset, (event_vector_size + offset))
        w = np.random.randn(
            dictionary.vocabulary_size,
            (event_vector_size + offset)
        )
        _must_fail(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=TYPE_INT(target_size),
            context_size=TYPE_INT(context_size),
            negative_sample_size=TYPE_INT(negative_sample_size),
            event_vector_size=TYPE_INT(event_vector_size),
            dictionary=dictionary,
            W=w,
            log_level=logging.DEBUG,
            msg=msg
        )

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_embedding_instance_properties_access_to_fail(caplog):
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_embedding_instance_properties_access_to_fail"
    dictionary: EventIndexing = _instantiate_event_indexing()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        # First validate the correct configuration, then change parameter one by one.
        target_size = TYPE_INT(np.random.randint(1, 10))
        context_size = TYPE_INT(2 * np.random.randint(1, 10))
        negative_sample_size = TYPE_INT(np.random.randint(5, 20))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 100))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        embedding, event_context = _must_succeed(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
            msg="must succeed"
        )

        # --------------------------------------------------------------------------------
        # To fail
        # --------------------------------------------------------------------------------
        msg = "Accessing uninitialized property of the layer must fail."
        try:
            print(embedding.X)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(embedding.N)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(embedding.dX)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(embedding.dW)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(embedding.Y)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(embedding.dY)
            raise RuntimeError(msg)
        except AssertionError:
            pass

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_embedding_instance_properties_access_to_succeed(caplog):
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the initialized parameters and succeeds.
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_embedding_instance_properties_access_to_succeed"
    dictionary: EventIndexing = _instantiate_event_indexing()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(NUM_MAX_TEST_TIMES):
        # First validate the correct configuration, then change parameter one by one.
        target_size = TYPE_INT(np.random.randint(1, 10))
        context_size = TYPE_INT(2 * np.random.randint(1, 10))
        negative_sample_size = TYPE_INT(np.random.randint(5, 20))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 100))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size).astype(TYPE_FLOAT)

        embedding, event_context = _must_succeed(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
            msg="must succeed"
        )

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not embedding.name == name:
                raise RuntimeError("embedding.name == name should be true")
        except AssertionError:
            raise RuntimeError("Access to name should be allowed as already initialized.")

        try:
            if not embedding.V == dictionary.vocabulary_size:
                raise RuntimeError("embedding.V == vocabulary_size should be true")
        except AssertionError:
            raise RuntimeError("Access to V should be allowed as already initialized.")

        try:
            if not embedding.E == target_size:
                raise RuntimeError("embedding.V == target_size should be true")
        except AssertionError:
            raise RuntimeError("Access to E should be allowed as already initialized.")

        try:
            if not embedding.C == context_size:
                raise RuntimeError("embedding.C == context_size should be true")
        except AssertionError:
            raise RuntimeError("Access to C should be allowed as already initialized.")

        try:
            if not embedding.window_size == target_size+context_size:
                raise RuntimeError("embedding.window_size == target_size+context_size should be true")
        except AssertionError:
            raise RuntimeError("Access to window_size should be allowed as already initialized.")

        try:
            if not embedding.SL == negative_sample_size:
                raise RuntimeError("embedding.negative_sample_size == negative_sample_size should be true")
        except AssertionError:
            raise RuntimeError("Access to negative_sample_size should be allowed as already initialized.")

        try:
            if embedding.dictionary is not dictionary:
                raise RuntimeError("embedding.dictionary is dictionary should be true")
        except AssertionError:
            raise RuntimeError("Access to dictionary should be allowed as already initialized.")

        try:
            # if embedding.W is not W: # Embedding internally deepcopy W to avoid unexpected change
            if not np.array_equal(embedding.W, W):
                raise RuntimeError("np.array_equal(embedding.W, W) should be true")
        except AssertionError:
            raise RuntimeError("Access to W should be allowed as already initialized.")

        try:
            if not isinstance(embedding.optimizer, SGD):
                raise RuntimeError("isinstance(embedding.optimizer, SGD) should be true")
        except AssertionError:
            raise RuntimeError("Access to optimizer should be allowed as already initialized.")

        try:
            opt = SGD()
            if not embedding.lr == opt.lr:
                raise RuntimeError("embedding.lr == lr should be true")
        except AssertionError:
            raise RuntimeError("Access to lr should be allowed as already initialized.")

        try:
            opt = SGD()
            if not embedding.l2 == opt.l2:
                raise RuntimeError("embedding.l2 == context_size should be true")
        except AssertionError:
            raise RuntimeError("Access to l2 should be allowed as already initialized.")

        try:
            if not np.array_equal(embedding.S["target_size"], embedding.E):
                raise RuntimeError("embedding.E == E should be true")
        except AssertionError:
            raise RuntimeError("Access to S['target_size'] should be allowed as already initialized.")

        try:
            if not np.array_equal(embedding.S["context_size"], embedding.C):
                raise RuntimeError("embedding.C == C should be true")
        except AssertionError:
            raise RuntimeError("Access to S['context_size'] should be allowed as already initialized.")

        try:
            if not np.array_equal(embedding.S["negative_sample_size"], embedding.SL):
                raise RuntimeError("embedding.SL == SL should be true")
        except AssertionError:
            raise RuntimeError("Access to S['negative_sample_size'] should be allowed as already initialized.")

        try:
            if embedding.S["dictionary"] is not embedding.dictionary:
                raise RuntimeError("embedding.dictionary is dictionary should be true")
        except AssertionError:
            raise RuntimeError("Access to S['dictionary'] should be allowed as already initialized.")

        try:
            if not np.array_equal(embedding.S["W"], embedding.W):
                raise RuntimeError("embedding.W == W should be true")
        except AssertionError:
            raise RuntimeError("Access to S['W'] should be allowed as already initialized.")

        try:
            if not isinstance(embedding.logger, logging.Logger):
                raise RuntimeError("isinstance(embedding.logger, logging.Logger) should be true")
        except AssertionError:
            raise RuntimeError("Access to logger should be allowed as already initialized.")

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_embedding_function_multi_lines(caplog):
    """
    Objective:
        Verify the EventIndexing function can handle multi line sentences
    Expected:
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_embedding_function_multi_lines"

    sentences = """
    Verify the EventIndexing function can handle multi line sentences
    the asbestos fiber <unk> is unusually <unk> once it enters the <unk> 
    with even brief exposures to it causing symptoms that show up decades later researchers said
    """

    dictionary: EventIndexing = _instantiate_event_indexing()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        # First validate the correct configuration, then change parameter one by one.
        target_size = TYPE_INT(np.random.randint(1, 3))
        context_size = TYPE_INT(2 * np.random.randint(1, 5))
        negative_sample_size = TYPE_INT(np.random.randint(1, 5))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 20))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        embedding, event_context = _must_succeed(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
            msg="must succeed"
        )

        sequences = dictionary.function(sentences)
        target_context_pairs = event_context.function(sequences)
        embedding.function(target_context_pairs)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_embedding_gradient_single_sentence(caplog):
    """
    Objective:
        Verify the EventIndexing gradient
    Expected:
        gradient()
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_embedding_gradient_multi_lines"

    sentences = """
    Verify the EventIndexing function can handle multi line sentences
    the asbestos fiber <unk> is unusually <unk> once it enters the <unk> 
    with even brief exposures to it causing symptoms that show up decades later researchers said
    """
    sentence = "I am a cat who has no name"

    dictionary: EventIndexing = _instantiate_event_indexing()

    profiler = cProfile.Profile()
    profiler.enable()

    def L(x):
        loss = Function.sum(
            x, axis=None, keepdims=False
        )
        return loss

    for _ in range(NUM_MAX_TEST_TIMES):
        # First validate the correct configuration, then change parameter one by one.
        target_size = TYPE_INT(1)
        context_size = TYPE_INT(2)
        negative_sample_size = TYPE_INT(1)
        event_vector_size: TYPE_INT = TYPE_INT(3)
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        embedding, event_context = _must_succeed(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
            msg="must succeed"
        )
        embedding.objective = L
        sequences = dictionary.function(sentence)
        target_context_pairs = event_context.function(sequences)

        # --------------------------------------------------------------------------------
        # Forward path
        # --------------------------------------------------------------------------------
        Y = embedding.function(target_context_pairs)
        EDWe, EDWs, EDWc = embedding.gradient_numerical()

        # --------------------------------------------------------------------------------
        # Backward path
        # --------------------------------------------------------------------------------
        dY = Function.ones(shape=Function.tensor_shape(Y))
        embedding.gradient(dY)

        # --------------------------------------------------------------------------------
        # Backward path
        # --------------------------------------------------------------------------------
        dWe, dWs, dWc = embedding.update()

        # ********************************************************************************
        # Constraint: dW is close to EDW
        # ********************************************************************************
        Function.assert_all_close(EDWe, dWe, msg="Expected dWe:\n%s\nActual\n%s\n")
        Function.assert_all_close(EDWs, dWs, msg="Expected dWs:\n%s\nActual\n%s\n")
        Function.assert_all_close(EDWc, dWc, msg="Expected dWc:\n%s\nActual\n%s\n")

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_embedding_save_load(caplog):
    """
    Objective:
        Verify the save/load function of EventIndexing
    Expected:
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_embedding_save_load"
    sentences = """
    Verify 
    """

    dictionary: EventIndexing = _instantiate_event_indexing()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):
        # First validate the correct configuration, then change parameter one by one.
        target_size = TYPE_INT(np.random.randint(1, 2))
        context_size = TYPE_INT(2 * np.random.randint(1, 2))
        negative_sample_size = TYPE_INT(np.random.randint(1, 2))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(2, 3))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        embedding, event_context = _must_succeed(
            name=name,
            num_nodes=(1+negative_sample_size),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
            msg="must succeed"
        )

        # --------------------------------------------------------------------------------
        # Run methods and save the results to compare later
        # --------------------------------------------------------------------------------
        sequences = dictionary.function(sentences)
        target_context_pairs = event_context.function(sequences)
        Y1 = embedding.function(target_context_pairs)

        # --------------------------------------------------------------------------------
        # Save the layer state and invalidate the state variables
        # --------------------------------------------------------------------------------
        tester = testing.layer.Function(instance=embedding)
        tester.test_save()
        backup_W = copy.deepcopy(embedding.W)
        embedding._W = np.empty(())

        # --------------------------------------------------------------------------------
        # Confirm the layer does not function anymore
        # --------------------------------------------------------------------------------
        try:
            embedding.function(target_context_pairs)
            raise RuntimeError("Must fail with state deleted")
        except Exception as e:
            pass

        # --------------------------------------------------------------------------------
        # Restore the state and confirm the layer functions as expected.
        # Because of random negative sampling, the result of context part differs every time.
        # Hence only true label (target) part can be the same.
        # --------------------------------------------------------------------------------
        try:
            # Constraint:
            #   Layer works after state reloaded
            tester.test_load()
            assert np.allclose(backup_W, embedding.W)
            Y2 = embedding.function(target_context_pairs)
            assert \
                np.array_equal(Y1[::, 0], Y2[::,0]), \
                "Expected Y\n%s\nActual\n%s\ndiff\n%s\n" \
                % (Y1[::, 0], Y2[::,0], (Y1[::,0]-Y2[::,0]))

        except Exception as e:
            raise e
        finally:
            tester.clean()

    profiler.disable()
    profiler.print_stats(sort="cumtime")

