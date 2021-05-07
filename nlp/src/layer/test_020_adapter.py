import cProfile
import logging
import os
from typing import (
    Tuple,
    Callable,
)

import numpy as np

from common.constant import (
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_LABEL,
    TYPE_TENSOR
)
from common.function import (
    sigmoid_cross_entropy_log_loss,
    sigmoid
)
from layer.adapter import (
    Adapter
)
from layer.objective import (
    CrossEntropyLogLoss
)
from layer.preprocessing import (
    EventIndexing,
    EventContext
)
from layer.test_020_embedding import (
    _instantiate_event_indexing,
    _instantiate as _instantiate_embedding
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


def _instantiate(
        name: str = __name__,
        num_nodes: TYPE_INT = 1,        # number of outputs from the adapter layer
        target_size: TYPE_INT = 1,
        context_size: TYPE_INT = 4,
        negative_sample_size: TYPE_INT = 10,
        event_vector_size: TYPE_INT = 20,
        dictionary: EventIndexing = None,
        W: TYPE_TENSOR = None,
        log_level: int = logging.ERROR
):
    """
    1. Instantiate layers to test the Adapter.
    - Log loss
    - EventContext
    - Embedding
    - Adapter

    2. Adapt Embedding to Log loss via Adapter
    - Set log loss layer to the adapter function
    """
    assert dictionary is not None
    embedding: Embedding

    # --------------------------------------------------------------------------------
    # Log loss
    # --------------------------------------------------------------------------------
    loss = CrossEntropyLogLoss(
        name="embedding_loss",
        num_nodes=num_nodes,
        log_loss_function=sigmoid_cross_entropy_log_loss,
        log_level=logging.DEBUG
    )

    # --------------------------------------------------------------------------------
    # Embedding and EventContext layers
    # --------------------------------------------------------------------------------
    event_context: EventContext
    embedding, event_context = _instantiate_embedding(
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

    # --------------------------------------------------------------------------------
    # Adapter functions
    # --------------------------------------------------------------------------------
    adapter_function: Callable = embedding.adapt_function_to_logistic_log_loss(loss=loss)
    adapter_gradient: Callable = embedding.adapt_gradient_to_logistic_log_loss()

    # --------------------------------------------------------------------------------
    # Adapter layer
    # --------------------------------------------------------------------------------
    adapter: Adapter = Adapter(
        name=name,
        num_nodes=num_nodes,
        function=adapter_function,
        gradient=adapter_gradient,
        log_level=log_level
    )

    return loss, adapter, embedding, event_context


def _instantiate_must_fail(
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


def _instantiate_must_succeed(
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
) -> Tuple[CrossEntropyLogLoss, Adapter, Embedding, EventContext]:
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


def _function_must_fail(adapter: Adapter, Y: TYPE_TENSOR, msg: str = "must fail"):
    try:
        adapter.function(Y)
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _function_must_succeed(adapter: Adapter, Y: TYPE_TENSOR, msg: str = "must succeed"):
    try:
        return adapter.function(Y)
    except Exception as e:
        raise RuntimeError(msg) from e


def _gradient_must_fail(adapter: Adapter, dY: TYPE_TENSOR, msg: str = "must fail"):
    try:
        adapter.gradient(dY)
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _gradient_must_succeed(adapter: Adapter, dY: TYPE_TENSOR, msg: str = "must succeed"):
    try:
        return adapter.gradient(dY)
    except Exception as e:
        raise RuntimeError(msg) from e


def _update_must_fail(adapter: Adapter, msg: str = "must fail"):
    try:
        adapter.update()
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _update_must_succeed(adapter: Adapter, msg: str = "must succeed"):
    try:
        return adapter.update()
    except Exception as e:
        raise RuntimeError(msg) from e


def test_020_adapt_embedding_logistic_loss_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_020_adapt_embedding_logistic_loss_instantiation_to_fail"

    # First validate the correct configuration, then change parameter one by one.
    dictionary: EventIndexing = _instantiate_event_indexing()
    target_size = TYPE_INT(np.random.randint(1, 10))
    context_size = TYPE_INT(2 * np.random.randint(1, 10))
    negative_sample_size = TYPE_INT(np.random.randint(5, 20))
    event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 100))
    W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

    _instantiate(
        name=name,
        num_nodes=TYPE_INT(1),
        target_size=target_size,
        context_size=context_size,
        negative_sample_size=negative_sample_size,
        event_vector_size=event_vector_size,
        dictionary=dictionary,
        W=W,
        log_level=logging.DEBUG,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(NUM_MAX_TEST_TIMES):
        target_size = TYPE_INT(np.random.randint(1, 10))
        context_size = TYPE_INT(2 * np.random.randint(1, 10))
        negative_sample_size = TYPE_INT(np.random.randint(5, 20))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 100))

        msg = "Name is string with length > 0."
        _instantiate_must_fail(
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
        _instantiate_must_fail(
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

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_adapt_embedding_loss_adapter_function_to_fail(caplog):
    """
    Objective:
        Verify the Adapter function invalidate the shape other than
        - Y:(N, 1+SL)
        - ys:(N,SL)
        - ye:(N,1)
    Expected:
        Adapter.function(Y) to fail when Y shape is not (N, 1+SL)
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_adapt_embedding_logistic_loss_function_multi_lines"

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
        E = target_size = TYPE_INT(np.random.randint(1, 3))
        C = context_size = TYPE_INT(2 * np.random.randint(1, 5))
        SL = negative_sample_size = TYPE_INT(np.random.randint(1, 5))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 20))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        loss, adapter, embedding, event_context = _instantiate(
            name=name,
            num_nodes=TYPE_INT(1),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
        )

        sequences = dictionary.function(sentences)
        target_context_pairs = event_context.function(sequences)
        Y = embedding.function(target_context_pairs)
        N, _ = embedding.tensor_shape(Y)

        # ********************************************************************************
        # Constraint: Y in shape:(N,M) where M > SL+1 must fail
        # ********************************************************************************
        shape = (N, 1+SL+np.random.randint(1, 100))
        msg = "Y shape %s which is not the expected shape %s must fail" % \
              (shape, (N, SL+1))
        dummy_Y = np.random.uniform(size=shape).astype(TYPE_FLOAT)
        _function_must_fail(adapter=adapter, Y=dummy_Y, msg=msg)

        # ********************************************************************************
        # Constraint: Y in shape (N+,) must fail.
        # Adapter function can accept (N,) but not (N+,)
        # ********************************************************************************
        shape = (N + np.random.randint(1, 100),)
        msg = "Y shape %s which is not the expected shape %s must fail" % \
              (shape, (N,))
        dummy_Y = np.random.uniform(size=shape).astype(TYPE_FLOAT)
        _function_must_fail(adapter=adapter, Y=dummy_Y, msg=msg)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_adapt_embedding_loss_adapter_function_Y_to_succeed(caplog):
    """
    Objective:
        Verify the Adapter function handles Y in shape
        - Y:(N, 1+SL)
        - ys:(N,SL)
        - ye:(N,1)
    Expected:
        Adapter.function(Y) returns
        - For Y:(N, 1+SL), the return is in shape (N*(1+SL),1).
          Log loss T is set to the same shape

        - For Y:(N, SL), the return is in shape (N*SL,1).
          Log loss T is set to the same shape

        - For Y:(N,), the return is in shape (N,1).
          Log loss T is set to the same shape
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_adapt_embedding_logistic_loss_function_multi_lines"

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
        E = target_size = TYPE_INT(np.random.randint(1, 3))
        C = context_size = TYPE_INT(2 * np.random.randint(1, 5))
        SL = negative_sample_size = TYPE_INT(np.random.randint(1, 5))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 20))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        loss, adapter, embedding, event_context = _instantiate(
            name=name,
            num_nodes=TYPE_INT(1),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
        )

        sequences = dictionary.function(sentences)
        target_context_pairs = event_context.function(sequences)
        Y = embedding.function(target_context_pairs)
        N, _ = embedding.tensor_shape(Y)

        # ********************************************************************************
        # Constraint:
        # - Adapter function returns (N*(SL+1),1) with the same values of Y
        # - Adapter function has set T:(N*(SL+1),1) in the loss layer
        # ********************************************************************************
        msg = "Y must succeed"
        EZ = expected_Z = embedding.reshape(Y, shape=(N*(SL+1), 1))
        Z = _function_must_succeed(adapter=adapter, Y=Y, msg=msg)
        assert embedding.all_close(
            Z,
            EZ,
            "Z must close to EZ. Z:\n%s\nEZ\n%s\nDiff\n%s\n" % (Z, EZ, (EZ-Z))
        )
        T = np.zeros(shape=(N, (1+SL)), dtype=TYPE_LABEL)
        T[
            ::,
            0
        ] = TYPE_LABEL(1)
        T = embedding.reshape(T, shape=(-1, 1))
        assert embedding.all_equal(T, loss.T), \
            "Expected T must equals loss.T. Expected\n%s\nLoss.T\n%s\n" % (T, loss.T)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_adapt_embedding_loss_adapter_function_ys_to_succeed(caplog):
    """
    Objective:
        Verify the Adapter function handles Y in shape
        - Y:(N, 1+SL)
        - ys:(N,SL)
        - ye:(N,1)
    Expected:
        Adapter.function(Y) returns
        - For Y:(N, 1+SL), the return is in shape (N*(1+SL),1).
          Log loss T is set to the same shape

        - For Y:(N, SL), the return is in shape (N*SL,1).
          Log loss T is set to the same shape

        - For Y:(N,), the return is in shape (N,1).
          Log loss T is set to the same shape
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_adapt_embedding_logistic_loss_function_multi_lines"

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
        E = target_size = TYPE_INT(np.random.randint(1, 3))
        C = context_size = TYPE_INT(2 * np.random.randint(1, 5))
        SL = negative_sample_size = TYPE_INT(np.random.randint(1, 5))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 20))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        loss, adapter, embedding, event_context = _instantiate(
            name=name,
            num_nodes=TYPE_INT(1),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
        )

        sequences = dictionary.function(sentences)
        target_context_pairs = event_context.function(sequences)
        Y = embedding.function(target_context_pairs)
        N, _ = embedding.tensor_shape(Y)

        # ********************************************************************************
        # Constraint:
        # - Adapter function returns (N*(SL),1) with the same values of ys
        # - Adapter function has set T:(N*SL,1) with value 0 in the loss layer
        # ********************************************************************************
        msg = "ys must succeed"
        ys = Y[
            ::,
            1:
        ]
        EZ = expected_Z = embedding.reshape(ys, shape=(N*SL, 1))
        Z = _function_must_succeed(adapter=adapter, Y=ys, msg=msg)
        assert embedding.all_close(
            Z,
            EZ,
            "Z must close to EZ. Z:\n%s\nEZ\n%s\nDiff\n%s\n" % (Z, EZ, (EZ-Z))
        )
        T = np.zeros(shape=(N*SL, 1), dtype=TYPE_LABEL)
        assert embedding.all_equal(T, loss.T), \
            "Expected T must equals loss.T. Expected\n%s\nLoss.T\n%s\n" % (T, loss.T)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_adapt_embedding_loss_adapter_function_ye_to_succeed(caplog):
    """
    Objective:
        Verify the Adapter function handles Y in shape
        - Y:(N, 1+SL)
        - ys:(N,SL)
        - ye:(N,1)
    Expected:
        Adapter.function(Y) returns
        - For Y:(N, 1+SL), the return is in shape (N*(1+SL),1).
          Log loss T is set to the same shape

        - For Y:(N, SL), the return is in shape (N*SL,1).
          Log loss T is set to the same shape

        - For Y:(N,), the return is in shape (N,1).
          Log loss T is set to the same shape
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_adapt_embedding_logistic_loss_function_multi_lines"

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
        E = target_size = TYPE_INT(np.random.randint(1, 3))
        C = context_size = TYPE_INT(2 * np.random.randint(1, 5))
        SL = negative_sample_size = TYPE_INT(np.random.randint(1, 5))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 20))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        loss, adapter, embedding, event_context = _instantiate(
            name=name,
            num_nodes=TYPE_INT(1),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
        )

        sequences = dictionary.function(sentences)
        target_context_pairs = event_context.function(sequences)
        Y = embedding.function(target_context_pairs)
        N, _ = embedding.tensor_shape(Y)

        # ********************************************************************************
        # Constraint:
        # - Adapter function returns (N,1) with the same values of ye
        # - Adapter function has set T:(N,1) with value 1 in the loss layer
        # ********************************************************************************
        msg = "ye must succeed"
        ye = Y[
            ::,
            0
        ]
        EZ = expected_Z = embedding.reshape(ye, shape=(N, 1))
        Z = _function_must_succeed(adapter=adapter, Y=ye, msg=msg)
        assert embedding.all_close(
            Z,
            EZ,
            "Z must close to EZ. Z:\n%s\nEZ\n%s\nDiff\n%s\n" % (Z, EZ, (EZ-Z))
        )
        T = np.ones(shape=(N, 1), dtype=TYPE_LABEL)
        assert embedding.all_equal(T, loss.T), \
            "Expected T must equals loss.T. Expected\n%s\nLoss.T\n%s\n" % (T, loss.T)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_adapt_embedding_loss_adapter_gradient_to_succeed(caplog):
    """
    Objective:
        Verify the Adapter gradient method handles dY in shape (N, 1+SL)

        Adapter.function(Y) returns
        - For Y:(N, 1+SL), the return is in shape (N*(1+SL),1).
          Log loss T is set to the same shape

    Expected:
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_adapt_embedding_logistic_loss_function_multi_lines"

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
        E = target_size = TYPE_INT(np.random.randint(1, 3))
        C = context_size = TYPE_INT(2 * np.random.randint(1, 5))
        SL = negative_sample_size = TYPE_INT(np.random.randint(1, 5))
        event_vector_size: TYPE_INT = TYPE_INT(np.random.randint(5, 20))
        W: TYPE_TENSOR = np.random.randn(dictionary.vocabulary_size, event_vector_size)

        loss, adapter, embedding, event_context = _instantiate(
            name=name,
            num_nodes=TYPE_INT(1),
            target_size=target_size,
            context_size=context_size,
            negative_sample_size=negative_sample_size,
            event_vector_size=event_vector_size,
            dictionary=dictionary,
            W=W,
            log_level=logging.DEBUG,
        )

        # ================================================================================
        # Forward path
        # ================================================================================
        # --------------------------------------------------------------------------------
        # Event indexing
        # --------------------------------------------------------------------------------
        sequences = dictionary.function(sentences)

        # --------------------------------------------------------------------------------
        # Event context pairs
        # --------------------------------------------------------------------------------
        target_context_pairs = event_context.function(sequences)

        # --------------------------------------------------------------------------------
        # Embedding
        # --------------------------------------------------------------------------------
        Y = embedding.function(target_context_pairs)
        N, _ = embedding.tensor_shape(Y)
        batch_size = TYPE_FLOAT(N*(1+SL))

        # --------------------------------------------------------------------------------
        # Adapter
        # --------------------------------------------------------------------------------
        Z = adapter.function(Y)

        # --------------------------------------------------------------------------------
        # Loss
        # --------------------------------------------------------------------------------
        L = loss.function(Z)

        # ********************************************************************************
        # Constraint:
        #   loss.T is set to the T by adapter.function()
        # ********************************************************************************
        T = np.zeros(shape=(N, (1+SL)), dtype=TYPE_LABEL)
        T[
            ::,
            0
        ] = TYPE_LABEL(1)
        assert embedding.all_equal(T.reshape(-1, 1), loss.T), \
            "Expected T must equals loss.T. Expected\n%s\nLoss.T\n%s\n" % (T, loss.T)

        # ********************************************************************************
        # Constraint:
        #   Expected loss is sum(sigmoid_cross_entropy_log_loss(Y, T)) / (N*(1+SL))
        #   The batch size for the Log Loss is (N*(1+SL))
        # ********************************************************************************
        EJ, EP = sigmoid_cross_entropy_log_loss(
            X=Z,
            T=T.reshape(-1, 1)
        )
        EL = np.sum(EJ, dtype=TYPE_FLOAT) / batch_size

        assert embedding.all_close(EL, L), \
            "Expected EL=L but EL=\n%s\nL=\n%s\nDiff=\n%s\n" % (EL, L, (EL-L))

        # ================================================================================
        # Backward path
        # ================================================================================
        # ********************************************************************************
        # Constraint:
        #   Expected dL/dY from the Log Loss is (P-T)/N
        # ********************************************************************************
        EDY = (sigmoid(Y) - T.astype(TYPE_FLOAT)) / batch_size
        assert EDY.shape == Y.shape

        dY = adapter.gradient(loss.gradient(TYPE_FLOAT(1)))
        assert dY.shape == Y.shape
        assert embedding.all_close(EDY, dY), \
            "Expected EDY==dY. EDY=\n%s\nDiff\n%s\n" % (EDY, (EDY-dY))

    profiler.disable()
    profiler.print_stats(sort="cumtime")
