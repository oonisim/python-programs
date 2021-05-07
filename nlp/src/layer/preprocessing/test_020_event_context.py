import cProfile
import logging

import numpy as np
from common.constant import (
    TYPE_INT,
)
from common.utility import (
    random_string
)
from layer.constants import (
    _NAME,
    _NUM_NODES
)
from layer.preprocessing import (
    EventContext
)
from testing.config import (
    NUM_MAX_TEST_TIMES
)

Logger = logging.getLogger(__name__)


def _instantiate(name: str, num_nodes: TYPE_INT, window_size: TYPE_INT, event_size: TYPE_INT,):
    if np.random.uniform() > 0.5:
        event_context = EventContext.build({
            _NAME: name,
            _NUM_NODES: num_nodes,
            "window_size": window_size,
            "event_size": event_size
        })
    else:
        event_context = EventContext(name, num_nodes=num_nodes, window_size=window_size, event_size=event_size)

    return event_context


def _must_fail(name: str, num_nodes: TYPE_INT, window_size: TYPE_INT, event_size: TYPE_INT, msg: str):
    try:
        _instantiate(name, num_nodes=num_nodes, window_size=window_size, event_size=event_size)
        raise RuntimeError(msg)
    except AssertionError:
        pass


def _must_succeed(name: str, num_nodes: TYPE_INT, window_size: TYPE_INT, event_size: TYPE_INT, msg: str):
    try:
        return _instantiate(name, num_nodes=num_nodes, window_size=window_size, event_size=event_size)
    except Exception as e:
        raise RuntimeError(msg)


def test_020_event_context_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_020_event_context_instantiation_to_fail"
    stride: TYPE_INT = np.random.randint(1, 100)
    event_size: TYPE_INT = np.random.randint(1, 100)
    window_size: TYPE_INT = 2 * stride + event_size

    event_context = _must_succeed(
        name=name, num_nodes=1, window_size=window_size, event_size=event_size, msg="must scceed"
    )

    for _ in range(NUM_MAX_TEST_TIMES):
        msg = "Name is string with length > 0."
        _must_fail(name="", num_nodes=1, window_size=window_size, event_size=event_size, msg=msg)

        msg = "window_size = 2 * stride + event_size"
        _must_fail(name=name, num_nodes=1, window_size=0, event_size=event_size, msg=msg)
        _must_fail(name=name, num_nodes=1, window_size=event_size, event_size=event_size, msg=msg)
        _must_fail(name=name, num_nodes=1, window_size=2 * stride, event_size=3, msg=msg)

        msg = "event_size = (window_size - event_size) / 2"
        _must_fail(name=name, num_nodes=1, window_size=window_size, event_size=0, msg=msg)
        _must_fail(name=name, num_nodes=1, window_size=window_size, event_size=window_size + stride, msg=msg)


def test_020_event_context_instance_properties(caplog):
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_event_context_instance_properties"
    msg = "Accessing uninitialized property of the layer must fail."

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(NUM_MAX_TEST_TIMES):
        stride: TYPE_INT = np.random.randint(1, 100)
        event_size: TYPE_INT = np.random.randint(1, 100)
        window_size: TYPE_INT = 2 * stride + event_size

        name = random_string(np.random.randint(1, 10))
        event_context = _must_succeed(
            name=name, num_nodes=TYPE_INT(1), window_size=window_size, event_size=event_size, msg=msg
        )

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not event_context.name == name:
                raise RuntimeError("event_context.name == name should be true")
        except AssertionError as e:
            raise RuntimeError("Access to name should be allowed as already initialized.") from e

        try:
            if not isinstance(event_context.logger, logging.Logger):
                raise RuntimeError("isinstance(event_context.logger, logging.Logger) should be true")
        except AssertionError as e:
            raise RuntimeError("Access to logger should be allowed as already initialized.") from e

        assert event_context.window_size == window_size
        assert event_context.event_size == event_size

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_event_context_function_event_size_1(caplog):
    """
    Objective:
        Verify the layer function with event_size =1
    Expected:
        function generates expected event_context pairs
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_event_context_instance_properties"
    msg = "Accessing uninitialized property of the layer must fail."

    profiler = cProfile.Profile()
    profiler.enable()

    stride: TYPE_INT = 2
    event_size: TYPE_INT = 1
    window_size: TYPE_INT = 2 * stride + event_size

    event_context = _must_succeed(
        name=name, num_nodes=1, window_size=window_size, event_size=event_size, msg=msg
    )

    # 0, 1, 2, 3, 4
    # expected event_context pair: [[2, 0, 1, 3, 4]]
    X1 = np.arange(5)
    expected1 = np.array([[2, 0, 1, 3, 4]])
    Y1 = event_context.function(X1)
    assert \
        np.array_equal(Y1, expected1), \
        "Expected\n%s\nActual\n%s\n" % (expected1, Y1)

    # 0, 1, 2, 3, 4, 5, 6
    # expected event_context pair: [
    #   [2, 0, 1, 3, 4],
    #   [3, 1, 2, 4, 5],
    #   [4, 2, 3, 5, 6]
    # ]
    X2 = np.arange(7)
    expected2 = np.array([
        [2, 0, 1, 3, 4],
        [3, 1, 2, 4, 5],
        [4, 2, 3, 5, 6]
    ])
    Y2 = event_context.function(X2)
    assert np.array_equal(Y2, expected2), \
        "Expected\n%s\nActual\n%s\n" % (expected2, Y2)

    # Each row will create 2D matrix [[...]] always.
    X3 = np.array([
        [2, 0, 1, 3, 4],    # -> [[...]]
        [3, 1, 2, 4, 5],
        [4, 2, 3, 5, 6]
    ])
    # (event, context) where 'event' is UNK or NIL will be omitted, and
    # the result Y=(event,context) shape is (N,M). See the explanation
    # in EventContext.function()
    expected3 = np.array([
        # [1, 2, 0, 3, 4]],
        [2, 3, 1, 4, 5],
        [3, 4, 2, 5, 6]
    ])
    Y3 = event_context.function(X3)
    assert np.array_equal(Y3, expected3), \
        "Expected\n%s\nActual\n%s\n" % (expected3, Y3)

    # Each row will create 2D matrix [[...]] always.
    X4 = np.array([
        [0, 1, 2, 3, 4, 5],    # -> [[...], [...]]
        [6, 7, 8, 9, 10, 11]
    ])
    expected4 = np.array([
        [2, 0, 1, 3, 4],
        [3, 1, 2, 4, 5],
        [8, 6, 7, 9, 10],
        [9, 7, 8, 10, 11]
    ])
    Y4 = event_context.function(X4)
    assert np.array_equal(Y4, expected4), \
        "Expected\n%s\nActual\n%s\n" % (expected4, Y4)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_event_context_function_event_size_2(caplog):
    """
    Objective:
        Verify the layer function with event_size = 2
    Expected:
        function generates expected event_context pairs
    """
    caplog.set_level(logging.DEBUG)
    name = "test_020_event_context_instance_properties"
    msg = "Accessing uninitialized property of the layer must fail."

    profiler = cProfile.Profile()
    profiler.enable()

    stride: TYPE_INT = 2
    event_size: TYPE_INT = 2
    window_size: TYPE_INT = 2 * stride + event_size
    num_nodes = TYPE_INT(1)

    event_context = _must_succeed(
        name=name, num_nodes=num_nodes, window_size=window_size, event_size=event_size, msg=msg
    )

    # 0, 1, 2, 3, 4, 5
    # expected event_context pair: [
    #   [2, 3, 0, 1, 4, 5],
    X1 = np.arange(6)
    expected1 = np.array([
        [2, 3, 0, 1, 4, 5]
    ])
    Y1 = event_context.function(X1)
    assert \
        np.array_equal(Y1, expected1), \
        "Expected\n%s\nActual\n%s\n" % (expected1, Y1)

    # 0, 1, 2, 3, 4, 5, 6
    # expected event_context pair: [
    #   [2, 3, 0, 1, 4, 5],
    X2 = np.arange(7)
    expected2 = np.array([
        [2, 3, 0, 1, 4, 5],
        [3, 4, 1, 2, 5, 6]
    ])
    Y2 = event_context.function(X2)
    assert np.array_equal(Y2, expected2), \
        "Expected\n%s\nActual\n%s\n" % (expected2, Y2)

    profiler.disable()
    profiler.print_stats(sort="cumtime")
