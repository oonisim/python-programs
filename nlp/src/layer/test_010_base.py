"""Network base layer test cases"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple
)
import logging
import numpy as np
from layer import (
    Layer
)

from common.test_config import (
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES
)


# ================================================================================
# Base layer
# ================================================================================
def test_010_base_instantiation_to_fail():
    """Test case for layer base class instantiation with wrong parameters.
    """
    # --------------------------------------------------------------------------------
    # Test initialization validation logic:
    # Expected the layer instance initialization fails.
    # --------------------------------------------------------------------------------
    M: int = np.random.randint(1, NUM_MAX_NODES)
    # Layer instance creation fails due to the invalid name.
    try:
        Layer(
            name="",
            num_nodes=1
        )
        raise RuntimeError("Layer initialization with invalid name must fail")
    except AssertionError as e:
        pass

    # Layer instance creation fails due to num_nodes = M < 1
    try:
        Layer(
            name="test_010_base",
            num_nodes=0
        )
        raise RuntimeError("Layer(num_nodes<1) must fail.")
    except AssertionError as e:
        pass

    # Layer instance creation fails due to the invalid log level.
    try:
        Layer(
            name="test_010_base",
            num_nodes=M,
            log_level=-1
        )
        raise RuntimeError("Layer initialization with invalid log level must fail")
    except (AssertionError, KeyError) as e:
        pass

    # Layer instance creation fails as X.shape[0] != T.shape[0]
    # X can be set later than T, hence this cannot be tested.
    # try:
    #     N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
    #     M: int = np.random.randint(1, NUM_MAX_NODES)
    #     layer: Layer = Layer(
    #         name="test_010_base",
    #         num_nodes=M,
    #         log_level=logging.DEBUG
    #     )
    #     X = np.random.randn(N, M)
    #     layer.X = X
    #     T = np.random.randint(0, M, N+1)
    #     layer.T = T
    #     raise RuntimeError("Layer initialization different batch size between X and T must fail")
    # except AssertionError as e:
    #     pass


def test_010_base_instance_properties():
    """Test for the base class validates non initialized properties"""
    msg = "Accessing uninitialized property of the layer must fail."
    M: int = np.random.randint(1, NUM_MAX_NODES)
    name = "test_010_base"
    layer = Layer(
        name=name,
        num_nodes=M,
        log_level=logging.DEBUG
    )
    try:
        print(layer.D)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        print(layer.X)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        layer.X = int(1)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        print(layer.dX)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        print(layer.Y)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass
    try:
        layer._Y = int(1)
        print(layer.Y)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        print(layer.dY)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass
    try:
        layer._dY = int(1)
        print(layer.dY)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        print(layer.T)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        layer.T = float(1)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        layer.objective(np.array(1.0))
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    try:
        print(layer.N)
        raise RuntimeError(msg)
    except AssertionError as e:
        pass

    assert layer.name == name
    assert layer.num_nodes == M


def test_010_base_instantiation():
    """Test case for layer base class
    """
    # --------------------------------------------------------------------------------
    # name, num_nodes, log_level _init_ properties.
    # Logging debug outputs.
    # X setter/getter
    # T setter/getter
    # objective function setter/getter
    # function(x) repeats x.
    # gradient(dL/dY) repeats dL/dY,
    # gradient_numerical() returns 1
    # --------------------------------------------------------------------------------
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function"""
        return np.sum(X)

    N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
    M: int = np.random.randint(1, NUM_MAX_NODES)
    name = "test_010_base"
    layer: Layer = Layer(
        name=name,
        num_nodes=M,
        log_level=logging.DEBUG
    )

    # --------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------
    assert layer.name == name
    assert layer.num_nodes == layer.M == M

    layer._D = 1
    assert layer.D == 1

    X = np.random.randn(N, M)
    layer.X = X
    assert np.array_equal(layer.X, X)
    assert layer.N == N

    layer._dX = X
    assert np.array_equal(layer.dX, X)

    T = np.random.randint(0, M, N)
    layer.T = T
    assert np.array_equal(layer.T, T)

    layer._Y = np.dot(X, X.T)
    assert np.array_equal(layer.Y, np.dot(X, X.T))

    layer._dY = np.array(0.9)
    assert layer._dY == np.array(0.9)

    layer.logger.debug("This is a pytest")

    # --------------------------------------------------------------------------------
    # Methods
    # --------------------------------------------------------------------------------
    try:
        layer.function(int(1))
        raise RuntimeError("Invoke layer.function(int(1)) must fail.")
    except AssertionError as e:
        pass

    x = np.array(1.0)
    assert np.array_equal(layer.function(x), x)

    try:
        layer.gradient(int(1))
        raise RuntimeError("Invoke layer.gradient(int(1)) must fail.")
    except AssertionError as e:
        pass

    dY = np.array([1.0, 2.0])
    assert np.array_equal(layer.gradient(dY), dY)

    layer.objective = objective
    assert np.array_equal(layer.objective(X), objective(X))
