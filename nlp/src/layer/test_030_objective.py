"""Network objective layer test cases"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple
)
import copy
import logging
import numpy as np
from common import (
    cross_entropy_log_loss,
    softmax,
    numerical_jacobian,
)
from layer import (
    SoftmaxWithLogLoss
)
from common.test_config import (
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES,
    GRADIENT_DIFF_ACCEPTANCE_RATIO
)


# ================================================================================
# Base layer
# ================================================================================
def test_030_objective_instantiation_to_fail():
    """Test case for layer objective class instantiation with wrong parameters.
    """
    # --------------------------------------------------------------------------------
    # Test initialization validation logic:
    # Expected the layer instance initialization fails.
    # --------------------------------------------------------------------------------
    M: int = np.random.randint(1, NUM_MAX_NODES)
    # SoftmaxWithLogLoss instance creation fails due to the invalid name.
    try:
        SoftmaxWithLogLoss(
            name="",
            num_nodes=1
        )
        raise RuntimeError("SoftmaxWithLogLoss initialization with invalid name must fail")
    except AssertionError:
        pass

    # SoftmaxWithLogLoss instance creation fails due to num_nodes = M < 1
    try:
        SoftmaxWithLogLoss(
            name="test_030_objective",
            num_nodes=0
        )
        raise RuntimeError("SoftmaxWithLogLoss(num_nodes<1) must fail.")
    except AssertionError:
        pass

    # SoftmaxWithLogLoss instance creation fails due to the invalid log level.
    try:
        SoftmaxWithLogLoss(
            name="test_030_objective",
            num_nodes=M,
            log_level=-1
        )
        raise RuntimeError("SoftmaxWithLogLoss initialization with invalid log level must fail")
    except (AssertionError, KeyError):
        pass

    # SoftmaxWithLogLoss instance creation fails as W.shape[1] != num_nodes
    try:
        SoftmaxWithLogLoss(
            name="",
            num_nodes=1
        )
        raise RuntimeError("SoftmaxWithLogLoss initialization with invalid name must fail")
    except AssertionError:
        pass


def test_030_objective_instance_properties():
    """Test for the objective class validates non initialized properties"""
    msg = "Accessing uninitialized property of the layer must fail."
    M: int = np.random.randint(1, NUM_MAX_NODES)
    name = "test_030_objective"
    layer = SoftmaxWithLogLoss(
        name=name,
        num_nodes=M,
        log_level=logging.DEBUG
    )

    try:
        print(layer.X)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        layer.X = int(1)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        print(layer.dX)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        print(layer.Y)
        raise RuntimeError(msg)
    except AssertionError:
        pass
    try:
        layer._Y = int(1)
        print(layer.Y)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        print(layer.dY)
        raise RuntimeError(msg)
    except AssertionError:
        pass
    try:
        layer._dY = int(1)
        print(layer.dY)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        print(layer.T)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        layer.T = float(1)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        layer.objective(np.array(1.0))
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        print(layer.N)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    assert layer.name == name
    assert layer.num_nodes == M

    try:
        layer.function(int(1))
        raise RuntimeError("Invoke layer.function(int(1)) must fail.")
    except AssertionError:
        pass

    try:
        layer.function(1.0)
        layer.gradient(int(1))
        raise RuntimeError("Invoke layer.gradient(int(1)) must fail.")
    except AssertionError:
        pass


def test_030_objective_instantiation():
    """Test case for layer objective class
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
    D: int = np.random.randint(1, NUM_MAX_FEATURES)
    name = "test_030_objective"
    layer = SoftmaxWithLogLoss(
        name=name,
        num_nodes=M,
        log_level=logging.DEBUG
    )

    # --------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------
    assert layer.name == name
    assert layer.num_nodes == layer.M == M

    layer._D = D
    assert layer.D == D

    X = np.random.randn(N, D)
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


def test_030_objective_methods_1d():
    """To be implemented"""
    pass


def test_030_objective_methods_2d_ohe():
    """Test case for layer objective class with 2D input with OHE labels.
    """
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function to calculate the loss L"""
        assert X.ndim == 0, "The output of the log loss should be of shape ()"
        return X

    # --------------------------------------------------------------------------------
    # Instantiate a SoftmaxWithLogLoss layer
    # --------------------------------------------------------------------------------
    N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
    N = 2
    M: int = np.random.randint(1, NUM_MAX_NODES)
    M = 2
    name = "test_030_objective"

    layer = SoftmaxWithLogLoss(
        name=name,
        num_nodes=M,
        log_level=logging.DEBUG
    )
    layer.objective = objective

    # --------------------------------------------------------------------------------
    # Layer forward path
    # Calculate the layer output Y=f(X), and get the loss L = objective(Y)
    # Test the numerical gradient dL/dX=layer.gradient_numerical().
    # --------------------------------------------------------------------------------
    X = np.random.randn(N, M)
    T = np.zeros_like(X, dtype=int)     # OHE labels.
    T[
        np.arange(N),
        np.random.randint(0, M, N)
    ] = int(1)
    P = softmax(X)
    expected_dX = (P - T) / N

    layer.T = T
    Y = layer.function(X)
    L = layer.objective(Y)

    # Output of
    Z = np.array(np.sum(cross_entropy_log_loss(softmax(X), T))) / N
    # SoftmaxWithLogLoss outputs Y should be the same with L and Z
    assert np.array_equal(Y, L), f"SoftmaxLogLoss output should be {L} but {Y}."
    assert np.array_equal(Y, Z), f"SoftmaxLogLoss output should be {Z} but {Y}."

    # Numerical gradient should be the same with numerical Jacobian
    GN = layer.gradient_numerical()         # [dL/dX]
    LX = lambda x: layer.objective(layer.function(x))
    JX = numerical_jacobian(LX, X)           # Numerical dL/dX
    assert np.array_equal(GN[0], JX)

    # --------------------------------------------------------------------------------
    # Layer backward path
    # Calculate the analytical gradient dL/dX=layer.gradient(dL/dY=1)
    # Confirm the numerical gradient dL/dX is closer to the analytical one.
    # --------------------------------------------------------------------------------
    # SoftmaxWithLogLoss gradient dL/dX should be (P-T)/N.
    dY = float(1)
    dX = layer.gradient(dY)
    assert np.all(np.abs(dX-expected_dX) < 1e-6), \
        f"Layer gradient dL/dX \n{dX} \nneeds to be \n{expected_dX}."

    # SoftmaxWithLogLoss gradient dL/dX should be close to the numerical gradient GN.
    assert \
        np.all(np.abs(dX - GN[0]) < np.abs(GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0])), \
        f"dX is \n{dX}\nGN[0] is \n{GN[0]}\nRatio * GN[0] is \n{GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0]}.\n"
