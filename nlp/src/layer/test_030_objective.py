"""Objective (loss) layer test cases"""
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
    random_string
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


def test_030_objective_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    M: int = np.random.randint(1, NUM_MAX_NODES)
    # Constraint: Name is string with length > 0.
    try:
        SoftmaxWithLogLoss(
            name="",
            num_nodes=1
        )
        raise RuntimeError("SoftmaxWithLogLoss initialization with invalid name must fail")
    except AssertionError:
        pass

    # Constraint: num_nodes > 1
    try:
        SoftmaxWithLogLoss(
            name="test_030_objective",
            num_nodes=0
        )
        raise RuntimeError("SoftmaxWithLogLoss(num_nodes<1) must fail.")
    except AssertionError:
        pass

    # Constraint: logging level is correct.
    try:
        SoftmaxWithLogLoss(
            name="test_030_objective",
            num_nodes=M,
            log_level=-1
        )
        raise RuntimeError("SoftmaxWithLogLoss initialization with invalid log level must fail")
    except (AssertionError, KeyError):
        pass


def test_030_objective_instance_properties():
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    msg = "Accessing uninitialized property of the layer must fail."
    name = random_string(np.random.randint(1, 10))
    M: int = np.random.randint(1, NUM_MAX_NODES)
    layer = SoftmaxWithLogLoss(
        name=name,
        num_nodes=M,
        log_level=logging.DEBUG
    )

    # --------------------------------------------------------------------------------
    # To pass
    # --------------------------------------------------------------------------------
    try:
        print(layer.name)
    except AssertionError as e:
        raise RuntimeError("Access to name should be allowed as already initialized.")

    try:
        print(layer.M)
    except AssertionError as e:
        raise RuntimeError("Access to M should be allowed as already initialized.")

    try:
        print(layer.logger)
    except AssertionError as e:
        raise RuntimeError("Access to logger should be allowed as already initialized.")

    assert layer.name == name
    assert layer.num_nodes == M

    # --------------------------------------------------------------------------------
    # To fail
    # --------------------------------------------------------------------------------
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
        print(layer.N)
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
        print(layer.P)          # P is an alias for Y
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
        print(layer.L)
        raise RuntimeError(msg)
    except AssertionError:
        pass

    try:
        print(layer.J)
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
    """
    Objective:
        Verify the initialized layer instance provides its properties.
    Expected:
        * name, num_nodes, M, log_level are the same as initialized.
        * X, T, dY, objective returns what is set.
        * N, M property are provided after X is set.
        * Y, P, L properties are provided after function(X).
        * gradient(dL/dY) repeats dL/dY,
        * gradient_numerical() returns 1
    """
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
    layer.objective = objective

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
    assert layer.M == X.shape[0]

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

    assert layer.objective == objective


def test_030_objective_methods_1d():
    """To be implemented"""
    pass


def test_030_objective_methods_2d_ohe():
    """
    Objective:
        Verify the forward path constraints:
        1. Layer output L/loss is np.sum(cross_entropy_log_loss(softmax(X), T)) / N.
        2. gradient_numerical() == numerical Jacobian numerical_jacobian(O, X).

        Verify the backward path constraints:
        1. Analytical gradient G: gradient() == (P-1)/N
        2. Analytical gradient G is close to GN: gradient_numerical().
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function to calculate the loss L"""
        assert X.ndim == 0, "The output of the log loss should be of shape ()"
        return X

    # --------------------------------------------------------------------------------
    # Instantiate a SoftmaxWithLogLoss layer
    # --------------------------------------------------------------------------------
    N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
    M: int = np.random.randint(1, NUM_MAX_NODES)
    name = "test_030_objective"

    layer = SoftmaxWithLogLoss(
        name=name,
        num_nodes=M,
        log_level=logging.DEBUG
    )
    layer.objective = objective

    # ================================================================================
    # Layer forward path
    # ================================================================================
    X = np.random.randn(N, M)
    T = np.zeros_like(X, dtype=int)     # OHE labels.
    T[
        np.arange(N),
        np.random.randint(0, M, N)
    ] = int(1)
    layer.T = T

    P = softmax(X)
    EG = (P - T) / N       # Expected analytical gradient dL/dX = (P-T)/N

    # --------------------------------------------------------------------------------
    # constraint: L/loss == np.sum(cross_entropy_log_loss(softmax(X), T)) / N.
    # --------------------------------------------------------------------------------
    L = layer.function(X)
    Z = np.array(np.sum(cross_entropy_log_loss(softmax(X), T))) / N
    assert np.array_equal(L, Z), f"SoftmaxLogLoss output should be {L} but {Z}."

    # --------------------------------------------------------------------------------
    # constraint: gradient_numerical() == numerical Jacobian numerical_jacobian(O, X)
    # Use a dummy layer for the objective function because using the "layer"
    # updates the X, Y which can interfere the independence of the layer.
    # --------------------------------------------------------------------------------
    dummy= SoftmaxWithLogLoss(
        name=name,
        num_nodes=M,
        log_level=logging.DEBUG
    )
    dummy.T = T
    dummy.objective = objective
    GN = layer.gradient_numerical()                     # [dL/dX] from the layer
    O = lambda x: layer.objective(dummy.function(x))    # Objective function
    EGN = numerical_jacobian(O, X)                      # Expected numerical dL/dX
    assert np.array_equal(GN[0], EGN)

    # ================================================================================
    # Layer backward path
    # ================================================================================
    # --------------------------------------------------------------------------------
    # constraint: Analytical gradient G: gradient() == (P-1)/N.
    # --------------------------------------------------------------------------------
    dY = float(1)
    G = layer.gradient(dY)
    assert np.all(np.abs(G-EG) <= 1e-6), \
        f"Layer gradient dL/dX \n{G} \nneeds to be \n{EG}."

    # --------------------------------------------------------------------------------
    # constraint: Analytical gradient G is close to GN: gradient_numerical().
    # --------------------------------------------------------------------------------
    assert \
        np.all(np.abs(G - GN[0]) <= np.abs(GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0])), \
        f"dX is \n{G}\nGN[0] is \n{GN[0]}\nRatio * GN[0] is \n{GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0]}.\n"
