"""Network matmul layer test cases"""
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
    OFFSET_FOR_DELTA,
    numerical_jacobian,
    weights
)
from layer import (
    Matmul
)
from test_config import (
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES
)


# ================================================================================
# Base layer
# ================================================================================
def test_020_matmul_instantiation_to_fail():
    """Test case for layer matmul class instantiation with wrong parameters.
    """
    # --------------------------------------------------------------------------------
    # Test initialization validation logic:
    # Expected the layer instance initialization fails.
    # --------------------------------------------------------------------------------
    M: int = np.random.randint(1, NUM_MAX_NODES)
    D = 1
    # Matmul instance creation fails due to the invalid name.
    try:
        Matmul(
            name="",
            num_nodes=1,
            W=weights.xavier(M, D)
        )
        raise RuntimeError("Matmul initialization with invalid name must fail")
    except AssertionError as e:
        pass

    # Matmul instance creation fails due to num_nodes = M < 1
    try:
        Matmul(
            name="test_020_matmul",
            num_nodes=0,
            W=weights.xavier(M, D)
        )
        raise RuntimeError("Matmul(num_nodes<1) must fail.")
    except AssertionError as e:
        pass

    # Matmul instance creation fails due to the invalid log level.
    try:
        Matmul(
            name="test_020_matmul",
            num_nodes=M,
            W=weights.xavier(M, D),
            log_level=-1
        )
        raise RuntimeError("Matmul initialization with invalid log level must fail")
    except (AssertionError, KeyError) as e:
        pass

    # Matmul instance creation fails as W.shape[1] != num_nodes
    try:
        Matmul(
            name="",
            num_nodes=1,
            W=weights.xavier(2, D)
        )
        raise RuntimeError("Matmul initialization with invalid name must fail")
    except AssertionError as e:
        pass

    # Matmul instance creation fails as X.shape[0] != T.shape[0]
    try:
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(1, NUM_MAX_NODES)
        layer: Matmul = Matmul(
            name="test_020_matmul",
            num_nodes=M,
            W=weights.xavier(M, D),
            log_level=logging.DEBUG
        )
        X = np.random.randn(N, M)
        layer.X = X
        T = np.random.randint(0, M, N+1)
        layer.T = T
        raise RuntimeError("Matmul initialization different batch size between X and T must fail")
    except AssertionError as e:
        pass


def test_020_matmul_instance_properties():
    """Test for the matmul class validates non initialized properties"""
    msg = "Accessing uninitialized property of the layer must fail."
    M: int = np.random.randint(1, NUM_MAX_NODES)
    D: int = np.random.randint(1, NUM_MAX_FEATURES)
    name = "test_020_matmul"
    layer = Matmul(
        name=name,
        num_nodes=M,
        W=weights.uniform(M, D),
        log_level=logging.DEBUG
    )

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

    try:
        layer.function(int(1))
        raise RuntimeError("Invoke layer.function(int(1)) must fail.")
    except AssertionError as e:
        pass

    try:
        layer.function(1.0)
        layer.gradient(int(1))
        raise RuntimeError("Invoke layer.gradient(int(1)) must fail.")
    except AssertionError as e:
        pass


def test_020_matmul_instantiation():
    """Test case for layer matmul class
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
    name = "test_020_matmul"
    layer = Matmul(
        name=name,
        num_nodes=M,
        W=weights.he(M, D),
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


def test_020_matmul_methods():
    """Test case for layer matmul class
    """
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function to calculate the loss L"""
        return np.sum(X)

    # --------------------------------------------------------------------------------
    # Instantiate a Matmul layer
    # --------------------------------------------------------------------------------
    N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
    M: int = np.random.randint(1, NUM_MAX_NODES)
    D: int = np.random.randint(1, NUM_MAX_FEATURES)
    W = weights.he(M, D)
    name = "test_020_matmul"

    layer = Matmul(
        name=name,
        num_nodes=M,
        W=W,
        log_level=logging.DEBUG
    )
    layer.objective = objective

    # --------------------------------------------------------------------------------
    # Layer forward path
    # Calculate the layer output Y=f(X), and get the loss L = objective(Y)
    # Test the numerical gradient dL/dX=layer.gradient_numerical().
    # --------------------------------------------------------------------------------
    X = np.random.randn(N, D)
    Y = layer.function(X)
    L = layer.objective(Y)
    # Matmul outputs Y should be X@W.T
    assert np.array_equal(Y, np.matmul(X, W.T))

    # Numerical gradient should be the same with numerical Jacobian
    GN = layer.gradient_numerical()         # [dL/dX, dL/dW]
    LX = lambda x: layer.objective(layer.function(x))
    JX = numerical_jacobian(LX, X)           # Numerical dL/dX
    assert np.array_equal(GN[0], JX)

    LW = lambda w: layer.objective(np.matmul(X, w.T))
    JW = numerical_jacobian(LW, W)           # Numerical dL/dX
    assert np.array_equal(GN[1], JW)

    # --------------------------------------------------------------------------------
    # Layer backward path
    # Calculate the analytical gradient dL/dX=layer.gradient(dL/dY) with a dummy dL/dY.
    # Confirm the numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
    # --------------------------------------------------------------------------------
    # Matmul gradient dL/dX should be dL/dY @ W. Use a dummy dL/dY = 1.0.
    dY = np.ones_like(Y)
    dX = layer.gradient(dY)
    expected_dX = np.matmul(dY, W)
    assert np.array_equal(dX, expected_dX)

    # Matmul gradient dL/dX should be close to the numerical gradient GN.
    assert np.all(np.abs(dX - GN[0]) < OFFSET_FOR_DELTA)

    # --------------------------------------------------------------------------------
    # Gradient update.
    # Run the gradient descent to update Wn+1 = Wn - lr * dL/dX.
    # Confirm the new objective L(Yn+1) < L(Yn) with the Wn+1.
    # Confirm W in the layer has been updated by the gradient descent.
    # --------------------------------------------------------------------------------

    # Note Python pass the reference to W, and W will be directly updated by
    # the gradient descent to avoid a temporary copy.
    # Hence need to backup W before being changed to compare before/after.
    backup = copy.deepcopy(W)

    dS = layer.update()         # Analytical dL/dX, dL/dW
    assert np.all(np.abs(dS[0] - GN[0]) < OFFSET_FOR_DELTA) # dL/dX
    assert np.all(np.abs(dS[1] - GN[1]) < OFFSET_FOR_DELTA) # dL/dW

    # Objective L with the updated W should be smaller than previous L
    assert np.all(np.abs(objective(layer.function(X)) < L))
    assert np.any(backup != layer.W), "W has not been updated "

