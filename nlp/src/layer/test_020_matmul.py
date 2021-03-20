"""Matmul layer test cases
Batch X: shape(N, D):
--------------------
X is the input data into a Matmul layer, hence it does NOT include the bias.

Gradient dL/dX: shape(N, D)
--------------------
Same shape of X because L is scalar.

Weights W: shape(M, D+1)
--------------------
W includes the bias weight because we need to control the weight initializations
including the bias weight.

Gradient dL/dW: shape(M, D+1)
--------------------
Same shape with W.
"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple
)
import cProfile
import copy
import logging
import numpy as np
from common import (
    numerical_jacobian,
    weights,
    random_string
)
from layer import (
    Matmul
)
from common.test_config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES,
    GRADIENT_DIFF_ACCEPTANCE_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO
)


Logger = logging.getLogger("test_030_objective")
Logger.setLevel(logging.DEBUG)


def test_020_matmul_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_020_matmul_instantiation_to_fail"
    for _ in range(NUM_MAX_TEST_TIMES):
        M: int = np.random.randint(1, NUM_MAX_NODES)
        D = 1
        # Constraint: Name is string with length > 0.
        try:
            Matmul(
                name="",
                num_nodes=1,
                W=weights.xavier(M, D+1)
            )
            raise RuntimeError("Matmul initialization with invalid name must fail")
        except AssertionError:
            pass

        # Constraint: num_nodes > 1
        try:
            Matmul(
                name="test_020_matmul",
                num_nodes=0,
                W=weights.xavier(M, D+1)
            )
            raise RuntimeError("Matmul(num_nodes<1) must fail.")
        except AssertionError:
            pass

        # Constraint: logging level is correct.
        try:
            Matmul(
                name="test_020_matmul",
                num_nodes=M,
                W=weights.xavier(M, D+1),
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
                W=weights.xavier(2, D+1)
            )
            raise RuntimeError("Matmul initialization with invalid name must fail")
        except AssertionError:
            pass


def test_020_matmul_instance_properties():
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    msg = "Accessing uninitialized property of the layer must fail."

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(1, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)
        layer = Matmul(
            name=name,
            num_nodes=M,
            W=weights.uniform(M, D+1),
            log_level=logging.DEBUG
        )

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not layer.name == name: raise RuntimeError("layer.name == name should be true")
        except AssertionError:
            raise RuntimeError("Access to name should be allowed as already initialized.")

        try:
            if not layer.M == M: raise RuntimeError("layer.M == M should be true")
        except AssertionError:
            raise RuntimeError("Access to M should be allowed as already initialized.")

        try:
            if not isinstance(layer.logger, logging.Logger):
                raise RuntimeError("isinstance(layer.logger, logging.Logger) should be true")
        except AssertionError:
            raise RuntimeError("Access to logger should be allowed as already initialized.")

        try:
            a = layer.D
        except AssertionError:
            raise RuntimeError("Access to D should be allowed as already initialized.")

        try:
            layer.W is not None
        except AssertionError:
            raise RuntimeError("Access to W should be allowed as already initialized.")
            pass

        try:
            layer.optimizer is not None
        except AssertionError:
            raise RuntimeError("Access to optimizer should be allowed as already initialized.")

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
            print(layer.dX)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(layer.dW)
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
            layer = Matmul(
                name=name,
                num_nodes=M,
                W=weights.xavier(M, D+1),
                log_level=logging.DEBUG
            )
            layer.function(int(1))
            raise RuntimeError("Invoke layer.function(int(1)) must fail.")
        except AssertionError:
            pass

        try:
            layer = Matmul(
                name=name,
                num_nodes=M,
                W=weights.xavier(M, D+1),
                log_level=logging.DEBUG
            )
            layer.function(int(1))
            layer.gradient(int(1))
            raise RuntimeError("Invoke layer.gradient(int(1)) must fail.")
        except AssertionError:
            pass


def test_020_matmul_instantiation():
    """
    Objective:
        Verify the initialized layer instance provides its properties.
    Expected:
        * name, num_nodes, M, log_level are the same as initialized.
        * X, T, dX, objective returns what is set.
        * N, M property are provided after X is set.
        * Y, dY properties are provided after they are set.
    """
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function"""
        return np.sum(X)

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(1, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)
        name = "test_020_matmul_instantiation"
        layer = Matmul(
            name=name,
            num_nodes=M,
            W=weights.he(M, D+1),
            log_level=logging.DEBUG
        )
        layer.objective = objective

        assert layer.name == name
        assert layer.num_nodes == layer.M == M

        layer._D = D
        assert layer.D == D

        X = np.random.randn(N, D)
        layer.X = X
        assert np.array_equal(layer.X, X)
        assert layer.N == N == X.shape[0]

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


def test_020_matmul_round_trip():
    """
    Objective:
        Verify the forward and backward paths at matmul.

    Expected:
        Forward path:
        1. Matmul function(X) == X @ W.T
        2. Numerical gradient should be the same with numerical Jacobian

        Backward path:
        3. Analytical gradient dL/dX == dY @ W
        4. Analytical dL/dW == X.T @ dY
        5. Analytical gradients are similar to the numerical gradient ones

        Gradient descent
        6. W is updated via the gradient descent.
        7. Objective L is decreasing via the gradient descent.

    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        # --------------------------------------------------------------------------------
        # Instantiate a Matmul layer
        # --------------------------------------------------------------------------------
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(1, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)
        W = weights.he(M, D+1)
        name = "test_020_matmul_methods"

        def objective(X: np.ndarray) -> Union[float, np.ndarray]:
            """Dummy objective function to calculate the loss L"""
            return np.sum(X)

        layer = Matmul(
            name=name,
            num_nodes=M,
            W=W,
            log_level=logging.DEBUG
        )
        layer.objective = objective

        # ================================================================================
        # Layer forward path
        # Calculate the layer output Y=f(X), and get the loss L = objective(Y)
        # Test the numerical gradient dL/dX=layer.gradient_numerical().
        #
        # Note that bias columns are added inside the matmul layer instance, hence
        # layer.X.shape is (N, 1+D), layer.W.shape is (M, 1+D)
        # ================================================================================
        X = np.random.randn(N, D)
        Logger.debug("%s: X is \n%s", name, X)

        Y = layer.function(X)
        L = layer.objective(Y)

        # Constraint 1 : Matmul outputs Y should be X@W.T
        assert np.array_equal(Y, np.matmul(layer.X, layer.W.T))

        # Constraint 2: Numerical gradient should be the same with numerical Jacobian
        GN = layer.gradient_numerical()         # [dL/dX, dL/dW]

        # DO NOT use layer.function() as the objective function for numerical_jacobian().
        # The state of the layer will be modified.
        # LX = lambda x: layer.objective(layer.function(x))
        def LX(x):
            y = np.matmul(x, layer.W.T)
            return layer.objective(y)

        EGNX = numerical_jacobian(LX, layer.X)          # Numerical dL/dX including bias
        EGNX = EGNX[::, 1::]                            # Remove bias for dL/dX
        assert np.array_equal(GN[0], EGNX), \
            "GN[0]\n%s\nEGNX=\n%s\n" % (GN[0], EGNX)

        # DO NOT use layer.function() as the objective function for numerical_jacobian().
        # The state of the layer will be modified.
        # LW = lambda w: layer.objective(np.matmul(X, w.T))
        def LW(w):
            Y = np.matmul(layer.X, w.T)
            return layer.objective(Y)

        EGNW = numerical_jacobian(LW, layer.W)          # Numerical dL/dW including bias
        assert np.array_equal(GN[1], EGNW)              # No need to remove bias

        # ================================================================================
        # Layer backward path
        # Calculate the analytical gradient dL/dX=layer.gradient(dL/dY) with a dummy dL/dY.
        # ================================================================================
        dY = np.ones_like(Y)
        dX = layer.gradient(dY)

        # Constraint 3: Matmul gradient dL/dX should be dL/dY @ W. Use a dummy dL/dY = 1.0.
        expected_dX = np.matmul(dY, layer.W)
        expected_dX = expected_dX[
            ::,
            1::     # Omit bias
        ]
        assert np.array_equal(dX, expected_dX)

        # Constraint 5: Analytical gradient dL/dX close to the numerical gradient GN.
        assert np.all(np.abs(dX - GN[0]) < GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            f"dX need close to GN[0] but dX \n%s\n GN[0] \n%s\n" % (dX, GN[0])

        # --------------------------------------------------------------------------------
        # Gradient update.
        # Run the gradient descent to update Wn+1 = Wn - lr * dL/dX.
        # --------------------------------------------------------------------------------
        # Python passes the reference to W, hence it is directly updated by the gradient-
        # descent to avoid a temporary copy. Backup W before to compare before/after.
        backup = copy.deepcopy(W)

        # Gradient descent and returns analytical dL/dX, dL/dW
        dS = layer.update()

        # Constraint 6.: W has been updated by the gradient descent.
        assert np.any(backup != layer.W), "W has not been updated "

        # Constraint 5: the numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
        assert np.allclose(
            dS[0],
            GN[0],
            atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
            rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
        ), "dS[0]=\n%s\nGN[0]=\n%sdiff=\n%s\n" % (dS[0], GN[0], (dS[0]-GN[0]))
        assert np.allclose(
            dS[1],
            GN[1],
            atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
            rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
        ), "dS[1]=\n%s\nGN[1]=\n%sdiff=\n%s\n" % (dS[1], GN[1], (dS[1]-GN[1]))

        # Constraint 7: gradient descent progressing with the new objective L(Yn+1) < L(Yn)
        assert np.all(np.abs(objective(layer.function(X)) < L))

    profiler.disable()
    profiler.print_stats(sort="cumtime")
