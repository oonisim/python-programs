"""Objective (loss) layer test cases"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple
)
import logging
import numpy as np
from common import (
    logistic_log_loss,
    categorical_log_loss,
    cross_entropy_log_loss,
    sigmoid,
    numerical_jacobian,
    random_string,
    BOUNDARY_SIGMOID
)
from layer import (
    CrossEntropyLogLoss
)
from common.test_config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES,
    GRADIENT_DIFF_ACCEPTANCE_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO
)


Logger = logging.getLogger(__name__)


def test_040_objective_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    for _ in range(NUM_MAX_TEST_TIMES):
        M: int = np.random.randint(1, NUM_MAX_NODES)
        # Constraint: Name is string with length > 0.
        try:
            CrossEntropyLogLoss(
                name="",
                num_nodes=1,
                activation=sigmoid
            )
            raise RuntimeError("CrossEntropyLogLoss initialization with invalid name must fail")
        except AssertionError:
            pass

        # Constraint: num_nodes == 1
        try:
            CrossEntropyLogLoss(
                name="test_040_objective",
                num_nodes=0,
                activation=sigmoid
            )
            raise RuntimeError("CrossEntropyLogLoss(num_nodes<1) must fail.")
        except AssertionError:
            pass
        try:
            CrossEntropyLogLoss(
                name="test_040_objective",
                num_nodes=np.random.randint(2, NUM_MAX_NODES),
                activation=sigmoid
            )
            raise RuntimeError("CrossEntropyLogLoss(num_nodes>1) must fail.")
        except AssertionError:
            pass

        # Constraint: logging level is correct.
        try:
            CrossEntropyLogLoss(
                name="test_040_objective",
                num_nodes=M,
                activation=sigmoid,
                log_level=-1
            )
            raise RuntimeError("CrossEntropyLogLoss initialization with invalid log level must fail")
        except (AssertionError, KeyError):
            pass


def test_040_objective_instance_properties():
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    msg = "Accessing uninitialized property of the layer must fail."
    name = random_string(np.random.randint(1, 10))
    for _ in range(NUM_MAX_TEST_TIMES):
        M: int = 1
        layer = CrossEntropyLogLoss(
            name=name,
            num_nodes=1,
            activation=sigmoid,
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


def test_040_objective_instantiation():
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

    name = "test_040_objective_instantiation"
    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = 1
        # For sigmoid log loss layer, the number of features N in X is the same with node number.
        D: int = M
        layer = CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            activation=sigmoid,
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
        assert layer.N == N == X.shape[0]
        # For sigmoid log loss layer, the number of features N in X is the same with node number.
        assert layer.M == X.shape[1]

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


def test_040_objective_methods_1d_ohe():
    """
    Objective:
        Verify the forward path constraints:
        1. Layer output L/loss is np.sum(cross_entropy_log_loss(sigmoid(X), T, f=logistic_log_loss))) / N.
        2. gradient_numerical() == numerical Jacobian numerical_jacobian(O, X).

        Verify the backward path constraints:
        1. Analytical gradient G: gradient() == (P-1)/N
        2. Analytical gradient G is close to GN: gradient_numerical().
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
        
        For X.ndim > 0, the layer transform X into 2D so as to use the numpy tuple-
        like indexing:
        P[
            (0,3),
            (2,4)
        ]
        Hence, the shape of GN, G are 2D.
    """
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function to calculate the loss L"""
        assert X.ndim == 0, "The output of the log loss should be of shape ()"
        return X

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    name = "test_040_objective_methods_1d_ohe"
    N = 1

    for _ in range(NUM_MAX_TEST_TIMES):
        layer = CrossEntropyLogLoss(
            name=name,
            num_nodes=1,
            activation=sigmoid,
            log_level=logging.DEBUG
        )
        layer.objective = objective

        # ================================================================================
        # Layer forward path
        # ================================================================================
        X = np.random.uniform(low=-BOUNDARY_SIGMOID, high=BOUNDARY_SIGMOID)
        T = np.random.randint(0, 2)                # OHE labels.
        T = 0
        layer.T = T

        # Expected analytical gradient dL/dX = (P-T)/N of shape (N,M)
        A = sigmoid(X)
        EG = ((A - T) / N).reshape(1, -1)

        Logger.debug("%s: X is \n%s\nT is %s\nP is %s\nEG is %s\n", name, X, T, A, EG)

        # --------------------------------------------------------------------------------
        # constraint: L/loss == np.sum(cross_entropy_log_loss(sigmoid(X), T, f=logistic_log_loss))) / N.
        # --------------------------------------------------------------------------------
        L = layer.function(X)       # L is shape ()
        Z = np.array(np.sum(cross_entropy_log_loss(P=sigmoid(X), T=T, f=logistic_log_loss))) / N
        assert np.array_equal(L, Z), f"LogLoss output should be {L} but {Z}."

        # --------------------------------------------------------------------------------
        # constraint: gradient_numerical() == numerical Jacobian numerical_jacobian(O, X)
        # Use a dummy layer for the objective function because using the "layer"
        # updates the X, Y which can interfere the independence of the layer.
        # --------------------------------------------------------------------------------
        GN = layer.gradient_numerical()                     # [dL/dX] from the layer

        # --------------------------------------------------------------------------------
        # Cannot use CrossEntropyLogLoss.function() to simulate the objective function L.
        # because it causes applying transform_X_T multiple times.
        # Because internally transform_X_T(X, T) has transformed T into the index label
        # in 1D with with length 1 by "T = T.reshape(-1)".
        # Then providing X in 1D into "dummy.function(x)" re-run "transform_X_T(X, T)"
        # again. The (X.ndim == T.ndim ==1) as an input and T must be OHE label for such
        # combination and T.shape == P.shape must be true for OHE labels.
        # However, T has been converted into the index format already by transform_X_T
        # (applying transform_X_T multiple times) and (T.shape=(1,1), X.shape=(1, > 1)
        # that violates the (X.shape == T.shape) constraint.
        # --------------------------------------------------------------------------------
        # dummy = CrossEntropyLogLoss(
        #     name="dummy",
        #     num_nodes=M,
        #     log_level=logging.DEBUG
        # )
        # dummy.T = T
        # dummy.objective = objective
        # dummy.function(X)
        # --------------------------------------------------------------------------------
        O = lambda x: np.sum(cross_entropy_log_loss(P=sigmoid(x), T=T, f=logistic_log_loss)) / N
        EGN = numerical_jacobian(O, X).reshape(1, -1)   # Expected numerical dL/dX
        assert np.array_equal(GN[0], EGN), \
            f"Layer gradient_numerical GN \n{GN} \nneeds to be \n{EGN}."

        # ================================================================================
        # Layer backward path
        # ================================================================================
        # --------------------------------------------------------------------------------
        # constraint: Analytical gradient G: gradient() == (P-1)/N.
        # --------------------------------------------------------------------------------
        dY = float(1)
        G = layer.gradient(dY)
        assert np.all(np.abs(G-EG) <= GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            f"Layer gradient dL/dX \n{G} \nneeds to be \n{EG}."

        # --------------------------------------------------------------------------------
        # constraint: Analytical gradient G is close to GN: gradient_numerical().
        # --------------------------------------------------------------------------------
        assert \
            np.all(np.abs(G-GN[0]) <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or \
            np.all(np.abs(G-GN[0]) <= np.abs(GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0])), \
            "dX is \n%s\nGN is \n%s\nG-GN is \n%s\n Ratio * GN[0] is \n%s.\n" \
            % (G, GN[0], G-GN[0], GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0])


def test_040_objective_methods_2d_ohe(caplog):
    """
    Objective:
        Verify the forward path constraints:
        1. Layer output L/loss is np.sum(cross_entropy_log_loss(sigmoid(X), T, f=logistic_log_loss))) / N.
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

    # caplog.set_level(logging.DEBUG, logger=__name__)
    caplog.set_level(logging.DEBUG)

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    name = "test_040_objective_methods_2d_ohe"
    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = 1
        layer = CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            activation=sigmoid,
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

        Logger.debug("%s: X is \n%s\nT is \n%s" % (name, X, T))

        P = sigmoid(X)
        EG = (P - T) / N       # Expected analytical gradient dL/dX = (P-T)/N

        # --------------------------------------------------------------------------------
        # constraint: L/loss == np.sum(cross_entropy_log_loss(sigmoid(X), T)) / N.
        # --------------------------------------------------------------------------------
        L = layer.function(X)
        Z = np.array(np.sum(cross_entropy_log_loss(P=sigmoid(X), T=T, f=logistic_log_loss))) / N
        assert np.array_equal(L, Z), f"SoftmaxLogLoss output should be {L} but {Z}."

        # --------------------------------------------------------------------------------
        # constraint: gradient_numerical() == numerical Jacobian numerical_jacobian(O, X)
        # --------------------------------------------------------------------------------
        GN = layer.gradient_numerical()                     # [dL/dX] from the layer

        # --------------------------------------------------------------------------------
        # DO not use CrossEntropyLogLoss.function() to simulate the objective function for
        # the expected GN. See the same part in test_040_objective_methods_1d_ohe().
        # --------------------------------------------------------------------------------
        # dummy= CrossEntropyLogLoss(
        #     name=name,
        #     num_nodes=M,
        #     log_level=logging.DEBUG
        # )
        # dummy.T = T
        # dummy.objective = objective
        # --------------------------------------------------------------------------------
        O = lambda x: np.sum(cross_entropy_log_loss(P=sigmoid(x), T=T, f=logistic_log_loss)) / N
        EGN = numerical_jacobian(O, X)                      # Expected numerical dL/dX
        assert np.array_equal(GN[0], EGN), \
            f"GN[0]==EGN expected but GN[0] is \n%s\n EGN is \n%s\n" % (GN[0], EGN)

        # ================================================================================
        # Layer backward path
        # ================================================================================
        # --------------------------------------------------------------------------------
        # constraint: Analytical gradient G: gradient() == (P-1)/N.
        # --------------------------------------------------------------------------------
        dY = float(1)
        G = layer.gradient(dY)
        assert np.all(np.abs(G-EG) <= GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            f"Layer gradient dL/dX \n{G} \nneeds to be \n{EG}."

        # --------------------------------------------------------------------------------
        # constraint: Analytical gradient G is close to GN: gradient_numerical().
        # --------------------------------------------------------------------------------
        assert \
            np.all(np.abs(G - GN[0]) <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or \
            np.all(np.abs(G - GN[0]) <= np.abs(GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0])), \
            f"dX is \n{G}\nGN[0] is \n{GN[0]}\nRatio * GN[0] is \n{GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0]}.\n"
