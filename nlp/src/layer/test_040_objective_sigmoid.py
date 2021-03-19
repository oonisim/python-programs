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
    TYPE_FLOAT,
    TYPE_LABEL,
    transform_X_T,
    sigmoid,
    logarithm,
    logistic_log_loss,
    categorical_log_loss,
    cross_entropy_log_loss,
    sigmoid_cross_entropy_log_loss,
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
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    ACTIVATION_DIFF_ACCEPTANCE_VALUE,
    LOSS_DIFF_ACCEPTANCE_VALUE
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
                log_loss_function=sigmoid_cross_entropy_log_loss
            )
            raise RuntimeError("CrossEntropyLogLoss initialization with invalid name must fail")
        except AssertionError:
            pass

        # Constraint: num_nodes == 1
        try:
            CrossEntropyLogLoss(
                name="test_040_objective",
                num_nodes=0,
                log_loss_function=sigmoid_cross_entropy_log_loss
            )
            raise RuntimeError("CrossEntropyLogLoss(num_nodes<1) must fail.")
        except AssertionError:
            pass

        try:
            CrossEntropyLogLoss(
                name="test_040_objective",
                num_nodes=np.random.randint(2, NUM_MAX_NODES),
                log_loss_function=sigmoid_cross_entropy_log_loss
            )
            raise RuntimeError("CrossEntropyLogLoss(num_nodes>1) must fail.")
        except AssertionError:
            pass

        # Constraint: logging level is correct.
        try:
            CrossEntropyLogLoss(
                name="test_040_objective",
                num_nodes=M,
                log_loss_function=sigmoid_cross_entropy_log_loss,
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
            log_loss_function=sigmoid_cross_entropy_log_loss,
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
            print(layer.P)
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
    name = "test_040_objective_instantiation"
    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = 1
        # For sigmoid log loss layer, the number of features N in X is the same with node number.
        D: int = M
        layer = CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            log_loss_function=sigmoid_cross_entropy_log_loss,
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
        assert layer.N == N == X.shape[0]
        # For sigmoid log loss layer, the number of features N in X is the same with node number.
        assert layer.M == X.shape[1]

        layer._dX = X
        assert np.array_equal(layer.dX, X)

        T = np.random.randint(0, M, N)
        layer.T = T
        assert np.array_equal(layer.T, T)

        # layer.function() gives the total loss L in shape ().
        # log_loss function require (X, T) in X(N, M), and T(N, M) in OHE label format.
        X, T = transform_X_T(X, T)
        L = layer.function(X)
        J, P = sigmoid_cross_entropy_log_loss(X, T)
        assert \
            L.shape == () and L == (np.sum(J) / N) and L == layer.Y, \
            "After setting T, layer.function(X) generates the total loss L but %s" % L

        # layer.function(X) sets layer.P to sigmoid_cross_entropy_log_loss(X, T)
        # P is nearly equal with sigmoid(X)
        assert \
            np.array_equal(layer.P, P) and \
            np.all(np.abs(layer.P - sigmoid(X)) < LOSS_DIFF_ACCEPTANCE_VALUE), \
            "layer.function(X) needs to set P as sigmoid_cross_entropy_log_loss(X, T) " \
            "which is close to sigmoid(X) but layer.P=\n%s\nP=\n%s\nsigmoid(X)=%s" \
            % (layer.P, P, sigmoid(X))

        # gradient of sigmoid cross entropy log loss layer is (P-T)/N
        G = layer.gradient()
        assert \
            np.all(np.abs(G - ((P-T)/N)) < GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            "Gradient G needs (P-T)/N but G=\n%s\n(P-T)/N=\n%s\n" % (G, (P-T)/N)

        layer.logger.debug("This is a pytest")

        assert layer.objective(np.array(1.0)) == np.array(1.0), \
            "Objective function of the output/last layer is an identity function."


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
    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    name = "test_040_objective_methods_1d_ohe"
    N = 1

    for _ in range(NUM_MAX_TEST_TIMES):
        layer = CrossEntropyLogLoss(
            name=name,
            num_nodes=1,
            log_loss_function=sigmoid_cross_entropy_log_loss,
            log_level=logging.DEBUG
        )

        # ================================================================================
        # Layer forward path
        # ================================================================================
        X = np.random.uniform(low=-BOUNDARY_SIGMOID, high=BOUNDARY_SIGMOID)
        T = np.random.randint(0, 2)                # OHE labels.

        # log_loss function require (X, T) in X(N, M), and T(N, M) in OHE label format.
        X, T = transform_X_T(X, T)
        layer.T = T

        # Expected analytical gradient dL/dX = (P-T)/N of shape (N,M)
        A = sigmoid(X)
        EG = ((A - T) / N).reshape(1, -1)

        Logger.debug("%s: X is \n%s\nT is %s\nP is %s\nEG is %s\n", name, X, T, A, EG)

        # --------------------------------------------------------------------------------
        # constraint: L/loss == np.sum(J) / N.
        # J, P = sigmoid_cross_entropy_log_loss(X, T)
        # --------------------------------------------------------------------------------
        L = layer.function(X)       # L is shape ()
        J, P = sigmoid_cross_entropy_log_loss(X, T)
        Z = np.array(np.sum(J)) / N
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
        def objective(x):
            j, p = sigmoid_cross_entropy_log_loss(x, T)
            return np.array(np.sum(j) / N, dtype=TYPE_FLOAT)

        EGN = numerical_jacobian(objective, X).reshape(1, -1)   # Expected numerical dL/dX
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
        1. Layer output L/loss is np.sum(sigmoid_cross_entropy_log_loss) / N.
        2. gradient_numerical() == numerical Jacobian numerical_jacobian(O, X).

        Verify the backward path constraints:
        1. Analytical gradient G: gradient() == (P-1)/N
        2. Analytical gradient G is close to GN: gradient_numerical().
    """
    caplog.set_level(logging.DEBUG)

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    name = "test_040_objective_methods_2d_ohe"
    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = 1      # node number is 1 for 0/1 binary classification.
        layer = CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            log_loss_function=sigmoid_cross_entropy_log_loss,
            log_level=logging.DEBUG
        )

        # ================================================================================
        # Layer forward path
        # ================================================================================
        X = np.random.randn(N, M)
        T = np.zeros_like(X, dtype=TYPE_LABEL)     # OHE labels.
        T[
            np.arange(N),
            np.random.randint(0, M, N)
        ] = int(1)

        # log_loss function require (X, T) in X(N, M), and T(N, M) in OHE label format.
        X, T = transform_X_T(X, T)
        layer.T = T
        Logger.debug("%s: X is \n%s\nT is \n%s", name, X, T)

        # --------------------------------------------------------------------------------
        # Expected analytical gradient EG = (dX/dL) = (A-T)/N
        # --------------------------------------------------------------------------------
        A = sigmoid(X)
        EG = (A - T) / N

        # --------------------------------------------------------------------------------
        # Total loss Z = np.sum(J)/N
        # Expected loss EL = sum((1-T)X + np.log(1 + np.exp(-X)))
        # (J, P) = sigmoid_cross_entropy_log_loss(X, T) and J:shape(N,) where J:shape(N,)
        # is loss for each input and P is activation by sigmoid(X).
        # --------------------------------------------------------------------------------
        L = layer.function(X)
        J, P = sigmoid_cross_entropy_log_loss(X, T)
        EL = np.array(np.sum((1-T) * X + logarithm(1 + np.exp(-X))) / N, dtype=TYPE_FLOAT)

        # Constraint: A == P as they are sigmoid(X)
        assert np.all(np.abs(A-P) < ACTIVATION_DIFF_ACCEPTANCE_VALUE), \
            f"Need A==P==sigmoid(X) but A=\n{A}\n P=\n{P}\n(A-P)=\n{(A-P)}\n"

        # Constraint: Log loss layer output L == sum(J) from the log loss function
        Z = np.array(np.sum(J) / N, dtype=TYPE_FLOAT)
        assert np.array_equal(L, Z), \
            f"Need log loss layer output L == sum(J) but L=\n{L}\nZ=\n{Z}."

        # Constraint: L/loss is close to expected loss EL.
        assert np.all(np.abs(EL-L) < LOSS_DIFF_ACCEPTANCE_VALUE), \
            "Need EL close to L but \nEL=\n{EL}\nL=\n{L}\n"

        # --------------------------------------------------------------------------------
        # constraint: gradient_numerical() == numerical_jacobian(objective, X)
        # TODO: compare the diff to accommodate numerical errors.
        # --------------------------------------------------------------------------------
        GN = layer.gradient_numerical()                     # [dL/dX] from the layer

        def objective(x):
            """Function to calculate the scalar loss L for cross entropy log loss"""
            j, p = sigmoid_cross_entropy_log_loss(x, T)
            return np.array(np.sum(j) / N, dtype=TYPE_FLOAT)

        EGN = numerical_jacobian(objective, X)              # Expected numerical dL/dX
        assert np.array_equal(GN[0], EGN), \
            f"GN[0]==EGN expected but GN[0] is \n%s\n EGN is \n%s\n" % (GN[0], EGN)

        # ================================================================================
        # Layer backward path
        # ================================================================================
        # constraint: Analytical gradient G: gradient() == (P-1)/N.
        dY = float(1)
        G = layer.gradient(dY)
        assert np.all(np.abs(G-EG) <= GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            f"Layer gradient dL/dX \n{G} \nneeds to be \n{EG}."

        # constraint: Analytical gradient G is close to GN: gradient_numerical().
        assert \
            np.all(np.abs(G - GN[0]) <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or \
            np.all(np.abs(G - GN[0]) <= np.abs(GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0])), \
            f"dX is \n{G}\nGN[0] is \n{GN[0]}\nRatio * GN[0] is \n{GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0]}.\n"

        # constraint: Gradient g of the log loss layer needs -1 < g < 1
        # abs(P-T) = abs(sigmoid(X)-T) cannot be > 1.
        assert np.all(np.abs(G) < 1), \
            f"Log loss layer gradient cannot be < -1 nor > 1 but\n{G}"
        assert np.all(np.abs(GN[0]) < (1+GRADIENT_DIFF_ACCEPTANCE_RATIO)), \
            f"Log loss layer gradient cannot be < -1 nor > 1 but\n{GN[0]}"
