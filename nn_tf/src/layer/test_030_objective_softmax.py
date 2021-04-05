"""Objective (loss) layer test cases"""
import cProfile
import logging
from typing import (
    Union
)

import numpy as np

from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL,
)
from common.function import (
    transform_X_T,
    softmax,
    logarithm,
    cross_entropy_log_loss,
    softmax_cross_entropy_log_loss,
    numerical_jacobian,
)
from common.utility import (
    random_string,
)
import layer
from layer.constants import (
    _NAME,
    _NUM_NODES,
    _SCHEME,
    _LOSS_FUNCTION,
    _PARAMETERS
)
from test.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    GRADIENT_DIFF_ACCEPTANCE_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    ACTIVATION_DIFF_ACCEPTANCE_VALUE,
    LOSS_DIFF_ACCEPTANCE_VALUE
)

Logger = logging.getLogger(__name__)


def _must_fail(
        name: str,
        num_nodes: int,
        log_level: int = logging.ERROR,
        msg: str = ""
):
    assert msg
    try:
        layer.CrossEntropyLogLoss(
            name=name,
            num_nodes=num_nodes,
            log_level=log_level
        )
        raise RuntimeError(msg)
    except AssertionError:
        pass


def test_030_objective_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_030_objective_instantiation_to_fail"
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        M: int = np.random.randint(1, NUM_MAX_NODES)
        # Constraint: Name is string with length > 0.
        _must_fail(
            name="",
            num_nodes=1,
            msg="CrossEntropyLogLoss initialization with invalid name must fail"
        )

        # Constraint: num_nodes > 1
        _must_fail(
            name=name,
            num_nodes=0,
            msg="CrossEntropyLogLoss(num_nodes<1) must fail."
        )
        _must_fail(
            name=name,
            num_nodes=1,
            msg="CrossEntropyLogLoss(num_nodes<2) must fail."
        )
        # Constraint: logging level is correct.
        try:
            layer.CrossEntropyLogLoss(
                name="test_030_objective",
                num_nodes=M,
                log_level=-1
            )
            raise RuntimeError()
        except (AssertionError, KeyError):
            pass

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_030_objective_instance_properties():
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    msg = "Accessing uninitialized property of the layer must fail."
    name = random_string(np.random.randint(1, 10))
    for _ in range(NUM_MAX_TEST_TIMES):
        M: int = np.random.randint(2, NUM_MAX_NODES)
        _layer = layer.CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not _layer.name == name: raise RuntimeError("layer.name == name should be true")
        except AssertionError:
            raise RuntimeError("Access to name should be allowed as already initialized.")

        try:
            if not _layer.M == M: raise RuntimeError("layer.M == M should be true")
        except AssertionError:
            raise RuntimeError("Access to M should be allowed as already initialized.")

        try:
            if not isinstance(_layer.logger, logging.Logger):
                raise RuntimeError("isinstance(layer.logger, logging.Logger) should be true")
        except AssertionError:
            raise RuntimeError("Access to logger should be allowed as already initialized.")

        # --------------------------------------------------------------------------------
        # To fail
        # --------------------------------------------------------------------------------
        try:
            print(_layer.X)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            _layer.X = int(1)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.N)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.dX)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.Y)
            raise RuntimeError(msg)
        except AssertionError:
            pass
        try:
            print(_layer.P)
            raise RuntimeError(msg)
        except AssertionError:
            pass
        try:
            _layer._Y = int(1)
            print(_layer.Y)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.dY)
            raise RuntimeError(msg)
        except AssertionError:
            pass
        try:
            _layer._dY = int(1)
            print(_layer.dY)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.T)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.L)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.J)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            _layer.T = float(1)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            _layer.function(int(1))
            raise RuntimeError("Invoke layer.function(int(1)) must fail.")
        except AssertionError:
            pass

        try:
            _layer.function(1.0)
            _layer.gradient(int(1))
            raise RuntimeError("Invoke layer.gradient(int(1)) must fail.")
        except AssertionError:
            pass

        del _layer


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
    name = "test_030_objective_instantiation"
    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        assert M >= 2, "Softmax is for multi label classification. "\
                       " Use Sigmoid for binary classification."

        # For softmax log loss layer, the number of features N in X is the same with node number.
        D: int = M
        _layer = layer.CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )

        # --------------------------------------------------------------------------------
        # Properties
        # --------------------------------------------------------------------------------
        assert _layer.name == name
        assert _layer.num_nodes == _layer.M == M

        _layer._D = D
        assert _layer.D == D

        X = np.random.randn(N, D)
        _layer.X = X
        assert np.array_equal(_layer.X, X)
        assert _layer.N == N == X.shape[0]
        # For softmax log loss _layer, the number of features N in X is the same with node number.
        assert _layer.M == X.shape[1]

        _layer._dX = X
        assert np.array_equal(_layer.dX, X)

        T = np.random.randint(0, M, N)
        _layer.T = T
        assert np.array_equal(_layer.T, T)

        # Once T is set, objective() is available and callable
        assert _layer.objective(_layer.function(X))

        _layer._Y = np.dot(X, X.T)
        assert np.array_equal(_layer.Y, np.dot(X, X.T))

        _layer._dY = np.array(0.9)
        assert _layer._dY == np.array(0.9)

        _layer.logger.debug("This is a pytest")

        assert _layer.objective(np.array(1.0)) == np.array(1.0)


def test_030_objective_specification():
    name = "loss001"
    num_nodes = 16
    expected_spec = {
        _SCHEME: layer.CrossEntropyLogLoss.__qualname__,
        _PARAMETERS: {
            _NAME: name,
            _NUM_NODES: num_nodes,
            _LOSS_FUNCTION: softmax_cross_entropy_log_loss.__qualname__
        }
    }
    actual_spec = layer.CrossEntropyLogLoss.specification(name=name, num_nodes=num_nodes)
    assert expected_spec == actual_spec, \
        "expected\n%s\nactual\n%s\n" % (expected_spec, actual_spec)


def test_030_objective_methods_1d_ohe():
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
    name = "test_030_objective_methods_1d_ohe"
    N = 1

    for _ in range(NUM_MAX_TEST_TIMES):
        M: int = np.random.randint(2, NUM_MAX_NODES)
        assert M >= 2, "Softmax is for multi label classification. "\
                       " Use Sigmoid for binary classification."

        _layer = layer.CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )

        # ================================================================================
        # Layer forward path
        # ================================================================================
        X = np.random.randn(M)
        T = np.zeros_like(X, dtype=TYPE_LABEL)     # OHE labels.
        T[
            np.random.randint(0, M)
        ] = int(1)
        _layer.T = T

        P = softmax(X)
        EG = ((P - T) / N).reshape(1, -1)     # Expected analytical gradient dL/dX = (P-T)/N

        Logger.debug(
            "%s: X is \n%s\nT is %s\nP is %s\nEG is %s\n",
            name, X, T, P, EG
        )

        # --------------------------------------------------------------------------------
        # constraint: L/loss == np.sum(cross_entropy_log_loss(softmax(X), T)) / N.
        # --------------------------------------------------------------------------------
        L = _layer.function(X)
        Z = np.array(np.sum(cross_entropy_log_loss(softmax(X), T))) / N
        assert np.array_equal(L, Z), f"SoftmaxLogLoss output should be {L} but {Z}."

        # --------------------------------------------------------------------------------
        # constraint: gradient_numerical() == numerical Jacobian numerical_jacobian(O, X)
        # Use a dummy _layer for the objective function because using the "_layer"
        # updates the X, Y which can interfere the independence of the _layer.
        # --------------------------------------------------------------------------------
        GN = _layer.gradient_numerical()                     # [dL/dX] from the _layer

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
        # O = lambda x: dummy.objective(dummy.function(x))    # Objective function
        O = lambda x: np.sum(cross_entropy_log_loss(softmax(x), T)) / N
        # --------------------------------------------------------------------------------
        EGN = numerical_jacobian(O, X).reshape(1, -1) # Expected numerical dL/dX
        assert np.array_equal(GN[0], EGN), \
            f"Layer gradient_numerical GN \n{GN} \nneeds to be \n{EGN}."

        # ================================================================================
        # Layer backward path
        # ================================================================================
        # --------------------------------------------------------------------------------
        # constraint: Analytical gradient G: gradient() == (P-1)/N.
        # --------------------------------------------------------------------------------
        dY = float(1)
        G = _layer.gradient(dY)
        assert np.all(np.abs(G-EG) <= GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            f"Layer gradient dL/dX \n{G} \nneeds to be \n{EG} but G-EG \n{np.abs(G-EG)}\n"

        # --------------------------------------------------------------------------------
        # constraint: Analytical gradient G is close to GN: gradient_numerical().
        # --------------------------------------------------------------------------------
        assert \
            np.all(np.abs(G - GN[0]) <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or \
            np.all(np.abs(G-GN[0]) <= np.abs(GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0])), \
            f"dX is \n{G}\nGN[0] is \n{GN[0]}\nRatio * GN[0] is \n{GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0]}.\n"


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
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    name = "test_030_objective_methods_2d_ohe"
    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        assert M >= 2, "Softmax is for multi label classification. "\
                       " Use Sigmoid for binary classification."

        _layer = layer.CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        _layer.objective = objective

        # ================================================================================
        # Layer forward path
        # ================================================================================
        X = np.random.randn(N, M)
        T = np.zeros_like(X, dtype=TYPE_LABEL)     # OHE labels.
        T[
            np.arange(N),
            np.random.randint(0, M, N)
        ] = int(1)
        _layer.T = T

        Logger.debug("%s: X is \n%s\nT is \n%s", name, X, T)

        P = softmax(X)
        EG = (P - T) / N       # Expected analytical gradient dL/dX = (P-T)/N

        # --------------------------------------------------------------------------------
        # constraint: L/loss == np.sum(cross_entropy_log_loss(softmax(X), T)) / N.
        # --------------------------------------------------------------------------------
        L = _layer.function(X)
        Z = np.array(np.sum(cross_entropy_log_loss(softmax(X), T))) / N
        assert np.array_equal(L, Z), f"SoftmaxLogLoss output should be {L} but {Z}."

        # --------------------------------------------------------------------------------
        # constraint: gradient_numerical() == numerical Jacobian numerical_jacobian(O, X)
        # --------------------------------------------------------------------------------
        GN = _layer.gradient_numerical()                     # [dL/dX] from the _layer

        # --------------------------------------------------------------------------------
        # DO not use CrossEntropyLogLoss.function() to simulate the objective function for
        # the expected GN. See the same part in test_030_objective_methods_1d_ohe().
        # --------------------------------------------------------------------------------
        # dummy= CrossEntropyLogLoss(
        #     name=name,
        #     num_nodes=M,
        #     log_level=logging.DEBUG
        # )
        # dummy.T = T
        # dummy.objective = objective
        # O = lambda x: dummy.objective(dummy.function(x))    # Objective function
        O = lambda x: np.sum(cross_entropy_log_loss(softmax(x), T)) / N
        # --------------------------------------------------------------------------------

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
        G = _layer.gradient(dY)
        assert np.all(abs(G-EG) <= GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            f"Layer gradient dL/dX \n{G} \nneeds to be \n{EG}."

        # --------------------------------------------------------------------------------
        # constraint: Analytical gradient G is close to GN: gradient_numerical().
        # --------------------------------------------------------------------------------
        assert \
            np.all(np.abs(G - GN[0]) <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or \
            np.all(np.abs(G - GN[0]) <= np.abs(GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0])), \
            f"dX is \n{G}\nGN[0] is \n{GN[0]}\nRatio * GN[0] is \n{GRADIENT_DIFF_ACCEPTANCE_RATIO * GN[0]}.\n"


def test_040_softmax_log_loss_2d(caplog):
    """
    Objective:
        Verify the forward path constraints:
        1. Layer output L/loss is np.sum(softmax_cross_entropy_log_loss) / N.
        2. gradient_numerical() == numerical_jacobian(objective, X).

        Verify the backward path constraints:
        1. Analytical gradient G: gradient() == (P-1)/N
        2. Analytical gradient G is close to GN: gradient_numerical().
    """
    caplog.set_level(logging.DEBUG)

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    name = "test_040_softmax_log_loss_2d_ohe"

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)    # number of node > 1
        _layer = layer.CrossEntropyLogLoss(
            name=name,
            num_nodes=M,
            log_loss_function=softmax_cross_entropy_log_loss,
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

        # log_loss function require (X, T) in X(N, M), and T(N, M) in index label format.
        X, T = transform_X_T(X, T)
        _layer.T = T
        Logger.debug("%s: X is \n%s\nT is \n%s", name, X, T)

        # --------------------------------------------------------------------------------
        # Expected analytical gradient EG = (dX/dL) = (A-T)/N
        # --------------------------------------------------------------------------------
        A = softmax(X)
        EG = np.copy(A)
        EG[
            np.arange(N),
            T
        ] -= 1   # Shape(N,), subtract from elements for T=1 only
        EG /= N

        # --------------------------------------------------------------------------------
        # Total loss Z = np.sum(J)/N
        # Expected loss EL = -sum(T*log(_A))
        # (J, P) = softmax_cross_entropy_log_loss(X, T) and J:shape(N,) where J:shape(N,)
        # is loss for each input and P is activation by sigmoid(X).
        # --------------------------------------------------------------------------------
        L = _layer.function(X)
        J, P = softmax_cross_entropy_log_loss(X, T)
        EL = np.array(-np.sum(logarithm(A[np.arange(N), T])) / N, dtype=TYPE_FLOAT)

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

        # constraint: gradient_numerical() == numerical_jacobian(objective, X)
        # TODO: compare the diff to accommodate numerical errors.
        GN = _layer.gradient_numerical()                     # [dL/dX] from the layer

        def objective(x):
            """Function to calculate the scalar loss L for cross entropy log loss"""
            j, p = softmax_cross_entropy_log_loss(x, T)
            return np.array(np.sum(j) / N, dtype=TYPE_FLOAT)

        EGN = numerical_jacobian(objective, X)              # Expected numerical dL/dX
        assert np.array_equal(GN[0], EGN), \
            f"GN[0]==EGN expected but GN[0] is \n%s\n EGN is \n%s\n" % (GN[0], EGN)

        # ================================================================================
        # Layer backward path
        # ================================================================================

        # constraint: Analytical gradient G: gradient() == EG == (P-1)/N.
        dY = float(1)
        G = _layer.gradient(dY)
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

    profiler.disable()
    profiler.print_stats(sort="cumtime")