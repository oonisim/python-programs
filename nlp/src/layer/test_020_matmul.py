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
import cProfile
import os
import copy
import pathlib
import logging
from typing import (
    Union
)

import numpy as np

from common.constant import (
    TYPE_FLOAT
)
import common.weights as weights
from common.function import (
    numerical_jacobian,
)
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS
)
from layer import (
    Matmul
)
from test.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES,
    GRADIENT_DIFF_ACCEPTANCE_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO
)
from test.layer_validations import (
    validate_against_expected_gradient
)
from optimizer import (
    SGD
)
from common.utility import (
    random_string
)


Logger = logging.getLogger(__name__)


def _instantiate(name: str, num_nodes: int, num_features: int, objective=None):
    category = np.random.uniform()
    if category < 0.3:
        W=weights.he(num_nodes, num_features + 1)
    elif category < 0.7:
        W=weights.xavier(num_nodes, num_features + 1)
    else:
        W=weights.uniform(num_nodes, num_features + 1)

    matmul = Matmul(
        name=name,
        num_nodes=num_nodes,
        W=W
    )
    if objective is not None:
        matmul.objective = objective
    return matmul


def _generate_X(N: int = -1, D: int = -1):
    N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE) if N <= 0 else N
    D: int = np.random.randint(1, NUM_MAX_FEATURES) if D <= 0 else D
    return np.random.rand(N, D)


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
        matmul = Matmul(
            name=name,
            num_nodes=M,
            W=weights.uniform(M, D+1),
            log_level=logging.DEBUG
        )

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not matmul.name == name: raise RuntimeError("matmul.name == name should be true")
        except AssertionError as e:
            raise RuntimeError("Access to name should be allowed as already initialized.") from e

        try:
            if not matmul.M == M: raise RuntimeError("matmul.M == M should be true")
        except AssertionError as e:
            raise RuntimeError("Access to M should be allowed as already initialized.") from e

        try:
            if not isinstance(matmul.logger, logging.Logger):
                raise RuntimeError("isinstance(matmul.logger, logging.Logger) should be true")
        except AssertionError as e:
            raise RuntimeError("Access to logger should be allowed as already initialized.") from e

        try:
            a = matmul.D
        except AssertionError:
            raise RuntimeError("Access to D should be allowed as already initialized.")

        try:
            matmul.W is not None
        except AssertionError:
            raise RuntimeError("Access to W should be allowed as already initialized.")

        try:
            matmul.optimizer is not None
        except AssertionError:
            raise RuntimeError("Access to optimizer should be allowed as already initialized.")

        # --------------------------------------------------------------------------------
        # To fail
        # --------------------------------------------------------------------------------
        try:
            print(matmul.X)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            matmul.X = int(1)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(matmul.dX)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(matmul.dW)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(matmul.Y)
            raise RuntimeError(msg)
        except AssertionError:
            pass
        try:
            matmul._Y = int(1)
            print(matmul.Y)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(matmul.dY)
            raise RuntimeError(msg)
        except AssertionError:
            pass
        try:
            matmul._dY = int(1)
            print(matmul.dY)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(matmul.T)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            matmul.T = float(1)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            # pylint: disable=not-callable
            matmul.objective(np.array(1.0))
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(matmul.N)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        assert matmul.name == name
        assert matmul.num_nodes == M

        try:
            matmul = Matmul(
                name=name,
                num_nodes=M,
                W=weights.xavier(M, D+1),
                log_level=logging.DEBUG
            )
            matmul.function(int(1))
            raise RuntimeError("Invoke matmul.function(int(1)) must fail.")
        except AssertionError:
            pass

        try:
            matmul = Matmul(
                name=name,
                num_nodes=M,
                W=weights.xavier(M, D+1),
                log_level=logging.DEBUG
            )
            matmul.gradient(int(1))
            raise RuntimeError("Invoke matmul.gradient(int(1)) must fail.")
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
        matmul = Matmul(
            name=name,
            num_nodes=M,
            W=weights.he(M, D+1),
            log_level=logging.DEBUG
        )
        matmul.objective = objective

        assert matmul.name == name
        assert matmul.num_nodes == matmul.M == M

        matmul._D = D
        assert matmul.D == D

        X = np.random.randn(N, D)
        matmul.X = X
        assert np.array_equal(matmul.X, X)
        assert matmul.N == N == X.shape[0]

        matmul._dX = X
        assert np.array_equal(matmul.dX, X)

        T = np.random.randint(0, M, N)
        matmul.T = T
        assert np.array_equal(matmul.T, T)

        matmul._Y = np.dot(X, X.T)
        assert np.array_equal(matmul.Y, np.dot(X, X.T))

        matmul._dY = np.array(0.9)
        assert matmul._dY == np.array(0.9)

        matmul.logger.debug("This is a pytest")

        assert matmul.objective == objective


def test_020_matmul_builder_to_fail_matmul_spec():
    """
    Objective:
        Verify the Matmul.build()
    Expected:
        build() parse the spec and fail with invalid configurations
    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        M = np.random.randint(1, 100)
        D = np.random.randint(1, 100)   # NOT including bias

        # ----------------------------------------------------------------------
        # Validate the correct specification.
        # NOTE: Invalidate one parameter at a time from the correct one.
        # Otherwise not sure what you are testing.
        # ----------------------------------------------------------------------
        valid_matmul_spec = {
            _NAME: "test_020_matmul_builder_to_fail_matmul_spec",
            _NUM_NODES: M,
            _NUM_FEATURES: D,
            _WEIGHTS: {
                _SCHEME: "he"
            },
            "log_level": logging.ERROR
        }
        try:
            Matmul.build(valid_matmul_spec)
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_NAME] = ""
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid name")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_NUM_NODES] = np.random.randint(-100, 0)
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with num_nodes <=0")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_NUM_FEATURES] = np.random.randint(-100, 0)
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with num_features <=0")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["log_level"] = -999
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with log_level <-1")
        except KeyError:
            pass

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_matmul_builder_to_fail_weight_spec():
    """
    Objective:
        Verify the Matmul.build()
    Expected:
        build() parse the spec and fail with invalid weight configurations
    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        M = np.random.randint(1, 100)
        D = np.random.randint(1, 100)   # NOT including bias

        # ----------------------------------------------------------------------
        # Validate the correct specification.
        # NOTE: Invalidate one parameter at a time from the correct one.
        # Otherwise not sure what you are testing.
        # ----------------------------------------------------------------------
        valid_matmul_spec = {
            _NAME: "test_020_matmul_builder_to_fail_matmul_spec",
            _NUM_NODES: M,
            _NUM_FEATURES: D,
            _WEIGHTS: {
                _SCHEME: "he"
            }
        }
        try:
            Matmul.build(valid_matmul_spec)
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_WEIGHTS][_SCHEME] = "invalid_scheme"
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid weight scheme")
        except AssertionError:
            pass

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_matmul_builder_to_fail_optimizer_spec():
    """
    Objective:
        Verify the Matmul.build()
    Expected:
        build() parse the spec and fail with invalid configurations
    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        M = np.random.randint(1, 100)
        D = np.random.randint(1, 100)   # NOT including bias

        # ----------------------------------------------------------------------
        # Validate the correct specification.
        # NOTE: Invalidate one parameter at a time from the correct one.
        # Otherwise not sure what you are testing.
        # ----------------------------------------------------------------------
        valid_matmul_spec = {
            _NAME: "test_020_matmul_builder_to_fail_matmul_spec",
            _NUM_NODES: M,
            _NUM_FEATURES: D,
            _WEIGHTS: {
                _SCHEME: "he"
            },
            _OPTIMIZER: {
                _SCHEME: "sGd",
                _PARAMETERS: {
                    "lr": np.random.uniform(),
                    "l2": np.random.uniform()
                }
            },
            "log_level": logging.ERROR
        }
        try:
            Matmul.build(valid_matmul_spec)
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_OPTIMIZER] = ""
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid optimizer spec")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_OPTIMIZER][_SCHEME] = "invalid"
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid optimizer spec")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_OPTIMIZER][_PARAMETERS]["lr"] = np.random.uniform(-1, 0)
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid lr value")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_OPTIMIZER][_PARAMETERS]["l2"] = np.random.uniform(-1, 0)
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid l2 value")
        except AssertionError:
            pass

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_matmul_build_specification():
    name = "matmul01"
    num_nodes = 8
    num_features = 2
    weights_initialization_scheme = "he"
    expected_spec = {
        _SCHEME: Matmul.__qualname__,
        _PARAMETERS: {
            _NAME: name,
            _NUM_NODES: num_nodes,
            _NUM_FEATURES: num_features,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: weights_initialization_scheme
            },
            _OPTIMIZER: SGD.specification(name="sgd")
        }
    }
    actual_spec = Matmul.specification(
        name=name,
        num_nodes=num_nodes,
        num_features=num_features,
        weights_initialization_scheme=weights_initialization_scheme,

    )
    assert expected_spec == actual_spec, \
        "expected\n%s\nactual\n%s\n" % (expected_spec, actual_spec)


def test_020_matmul_builder_to_succeed():
    """
    Objective:
        Verify the Matmul.build()
    Expected:
        build() parse the spec and succeed
    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        M = np.random.randint(1, 100)
        D = np.random.randint(1, 100)   # NOT including bias

        # ----------------------------------------------------------------------
        # Validate the correct specification.
        # NOTE: Invalidate one parameter at a time from the correct one.
        # Otherwise not sure what you are testing.
        # ----------------------------------------------------------------------
        lr = np.random.uniform()
        l2 = np.random.uniform()
        valid_matmul_spec = {
            _NAME: "test_020_matmul_builder_to_fail_matmul_spec",
            _NUM_NODES: M,
            _NUM_FEATURES: D,
            _WEIGHTS: {
                _SCHEME: "he",
            },
            _OPTIMIZER: {
                _SCHEME: "sGd",
                _PARAMETERS: {
                    "lr": lr,
                    "l2": l2
                }
            }
        }
        try:
            matmul:Matmul = Matmul.build(valid_matmul_spec)
            assert matmul.optimizer.lr == lr
            assert matmul.optimizer.l2 == l2
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_OPTIMIZER][_SCHEME] = "sgd"
        try:
            Matmul.build(valid_matmul_spec)
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec[_OPTIMIZER][_SCHEME] = "SGD"
        try:
            Matmul.build(valid_matmul_spec)
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


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

        # Test both static instantiation and build()
        if np.random.uniform() < 0.5:
            matmul = Matmul(
                name=name,
                num_nodes=M,
                W=W,
                log_level=logging.DEBUG
            )
        else:
            matmul_spec = {
                _NAME: "test_020_matmul_builder_to_fail_matmul_spec",
                _NUM_NODES: M,
                _NUM_FEATURES: D,
                _WEIGHTS: {
                    _SCHEME: "he",
                },
                _OPTIMIZER: {
                    _SCHEME: "sGd"
                }
            }
            matmul = Matmul.build(matmul_spec)

        matmul.objective = objective

        # ================================================================================
        # Layer forward path
        # Calculate the layer output Y=f(X), and get the loss L = objective(Y)
        # Test the numerical gradient dL/dX=matmul.gradient_numerical().
        #
        # Note that bias columns are added inside the matmul layer instance, hence
        # matmul.X.shape is (N, 1+D), matmul.W.shape is (M, 1+D)
        # ================================================================================
        X = np.random.randn(N, D)
        Logger.debug("%s: X is \n%s", name, X)

        # pylint: disable=not-callable
        Y = matmul.function(X)
        # pylint: disable=not-callable
        L = matmul.objective(Y)

        # Constraint 1 : Matmul outputs Y should be X@W.T
        assert np.array_equal(Y, np.matmul(matmul.X, matmul.W.T))

        # Constraint 2: Numerical gradient should be the same with numerical Jacobian
        GN = matmul.gradient_numerical()         # [dL/dX, dL/dW]

        # DO NOT use matmul.function() as the objective function for numerical_jacobian().
        # The state of the layer will be modified.
        # LX = lambda x: matmul.objective(matmul.function(x))
        def LX(x):
            y = np.matmul(x, matmul.W.T)
            # pylint: disable=not-callable
            return matmul.objective(y)

        EGNX = numerical_jacobian(LX, matmul.X)          # Numerical dL/dX including bias
        EGNX = EGNX[::, 1::]                            # Remove bias for dL/dX
        assert np.array_equal(GN[0], EGNX), \
            "GN[0]\n%s\nEGNX=\n%s\n" % (GN[0], EGNX)

        # DO NOT use matmul.function() as the objective function for numerical_jacobian().
        # The state of the layer will be modified.
        # LW = lambda w: matmul.objective(np.matmul(X, w.T))
        def LW(w):
            Y = np.matmul(matmul.X, w.T)
            # pylint: disable=not-callable
            return matmul.objective(Y)

        EGNW = numerical_jacobian(LW, matmul.W)          # Numerical dL/dW including bias
        assert np.array_equal(GN[1], EGNW)              # No need to remove bias

        # ================================================================================
        # Layer backward path
        # Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dummy dL/dY.
        # ================================================================================
        dY = np.ones_like(Y)
        dX = matmul.gradient(dY)

        # Constraint 3: Matmul gradient dL/dX should be dL/dY @ W. Use a dummy dL/dY = 1.0.
        expected_dX = np.matmul(dY, matmul.W)
        expected_dX = expected_dX[
            ::,
            1::     # Omit bias
        ]
        assert np.array_equal(dX, expected_dX)

        # Constraint 5: Analytical gradient dL/dX close to the numerical gradient GN.
        assert np.all(np.abs(dX - GN[0]) < GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            "dX need close to GN[0] but dX \n%s\n GN[0] \n%s\n" % (dX, GN[0])

        # --------------------------------------------------------------------------------
        # Gradient update.
        # Run the gradient descent to update Wn+1 = Wn - lr * dL/dX.
        # --------------------------------------------------------------------------------
        # Python passes the reference to W, hence it is directly updated by the gradient-
        # descent to avoid a temporary copy. Backup W before to compare before/after.
        backup = copy.deepcopy(W)

        # Gradient descent and returns analytical dL/dX, dL/dW
        dS = matmul.update()
        dW = dS[0]

        # Constraint 6.: W has been updated by the gradient descent.
        assert np.any(backup != matmul.W), "W has not been updated "

        # Constraint 5: the numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
        assert validate_against_expected_gradient(GN[0], dX), \
            "dX=\n%s\nGN[0]=\n%sdiff=\n%s\n" % (dX, GN[0], (dX-GN[0]))
        assert validate_against_expected_gradient(GN[1], dW), \
            "dW=\n%s\nGN[1]=\n%sdiff=\n%s\n" % (dW, GN[1], (dW-GN[1]))

        # Constraint 7: gradient descent progressing with the new objective L(Yn+1) < L(Yn)
        # pylint: disable=not-callable
        assert np.all(np.abs(objective(matmul.function(X)) < L))

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_matmul_save_load():
    """
    Objective:
        Verify the load/save methods.

    Constraints:
        1. Be able to save the layer state.
        2. Be able to load the layer state and the state is same with the layer state S.
    """

    name = "test_020_matmul_save_load"

    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function to calculate the loss L"""
        return np.sum(X)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(1, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)

        name = "test_020_matmul_methods"
        matmul = _instantiate(name=name, num_nodes=M, num_features=D, objective=objective)
        X = _generate_X(N, D)
        Y = matmul.function(X)
        matmul.gradient(Y)
        matmul.update()

        backup_S = copy.deepcopy(matmul.S)
        backup_W = copy.deepcopy(matmul.W)

        pathname = os.path.sep + os.path.sep.join([
            "tmp",
            name + random_string(12) + ".pkl"
        ])

        # ********************************************************************************
        # Constraint:
        #   load must fail with the saved file being deleted.
        #   Make sure load() is loading the path
        # ********************************************************************************
        matmul.save(pathname)
        path = pathlib.Path(pathname)
        path.unlink()
        try:
            msg = "load must fail with the saved file being deleted."
            matmul.load(pathname)
            raise AssertionError(msg)
        except RuntimeError as e:
            pass

        # ********************************************************************************
        # Constraint:
        #   load restore the state before save
        # ********************************************************************************
        matmul.save(pathname)
        matmul._W = np.zeros(shape=matmul.W.shape, dtype=TYPE_FLOAT)
        matmul.load(pathname)
        assert np.array_equal(backup_W, matmul.W), \
            "expected \n%s\n actual \n%s\n" % (backup_W, matmul.W)

        path = pathlib.Path(pathname)
        path.unlink()

        Y = matmul.function(X)
        matmul.gradient(Y)
        matmul.update()

    profiler.disable()
    profiler.print_stats(sort="cumtime")
