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
import copy
import logging
from typing import (
    Union
)

import numpy as np

import common.weights as weights
from common.functions import (
    numerical_jacobian,
)
from common.utilities import (
    random_string
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
            "name": "test_020_matmul_builder_to_fail_matmul_spec",
            "num_nodes": M,
            "num_features": D,
            "weight": {
                "scheme": "he",
                "num_nodes": M,
                "num_features": D + 1
            },
            "log_level": logging.ERROR
        }
        try:
            Matmul.build(valid_matmul_spec)
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["name"] = ""
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid name")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["num_nodes"] = np.random.randint(-100, 0)
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with num_nodes <=0")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["num_features"] = np.random.randint(-100, 0)
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
            "name": "test_020_matmul_builder_to_fail_matmul_spec",
            "num_nodes": M,
            "num_features": D,
            "weight": {
                "scheme": "he",
                "num_nodes": M,
                "num_features": D + 1
            }
        }
        try:
            Matmul.build(valid_matmul_spec)
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["weight"]["scheme"] = "invalid_scheme"
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid weight scheme")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["weight"]["num_nodes"] = "hoge"
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with weight.shape != (M, D+1) scheme")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["weight"]["num_nodes"] = M+1
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with weight.shape != (M, D+1) scheme")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["weight"]["num_features"] = D
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with weight.shape != (M, D+1) scheme")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["weight"]["num_features"] = "hoge"
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with weight.shape != (M, D+1) scheme")
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
            "name": "test_020_matmul_builder_to_fail_matmul_spec",
            "num_nodes": M,
            "num_features": D,
            "weight": {
                "scheme": "he",
                "num_nodes": M,
                "num_features": D + 1
            },
            "optimizer": {
                "scheme": "sGd",
                "parameters": {
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
        matmul_spec["optimizer"] = ""
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid optimizer spec")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["optimizer"]["scheme"] = "invalid"
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid optimizer spec")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["optimizer"]["parameters"]["lr"] = np.random.uniform(-1, 0)
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid lr value")
        except AssertionError:
            pass

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["optimizer"]["parameters"]["l2"] = np.random.uniform(-1, 0)
        try:
            Matmul.build(matmul_spec)
            raise RuntimeError("Matmul.build() must fail with invalid l2 value")
        except AssertionError:
            pass

    profiler.disable()
    profiler.print_stats(sort="cumtime")


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
            "name": "test_020_matmul_builder_to_fail_matmul_spec",
            "num_nodes": M,
            "num_features": D,
            "weight": {
                "scheme": "he",
                "num_nodes": M,
                "num_features": D + 1
            },
            "optimizer": {
                "scheme": "sGd",
                "parameters": {
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
        matmul_spec["optimizer"]["scheme"] = "sgd"
        try:
            Matmul.build(valid_matmul_spec)
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_matmul_spec)

        matmul_spec = copy.deepcopy(valid_matmul_spec)
        matmul_spec["optimizer"]["scheme"] = "SGD"
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
                "name": "test_020_matmul_builder_to_fail_matmul_spec",
                "num_nodes": M,
                "num_features": D,
                "weight": {
                    "scheme": "he",
                    "num_nodes": M,
                    "num_features": D + 1
                },
                "optimizer": {
                    "scheme": "sGd"
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

        Y = matmul.function(X)
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

        # Constraint 6.: W has been updated by the gradient descent.
        assert np.any(backup != matmul.W), "W has not been updated "

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
        assert np.all(np.abs(objective(matmul.function(X)) < L))

    profiler.disable()
    profiler.print_stats(sort="cumtime")
