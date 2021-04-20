"""BN layer test cases
Batch X: shape(N, D):
--------------------
X is the input data into a BN layer, hence it does NOT include the bias.

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
import logging
from typing import (
    Union
)

import numpy as np

from common.constant import (
    TYPE_FLOAT,
)
from common.function import (
    standardize,
)
from common.utility import (
    random_string
)
from frederik_kratzert import (
    batchnorm_forward,
    batchnorm_backward
)
from layer import (
    FeatureScaleShift
)
from layer.constants import (
    _OPTIMIZER,
    _PARAMETERS,
    _LOG_LEVEL
)
from optimizer import (
    Optimizer
)
from testing.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE
)

Logger = logging.getLogger(__name__)


def _instance(name, num_nodes: int, log_level: int = logging.ERROR):
    return FeatureScaleShift(
        name=name,
        num_nodes=num_nodes,
        log_level=log_level
    )


def _must_fail(
        name: str,
        num_nodes: int,
        log_level: int = logging.ERROR,
        msg: str = ""
):
    assert msg
    try:
        _instance(name=name, num_nodes=num_nodes, log_level=log_level)
        raise RuntimeError(msg)
    except AssertionError:
        pass


def test_020_std_instantiation_to_fail():
    """
    Objective:
        Verify the _layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_020_std_instantiation_to_fail"
    for _ in range(NUM_MAX_TEST_TIMES):
        M: int = np.random.randint(1, NUM_MAX_NODES)
        D = 1
        # Constraint: Name is string with length > 0.
        msg = "initialization with invalid name must fail"
        _must_fail(name="", num_nodes=M, msg=msg)

        # Constraint: num_nodes > 1
        msg = "FeatureScaleShift(num_nodes<1) must fail."
        _must_fail(name=name, num_nodes=0, msg=msg)

        # Constraint: logging level is correct.
        msg = "initialization with invalid log level must fail"
        _must_fail(name=name, num_nodes=0, log_level=-1, msg=msg)

        # Constraint: Momentum is TYPE_FLOAT and 0 < momentum < 1.
        msg = "default momentum must be 0 < momentum < 1"
        __layer = FeatureScaleShift(
            name="test_020_std",
            num_nodes=M
        )


def test_020_fss_instance_properties_access_to_fail():
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
        _layer = FeatureScaleShift(
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
                raise RuntimeError("isinstance(_layer.logger, logging.Logger) should be true")
        except AssertionError:
            raise RuntimeError("Access to logger should be allowed as already initialized.")

        assert isinstance(_layer.gamma_optimizer, Optimizer), \
            "Access to optimizer should be allowed as already initialized."

        assert isinstance(_layer.beta_optimizer, Optimizer), \
            "Access to optimizer should be allowed as already initialized."

        # --------------------------------------------------------------------------------
        # To fail
        # --------------------------------------------------------------------------------
        try:
            print(_layer.X)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.N)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            _layer.X = int(1)
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
            # pylint: disable=not-callable
            _layer.objective(np.array(1.0))
            raise RuntimeError(msg)
        except AssertionError:
            pass


def test_020_fss_instance_properties_access_to_succeed():
    """
    Objective:
        Verify the layer class instance has initialized its properties.
    Expected:
        Layer parameter access to succeed
    """

    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function"""
        return np.sum(X)

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(1, NUM_MAX_NODES)
        _layer = FeatureScaleShift(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        _layer.objective = objective

        assert _layer.name == name
        assert _layer.num_nodes == M

        assert \
            _layer.gamma.dtype == TYPE_FLOAT and \
            _layer.gamma.shape == (M,) and \
            np.all(_layer.gamma == np.ones(M, dtype=TYPE_FLOAT))

        assert \
            _layer.dGamma.dtype == TYPE_FLOAT and \
            _layer.dGamma.shape == (M,)

        assert \
            _layer.beta.dtype == TYPE_FLOAT and \
            _layer.beta.shape == (M,) and \
            np.all(_layer.beta == np.zeros(M, dtype=TYPE_FLOAT))

        assert \
            _layer.dBeta.dtype == TYPE_FLOAT and \
            _layer.dBeta.shape == (M,)

        assert _layer.objective == objective


def test_020_fss_builder_to_succeed():
    """
    Objective:
        Verify the Matmul.build()
    Expected:
        build() parse the spec and succeed
    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        # ----------------------------------------------------------------------
        # Validate the correct specification.
        # NOTE: Invalidate one parameter at a time from the correct one.
        # Otherwise not sure what you are testing.
        # ----------------------------------------------------------------------
        valid_fss_parameters = FeatureScaleShift.specification_template()[_PARAMETERS]
        lr = valid_fss_parameters[_OPTIMIZER][_PARAMETERS]["lr"]
        l2 = valid_fss_parameters[_OPTIMIZER][_PARAMETERS]["l2"]
        log_level = valid_fss_parameters[_LOG_LEVEL]
        try:
            fss: FeatureScaleShift = FeatureScaleShift.build(parameters=valid_fss_parameters)
            assert fss.gamma_optimizer.lr == lr
            assert fss.gamma_optimizer.l2 == l2
            assert fss.beta_optimizer.lr == lr
            assert fss.beta_optimizer.l2 == l2
            assert fss.logger.getEffectiveLevel() == log_level
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_fss_parameters)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_fss_function_method_to_fail():
    """
    Objective:
        Verify the layer class instance function validates invalid inputs
    Expected:
        Layer method fails.
    """
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))

        # For BN which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.randn(N, M)

        _layer = _instance(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        try:
            _layer.function(int(1))
            raise RuntimeError("Invoke _layer.function(int(1)) must fail.")
        except AssertionError:
            pass

        try:
            _layer.gradient(int(1))
            raise RuntimeError("Invoke _layer.gradient(int(1)) must fail.")
        except AssertionError:
            pass


def _validate_storage_allocation(_layer, X):
    assert \
        _layer.N == X.shape[0]
    assert \
        _layer.Y.shape == X.shape and _layer.Y.dtype == TYPE_FLOAT
    assert \
        _layer.dX.shape == X.shape and _layer.dX.dtype == TYPE_FLOAT
    assert \
        _layer.gamma.shape == (X.shape[1],) and _layer.gamma.dtype == TYPE_FLOAT, \
        f"gamma.shape expected {(X.shape[1],)} but {_layer.gamma.shape}"
    assert \
        _layer.beta.shape == (X.shape[1],) and _layer.beta.dtype == TYPE_FLOAT, \
        f"beta.shape expected {(X.shape[1],)} but {_layer.beta.shape}"
    assert \
        _layer.dGamma.shape == (X.shape[1],) and _layer.dGamma.dtype == TYPE_FLOAT
    assert \
        _layer.dBeta.shape == (X.shape[1],) and _layer.dBeta.dtype == TYPE_FLOAT


def test_020_fss_method_function_to_succeed():
    """
    Objective:
        Verify the layer class instance function method
    Expected:
        Layer method calculate expected values.
    """
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function"""
        return np.sum(X)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        numexpr_enabled = bool(np.random.randint(0, 2))
        numba_enabled = bool(np.random.randint(0, 2))

        # For BN which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.randn(N, M)

        _layer = FeatureScaleShift(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        _layer.objective = objective
        Y = _layer.function(X)

        # ********************************************************************************
        # Constraint:
        #   _layer.N provides the latest X.shape[0]
        #   X related arrays should have its storage allocated and has the X.shape.
        #   * dX
        # ********************************************************************************
        assert _layer.N == X.shape[0]
        assert \
            _layer.dX.dtype == TYPE_FLOAT and \
            _layer.dX.shape == (N, M)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_fss_method_function_multi_invocations_to_succeed():
    """
    Objective:
        Verify the layer class instance function method
    Expected:
        Layer method calculate expected values.
    """
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function"""
        return np.sum(X)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        numexpr_enabled = bool(np.random.randint(0, 2))

        # For BN which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.randn(N, M)

        # ********************************************************************************
        # Constraint:
        #   layer needs to reallocate X related storages upon X.shape[0] change.
        # ********************************************************************************
        _layer = _instance(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        _layer.objective = objective

        for i in range(np.random.randint(1, 100)):
            _layer.function(
                X,
                numexpr_enabled=numexpr_enabled,
            )

        while True:
            Z = np.random.randn(np.random.randint(1, NUM_MAX_BATCH_SIZE), M)
            if Z.shape[0] != N:
                break

        _layer.function(
            Z,
            numexpr_enabled=numexpr_enabled,
        )

        # ********************************************************************************
        # Constraint: gamma, beta should match those of Z
        # ********************************************************************************
        _validate_storage_allocation(_layer, Z)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_fss_method_gradient_descent():
    """
    Objective:
        Verify the gradient descent
    Expected:
        The objective decrease with the descents.
    """
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function"""
        return np.sum(X)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        numexpr_enabled = bool(np.random.randint(0, 2))
        numba_enabled = bool(np.random.randint(0, 2))

        N: int = np.random.randint(2, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.randn(N, M)
        _layer = _instance(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        _layer.objective = objective

        u = 1e-5
        for _ in range(np.random.randint(1, 10)):
            dout = np.random.uniform(-1, 1, size=X.shape)

            Y = _layer.function(
                X,
                numexpr_enabled=numexpr_enabled,
            )
            # pylint: disable=not-callable
            L = _layer.objective(Y)
            G = _layer.gradient(
                dY=dout,
                numexpr_enabled=numexpr_enabled,
            )
            dGamma, dBeta = _layer.update()

            # ********************************************************************************
            # Constraint: expected gradients match with actual
            # ********************************************************************************
            expected_dGamma = np.sum(dout * _layer.X, axis=0)
            expected_dBeta = np.sum(dout, axis=0)
            assert np.allclose(expected_dGamma, dGamma, atol=u), \
                "Need dGamma\n%s\nbut\n%s\ndiff=\n%s\n" \
                % (expected_dGamma, dGamma, expected_dGamma-dGamma)
            assert np.allclose(expected_dBeta, dBeta, atol=u), \
                "Need dBeta\n%s\nbut\n%s\ndiff=\n%s\n" \
                % (expected_dBeta, dBeta, expected_dBeta-dBeta)


def test_020_fss_method_predict():
    """
    Objective:
        Verify the prediction function
    Expected:
        The objective
    """
    # pylint: disable=not-callable
    def objective(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective function"""
        return np.sum(X)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        numexpr_enabled = bool(np.random.randint(0, 2))
        numba_enabled = bool(np.random.randint(0, 2))

        # For BN which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        N: int = np.random.randint(2, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.randn(N, M)

        _layer = _instance(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        _layer.objective = objective
        Y = _layer.function(
            X,
            numexpr_enabled=numexpr_enabled,
        )
        # ********************************************************************************
        # Constraint: With only 1 invocation, predict should be the same with Y.
        # ********************************************************************************
        assert np.allclose(Y, _layer.predict(X), atol=1e-9, rtol=0)
