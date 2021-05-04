"""_layer test cases
Batch X: shape(N, D):
--------------------
X is the input data into a _layer, hence it does NOT include the bias.

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
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_TENSOR
)
from common.function import (
    standardize,
)
from common.utility import (
    random_string
)
from layer import (
    Standardization
)
from layer.constants import (
    _PARAMETERS,
    _LOG_LEVEL
)
from testing.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE
)

Logger = logging.getLogger(__name__)


def _must_fail(
        name: str,
        num_nodes: int,
        momentum: TYPE_FLOAT = 0.9,
        eps=TYPE_FLOAT(0),
        log_level: int = logging.ERROR,
        msg: str = ""
):
    assert msg
    try:
        Standardization(
            name=name,
            num_nodes=num_nodes,
            momentum=TYPE_FLOAT(momentum),
            eps=TYPE_FLOAT(eps),
            log_level=log_level
        )
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
        # Constraint: Name is string with length > 0.
        msg = "initialization with invalid name must fail"
        _must_fail(name="", num_nodes=M, msg=msg)

        # Constraint: num_nodes > 1
        msg = "Standardization(num_nodes<1) must fail."
        _must_fail(name=name, num_nodes=0, msg=msg)

        # Constraint: logging level is correct.
        msg = "initialization with invalid log level must fail"
        _must_fail(name=name, num_nodes=0, log_level=-1, msg=msg)

        # Constraint: Momentum is TYPE_FLOAT and 0 < momentum < 1.
        msg = "default momentum must be 0 < momentum < 1"
        _layer = Standardization(
            name="test_020_std",
            num_nodes=M
        )
        assert \
            isinstance(_layer.momentum, TYPE_FLOAT) and \
            TYPE_FLOAT(0.0) < _layer.momentum < TYPE_FLOAT(1.0), msg

        msg = "initialization with momentum <=0 must fail"
        _must_fail(name=name, num_nodes=M, momentum=TYPE_FLOAT(np.random.uniform(-1, 0)), msg=msg)

        msg = "initialization with momentum > 1 must fail"
        _must_fail(name=name, num_nodes=M, momentum=TYPE_FLOAT(np.random.randint(1, 100)), msg=msg)

        # Constraint: 0 < eps < 1e-3.
        msg = "initialization with eps < 0 must fail"
        _must_fail(name=name, num_nodes=M, eps=TYPE_FLOAT(np.random.uniform(-100.0, 0)), msg=msg)

        msg = "initialization with eps >=1e-3 must fail"
        _must_fail(name=name, num_nodes=M, eps=TYPE_FLOAT(np.random.uniform(1e-3, 100.0)), msg=msg)


def test_020_std_instance_properties_access_to_fail():
    """
    Objective:
        Verify the _layer class validates the parameters have been initialized before accessed.
    Expected:
        Initialization detects the access to the non-initialized parameters and fails.
    """
    msg = "Accessing uninitialized property of the _layer must fail."
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(1, NUM_MAX_NODES)
        _layer = Standardization(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not _layer.name == name:
                raise RuntimeError("_layer.name == name should be true")
        except AssertionError:
            raise RuntimeError("Access to name should be allowed as already initialized.")

        try:
            if not _layer.M == M:
                raise RuntimeError("_layer.M == M should be true")
        except AssertionError:
            raise RuntimeError("Access to M should be allowed as already initialized.")

        try:
            if not isinstance(_layer.logger, logging.Logger):
                raise RuntimeError("isinstance(_layer.logger, logging.Logger) should be true")
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
            print(_layer.Xmd)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.dXmd01)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(_layer.dXmd02)
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
            _layer.objective(np.array(1.0, dtype=TYPE_FLOAT))
            raise RuntimeError(msg)
        except AssertionError:
            pass


def test_020_std_instance_properties_access_to_succeed():
    """
    Objective:
        Verify the _layer class instance has initialized its properties.
    Expected:
        Layer parameter access to succeed
    """

    def objective(X: TYPE_TENSOR):
        """Dummy objective function"""
        return np.sum(X)

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(1, NUM_MAX_NODES)
        _layer = Standardization(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        _layer.objective = objective

        assert _layer.name == name
        assert _layer.num_nodes == M

        assert \
            _layer.U.dtype == TYPE_FLOAT and \
            _layer.U.shape == (M,)

        assert \
            _layer.dU.dtype == TYPE_FLOAT and \
            _layer.dU.size == M

        assert \
            _layer.dV.dtype == TYPE_FLOAT and \
            _layer.dV.size == M

        assert \
            _layer.SD.dtype == TYPE_FLOAT and \
            _layer.SD.shape == (M,)

        assert \
            _layer.norm.dtype == TYPE_FLOAT and \
            _layer.norm.shape == (M,)

        assert \
            _layer.RU.dtype == TYPE_FLOAT and \
            _layer.RU.shape == (M,)

        assert \
            _layer.RSD.dtype == TYPE_FLOAT and \
            _layer.RSD.shape == (M,)

        assert _layer.objective == objective


def test_020_std_builder_to_succeed():
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
        valid_std_parameters = Standardization.specification_template()[_PARAMETERS]
        eps = TYPE_FLOAT(valid_std_parameters["eps"])
        momentum = TYPE_FLOAT(valid_std_parameters["momentum"])
        log_level = valid_std_parameters[_LOG_LEVEL]
        try:
            std: Standardization = Standardization.build(parameters=valid_std_parameters)
            assert std.logger.getEffectiveLevel() == log_level
            assert std.eps == eps
            assert std.momentum == momentum
        except Exception as e:
            raise RuntimeError(
                "Matmul.build() must succeed with %s" % valid_std_parameters
            ) from e

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def _instance(
        name,
        num_nodes: int,
        momentum: TYPE_FLOAT,
        eps: TYPE_FLOAT = TYPE_FLOAT(0),
        log_level: int = logging.ERROR
):
    return Standardization(
        name=name,
        num_nodes=num_nodes,
        momentum=TYPE_FLOAT(momentum),
        eps=TYPE_FLOAT(eps),
        log_level=log_level
    )


def test_020_std_function_method_to_fail():
    """
    Objective:
        Verify the _layer class instance function validates invalid inputs
    Expected:
        Layer method fails.
    """
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))

        # For which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        M: int = np.random.randint(2, NUM_MAX_NODES)
        momentum = TYPE_FLOAT(0.85)

        try:
            _layer = _instance(name=name, num_nodes=M, momentum=momentum)
            _layer.function(int(1))
            raise RuntimeError("Invoke _layer.function(int(1)) must fail.")
        except AssertionError:
            pass

        try:
            _layer = _instance(name=name, num_nodes=M, momentum=momentum)
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
        _layer.Xmd.shape == X.shape and _layer.Xmd.dtype == TYPE_FLOAT
    assert \
        _layer.dXmd01.shape == X.shape and _layer.dXmd01.dtype == TYPE_FLOAT
    assert \
        _layer.dXmd02.shape == X.shape and _layer.dXmd02.dtype == TYPE_FLOAT


def _validate_layer_values(_layer: Standardization, X, eps):
    ddof = 1 if X.shape[0] > 1 else 0

    # ----------------------------------------------------------------------
    # Currently in standardize(), sd[sd==0.0] = 1.0 is implemented.
    # ----------------------------------------------------------------------
    md = X - X.mean(axis=0)     # md = mean deviation
    variance = X.var(axis=0, ddof=ddof)
    if eps > TYPE_FLOAT(0.0):
        sd = np.sqrt(variance + eps).astype(TYPE_FLOAT)
    else:
        sd = np.std(X, axis=0, ddof=ddof).astype(TYPE_FLOAT)
        sd[sd == TYPE_FLOAT(0.0)] = TYPE_FLOAT(1.0)

    expected_standardized = md / sd
    diff = expected_standardized - _layer.Y

    assert np.allclose(_layer.U, X.mean(axis=0), atol=1e-6, rtol=TYPE_FLOAT(0.0))
    assert np.allclose(_layer.Xmd, md, atol=1e-6, rtol=TYPE_FLOAT(0.0))
    assert np.allclose(_layer.SD, sd, atol=1e-6, rtol=TYPE_FLOAT(0.0))
    assert np.allclose(
        _layer.Y,
        expected_standardized,
        atol=TYPE_FLOAT(1e-6),
        rtol=TYPE_FLOAT(0)
    ), "Xstd\n%s\nexpected_standardized=\n%s\ndiff=\n%s\n" \
       % (_layer.Y, expected_standardized, diff)


def _validate_layer_running_statistics(
        _layer: Standardization, previous_ru, previous_rsd, X, eps
):
    momentum = _layer.momentum
    ddof = 1 if X.shape[0] > 1 else 0

    if _layer.total_training_invocations == 1:
        assert np.all(_layer.RU == _layer.U)
        assert np.all(_layer.RSD == _layer.SD)
        assert _layer.total_training_invocations * _layer.N == _layer.total_rows_processed
    else:
        # ----------------------------------------------------------------------
        # Currently in standardize(), sd[sd==0.0] = 1.0 is implemented.
        # ----------------------------------------------------------------------
        variance = X.var(axis=0, ddof=ddof)
        if eps > TYPE_FLOAT(0.0):
            sd = np.sqrt(variance + eps, dtype=TYPE_FLOAT)
        else:
            sd = np.std(X, axis=0, ddof=ddof, dtype=TYPE_FLOAT)
            sd[sd == TYPE_FLOAT(0.0)] = TYPE_FLOAT(1.0)

        expected_ru = \
            momentum * previous_ru + \
            (TYPE_FLOAT(1.0) - momentum) * X.mean(axis=0)
        expected_rsd = \
            momentum * previous_rsd + (TYPE_FLOAT(1.0) - momentum) * sd
        assert \
            np.allclose(
                _layer.RU,
                expected_ru,
                atol=TYPE_FLOAT(1e-6),
                rtol=TYPE_FLOAT(0)
            )
        assert \
            np.allclose(
                _layer.RSD,
                expected_rsd,
                atol=TYPE_FLOAT(1e-6),
                rtol=TYPE_FLOAT(0)
            ), \
            "X=\n%s\nX.sd()=\n%s\n_layer.SD=\n%s\n_layer.RSD=\n%s\n" \
            % (X, X.std(axis=0, ddof=ddof), _layer.SD, _layer.RSD)


def test_020_std_method_function_to_succeed():
    """
    Objective:
        Verify the _layer class instance function method
    Expected:
        Layer method calculate expected values.
    """
    def objective(x: TYPE_TENSOR):
        """Dummy objective function"""
        return np.sum(x, dtype=TYPE_FLOAT)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        numexpr_enabled = bool(np.random.randint(0, 2))
        numba_enabled = bool(np.random.randint(0, 2))

        # For which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.randn(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-10)) \
            if np.random.uniform() < 0.5 else TYPE_FLOAT(0)
        _layer: Standardization = \
            _instance(name=name, num_nodes=M, momentum=momentum, eps=eps)
        _layer.objective = objective

        # ********************************************************************************
        # Constraint: total_rows_processed = times_of_invocations * N
        # ********************************************************************************
        assert _layer.total_rows_processed == 0
        ru = _layer.RU
        rsd = _layer.RSD
        _layer.function(
            X,
            numexpr_enabled=numexpr_enabled,
        )
        _validate_layer_values(_layer, X, eps=eps)
        _validate_layer_running_statistics(
            _layer=_layer, previous_ru=ru, previous_rsd=rsd, X=X, eps=eps
        )

        # ********************************************************************************
        # Constraint:
        #   _layer.N provides the latest X.shape[0]
        #   X related arrays should have its storage allocated and has the X.shape.
        #   * dX
        #   * dXmd01
        #   * dXmd02
        # ********************************************************************************
        assert _layer.N == X.shape[0]
        assert \
            _layer.dX.dtype == TYPE_FLOAT and \
            _layer.dX.shape == (N, M)

        assert \
            _layer.dXmd01.dtype == TYPE_FLOAT and \
            _layer.dXmd01.shape == (N, M)

        assert \
            _layer.dXmd02.dtype == TYPE_FLOAT and \
            _layer.dXmd02.shape == (N, M)
        assert _layer.total_rows_processed == N

        # ********************************************************************************
        # Constraint: total_rows_processed = times_of_invocations * N
        # ********************************************************************************
        for i in range(np.random.randint(1, 100)):
            _layer.function(
                X,
                numexpr_enabled=numexpr_enabled,
            )
            assert _layer.total_rows_processed == TYPE_INT(N * (i + 2))

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_std_method_function_multi_invocations_to_succeed():
    """
    Objective:
        Verify the _layer class instance function method
    Expected:
        Layer method calculate expected values.
    """
    def objective(x: TYPE_TENSOR):
        """Dummy objective function"""
        return np.sum(x)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        numexpr_enabled = bool(np.random.randint(0, 2))

        # For which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.randn(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        if np.random.uniform() < 0.5:
            eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-10))
        else:
            eps = TYPE_FLOAT(0.0)

        # ********************************************************************************
        # Constraint:
        #   _layer needs to reallocate X related storages upon X.shape[0] change.
        # ********************************************************************************
        _layer: Standardization = \
            _instance(name=name, num_nodes=M, momentum=momentum, eps=eps, log_level=logging.DEBUG)
        _layer.objective = objective

        for i in range(np.random.randint(1, 100)):
            _layer.function(
                X,
                numexpr_enabled=numexpr_enabled,
            )

        total_rows_processed = _layer.total_rows_processed
        ru = _layer.RU
        rsd = _layer.RSD

        while True:
            Z = np.random.randn(np.random.randint(1, NUM_MAX_BATCH_SIZE), M)
            if Z.shape[0] != N:
                break

        _layer.function(
            Z,
            numexpr_enabled=numexpr_enabled,
        )

        # ********************************************************************************
        # Constraint: Properties of Y, U, Xmd, SD should match those of Z
        # ********************************************************************************
        _validate_storage_allocation(_layer, Z)
        _validate_layer_values(_layer, Z, eps=eps)

        # ********************************************************************************
        # Constraint: Statistics is updated with Z
        # ********************************************************************************
        assert _layer.total_rows_processed == total_rows_processed + Z.shape[0]
        _validate_layer_running_statistics(
            _layer=_layer, previous_ru=ru, previous_rsd=rsd, X=Z, eps=eps
        )

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_std_method_predict():
    """
    Objective:
        Verify the prediction function
    Expected:
        The objective
    """
    def objective(x: TYPE_TENSOR):
        """Dummy objective function"""
        return np.sum(x)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        numexpr_enabled = bool(np.random.randint(0, 2))

        # For which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        N: int = np.random.randint(2, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.randn(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        if np.random.uniform() < 0.5:
            eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-8))
        else:
            eps = TYPE_FLOAT(0.0)

        _layer: Standardization = \
            _instance(
                name=name,
                num_nodes=M,
                momentum=momentum,
                eps=eps,
                log_level=logging.DEBUG
            )
        _layer.objective = objective
        Y = _layer.function(
            X,
            numexpr_enabled=numexpr_enabled
        )
        # ********************************************************************************
        # Constraint: With only 1 invocation, predict should be the same with Y.
        # RU = momentum * RU + (1 - momentum) * U
        # After the 1st invocation, RU==U. Then momentum * U + (1 - momentum) * U -> U
        # ********************************************************************************
        assert np.allclose(Y, _layer.predict(X), atol=TYPE_FLOAT(1e-9), rtol=TYPE_FLOAT(0))

        # ********************************************************************************
        # Constraint: At 2nd invocation, predict should be the same with
        #
        # ********************************************************************************
        Z = np.random.randn(N, M).astype(TYPE_FLOAT)
        standardized, mean, sd, deviation = standardize(Z, eps=eps, keepdims=False)
        expected_RU = _layer.RU * momentum + mean * (TYPE_FLOAT(1)-momentum)
        expected_RSD = _layer.RSD * momentum + sd * (TYPE_FLOAT(1)-momentum)
        _layer.function(
            Z,
            numexpr_enabled=numexpr_enabled
        )
        assert np.allclose(_layer.RU, expected_RU, atol=TYPE_FLOAT(1e-10), rtol=TYPE_FLOAT(0))
        assert np.allclose(_layer.RSD, expected_RSD, atol=TYPE_FLOAT(1e-10), rtol=TYPE_FLOAT(0))
