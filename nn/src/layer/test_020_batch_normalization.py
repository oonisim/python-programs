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

import numpy as np

from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
)
from common.function import (
    standardize,
)
from common.utility import (
    random_string
)
from layer import (
    BatchNormalization
)
from layer.constants import (
    _OPTIMIZER,
    _PARAMETERS,
    _LOG_LEVEL
)
from layer.frederik_kratzert import (
    batchnorm_forward,
    batchnorm_backward
)
from optimizer import (
    Optimizer
)
from testing.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    GRADIENT_DIFF_ACCEPTANCE_VALUE
)

Logger = logging.getLogger(__name__)


def test_020_bn_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the initialization parameter constraints.
    Expected:
        Initialization detects parameter constraints not meet and fails.
    """
    name = "test_020_bn_instantiation_to_fail"
    for _ in range(NUM_MAX_TEST_TIMES):
        M: int = np.random.randint(1, NUM_MAX_NODES)
        # Constraint: Name is string with length > 0.
        try:
            BatchNormalization(
                name="",
                num_nodes=1
            )
            raise RuntimeError("BN initialization with invalid name must fail")
        except AssertionError:
            pass

        # Constraint: num_nodes > 1
        try:
            BatchNormalization(
                name="test_020_bn",
                num_nodes=0
            )
            raise RuntimeError("BatchNormalization(num_nodes<1) must fail.")
        except AssertionError:
            pass

        # Constraint: logging level is correct.
        try:
            BatchNormalization(
                name="test_020_bn",
                num_nodes=M,
                log_level=-1
            )
            raise RuntimeError("BN initialization with invalid log level must fail")
        except (AssertionError, KeyError):
            pass

        # Constraint: Momentum is TYPE_FLOAT and 0 < momentum < 1.
        layer = BatchNormalization(
            name="test_020_bn",
            num_nodes=1
        )
        assert \
            isinstance(layer.momentum, TYPE_FLOAT) and \
            TYPE_FLOAT(0.0) < layer.momentum < TYPE_FLOAT(1.0)

        try:
            BatchNormalization(
                name="test_020_bn",
                num_nodes=1,
                momentum=TYPE_FLOAT(np.random.uniform(-1, 0))
            )
            raise RuntimeError("BN initialization with momentum <=0 must fail")
        except AssertionError:
            pass

        try:
            BatchNormalization(
                name="test_020_bn",
                num_nodes=1,
                momentum=TYPE_FLOAT(np.random.randint(1, 100))
            )
            raise RuntimeError("BN initialization with momentum > 1 must fail")
        except AssertionError:
            pass

        # Constraint: 0 < eps < 1e-3.
        try:
            BatchNormalization(
                name="test_020_bn",
                num_nodes=np.random.randint(1, 100),
                eps=TYPE_FLOAT(np.random.uniform(-100.0, 0))
            )
            raise RuntimeError("BN initialization with eps < 0 must fail")
        except AssertionError:
            pass
        try:
            BatchNormalization(
                name="test_020_bn",
                num_nodes=np.random.randint(1, 100),
                eps=TYPE_FLOAT(np.random.uniform(1e-3, 100.0))
            )
            raise RuntimeError("BN initialization with eps >=1e-3 must fail")
        except AssertionError:
            pass


def test_020_bn_instance_properties_access_to_fail():
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
        layer = BatchNormalization(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not layer.name == name:
                raise RuntimeError("layer.name == name should be true")
        except AssertionError:
            raise RuntimeError("Access to name should be allowed as already initialized.")

        try:
            if not layer.M == M:
                raise RuntimeError("layer.M == M should be true")
        except AssertionError:
            raise RuntimeError("Access to M should be allowed as already initialized.")

        try:
            if not isinstance(layer.logger, logging.Logger):
                raise RuntimeError("isinstance(layer.logger, logging.Logger) should be true")
        except AssertionError:
            raise RuntimeError("Access to logger should be allowed as already initialized.")

        assert isinstance(layer.optimizer, Optimizer), \
            "Access to optimizer should be allowed as already initialized."

        # --------------------------------------------------------------------------------
        # To fail
        # --------------------------------------------------------------------------------
        try:
            print(layer.X)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(layer.N)
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
            print(layer.Xmd)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(layer.dXmd01)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(layer.dXmd02)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(layer.Xstd)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(layer.dXstd)
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
            # pylint: disable=not-callable
            layer.objective(np.array(1.0, dtype=TYPE_FLOAT))
            raise RuntimeError(msg)
        except AssertionError:
            pass


def test_020_bn_instance_properties_access_to_succeed():
    """
    Objective:
        Verify the layer class instance has initialized its properties.
    Expected:
        Layer parameter access to succeed
    """

    def objective(x: np.ndarray):
        """Dummy objective function"""
        return np.sum(x)

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(1, NUM_MAX_NODES)
        layer = BatchNormalization(
            name=name,
            num_nodes=M,
            log_level=logging.DEBUG
        )
        layer.objective = objective

        assert layer.name == name
        assert layer.num_nodes == M

        assert \
            layer.gamma.dtype == TYPE_FLOAT and \
            layer.gamma.shape == (M,) and \
            np.all(layer.gamma == np.ones(M, dtype=TYPE_FLOAT))

        assert \
            layer.dGamma.dtype == TYPE_FLOAT and \
            layer.dGamma.shape == (M,)

        assert \
            layer.beta.dtype == TYPE_FLOAT and \
            layer.beta.shape == (M,) and \
            np.all(layer.beta == np.zeros(M, dtype=TYPE_FLOAT))

        assert \
            layer.dBeta.dtype == TYPE_FLOAT and \
            layer.dBeta.shape == (M,)

        assert \
            layer.U.dtype == TYPE_FLOAT and \
            layer.U.shape == (M,)

        assert \
            layer.dU.dtype == TYPE_FLOAT and \
            layer.dU.size == M

        assert \
            layer.dV.dtype == TYPE_FLOAT and \
            layer.dV.size == M

        assert \
            layer.SD.dtype == TYPE_FLOAT and \
            layer.SD.shape == (M,)

        assert \
            layer.norm.dtype == TYPE_FLOAT and \
            layer.norm.shape == (M,)

        assert \
            layer.RU.dtype == TYPE_FLOAT and \
            layer.RU.shape == (M,)

        assert \
            layer.RSD.dtype == TYPE_FLOAT and \
            layer.RSD.shape == (M,)

        assert layer.objective == objective


def test_020_bn_builder_to_succeed():
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
        valid_bn_parameters = BatchNormalization.specification_template()[_PARAMETERS]
        eps = TYPE_FLOAT(valid_bn_parameters["eps"])
        momentum = TYPE_FLOAT(valid_bn_parameters["momentum"])
        lr: TYPE_FLOAT = TYPE_FLOAT(valid_bn_parameters[_OPTIMIZER][_PARAMETERS]["lr"])
        l2: TYPE_FLOAT = TYPE_FLOAT(valid_bn_parameters[_OPTIMIZER][_PARAMETERS]["l2"])
        log_level: int = valid_bn_parameters[_LOG_LEVEL]
        try:
            bn: BatchNormalization = BatchNormalization.build(parameters=valid_bn_parameters)
            assert bn.optimizer.lr == lr
            assert bn.optimizer.l2 == l2
            assert bn.logger.getEffectiveLevel() == log_level
            assert bn.eps == eps
            assert bn.momentum == momentum
        except Exception as e:
            raise RuntimeError("Matmul.build() must succeed with %s" % valid_bn_parameters) from e

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_bn_function_method_to_fail():
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
        M: int = np.random.randint(2, NUM_MAX_NODES)

        momentum = TYPE_FLOAT(0.85)
        try:
            layer = BatchNormalization(
                name=name,
                num_nodes=M,
                momentum=momentum,
                log_level=logging.DEBUG
            )
            layer.function(int(1))
            raise RuntimeError("Invoke layer.function(int(1)) must fail.")
        except AssertionError:
            pass

        try:
            layer = BatchNormalization(
                name=name,
                num_nodes=M,
                momentum=momentum,
                log_level=logging.DEBUG
            )
            layer.gradient(int(1))
            raise RuntimeError("Invoke layer.gradient(int(1)) must fail.")
        except AssertionError:
            pass


def _validate_storage_allocation(layer, X):
    assert \
        layer.N == X.shape[0]
    assert \
        layer.Y.shape == X.shape and layer.Y.dtype == TYPE_FLOAT
    assert \
        layer.dX.shape == X.shape and layer.dX.dtype == TYPE_FLOAT
    assert \
        layer.Xstd.shape == X.shape and layer.Xstd.dtype == TYPE_FLOAT
    assert \
        layer.Xmd.shape == X.shape and layer.Xmd.dtype == TYPE_FLOAT
    assert \
        layer.dXstd.shape == X.shape and layer.dXstd.dtype == TYPE_FLOAT
    assert \
        layer.dXmd01.shape == X.shape and layer.dXmd01.dtype == TYPE_FLOAT
    assert \
        layer.dXmd02.shape == X.shape and layer.dXmd02.dtype == TYPE_FLOAT


def _validate_layer_values(layer, X, eps):
    ddof = 1 if X.shape[0] > 1 else 0

    # ----------------------------------------------------------------------
    # Currently in standardize(), sd[sd==0.0] = 1.0 is implemented.
    # ----------------------------------------------------------------------
    md = X - X.mean(axis=0)     # md = mean deviation
    variance = X.var(axis=0, ddof=ddof)
    if eps > TYPE_FLOAT(0.0):
        sd = np.sqrt(variance + eps, dtype=TYPE_FLOAT)
    else:
        sd = np.std(X, axis=0, ddof=ddof, dtype=TYPE_FLOAT)
        sd[sd == TYPE_FLOAT(0.0)] = TYPE_FLOAT(1.0)

    expected_standardized = md / sd
    diff = expected_standardized - layer.Xstd

    assert np.allclose(layer.U, X.mean(axis=0), atol=TYPE_FLOAT(1e-6), rtol=TYPE_FLOAT(0.0))
    assert np.allclose(layer.Xmd, md, atol=TYPE_FLOAT(1e-6), rtol=TYPE_FLOAT(0.0))
    assert np.allclose(layer.SD, sd, atol=TYPE_FLOAT(1e-6), rtol=TYPE_FLOAT(0.0))
    assert np.allclose(
        layer.Xstd,
        expected_standardized,
        atol=TYPE_FLOAT(1e-6),
        rtol=TYPE_FLOAT(0.0)
    ), "Xstd\n%s\nexpected_standardized=\n%s\ndiff=\n%s\n" \
       % (layer.Xstd, expected_standardized, diff)


def _validate_layer_running_statistics(
        layer: BatchNormalization, previous_ru, previous_rsd, X, eps
):
    momentum = layer.momentum
    ddof = 1 if X.shape[0] > 1 else 0

    if layer.total_training_invocations == 1:
        assert np.all(layer.RU == layer.U)
        assert np.all(layer.RSD == layer.SD)
        assert layer.total_training_invocations * layer.N == layer.total_rows_processed
    else:
        # ----------------------------------------------------------------------
        # Currently in standardize(), sd[sd==0.0] = 1.0 is implemented.
        # ----------------------------------------------------------------------
        variance = X.var(axis=0, ddof=ddof)
        if eps > TYPE_FLOAT(0.0):
            sd = np.sqrt(variance + eps)
        else:
            sd = np.std(X, axis=0, ddof=ddof)
            sd[sd == TYPE_FLOAT(0.0)] = TYPE_FLOAT(1.0)

        expected_ru = momentum * previous_ru + (TYPE_FLOAT(1) - momentum) * X.mean(axis=0)
        expected_rsd = momentum * previous_rsd + (TYPE_FLOAT(1) - momentum) * sd
        assert np.allclose(layer.RU, expected_ru, atol=TYPE_FLOAT(1e-6), rtol=TYPE_FLOAT(0))
        assert \
            np.allclose(layer.RSD, expected_rsd, atol=TYPE_FLOAT(1e-6), rtol=TYPE_FLOAT(0)), \
            "X=\n%s\nX.sd()=\n%s\nlayer.SD=\n%s\nlayer.RSD=\n%s\n" \
            % (X, X.std(axis=0, ddof=ddof), layer.SD, layer.RSD)


def test_020_bn_method_function_to_succeed():
    """
    Objective:
        Verify the layer class instance function method
    Expected:
        Layer method calculate expected values.
    """
    def objective(x: np.ndarray):
        """Dummy objective function"""
        return np.sum(x)

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

        X = np.random.rand(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        if np.random.uniform() < 0.5:
            eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-10))
        else:
            eps = TYPE_FLOAT(0.0)

        layer = BatchNormalization(
            name=name,
            num_nodes=M,
            momentum=momentum,
            eps=eps,
            log_level=logging.DEBUG
        )
        layer.objective = objective

        # ********************************************************************************
        # Constraint: total_rows_processed = times_of_invocations * N
        # ********************************************************************************
        assert layer.total_rows_processed == 0
        ru = layer.RU
        rsd = layer.RSD
        layer.function(
            X,
            numexpr_enabled=numexpr_enabled,
            numba_enabled=numba_enabled
        )
        _validate_layer_values(layer, X, eps=eps)
        _validate_layer_running_statistics(
            layer=layer, previous_ru=ru, previous_rsd=rsd, X=X, eps=eps
        )

        # ********************************************************************************
        # Constraint:
        #   layer.N provides the latest X.shape[0]
        #   X related arrays should have its storage allocated and has the X.shape.
        #   * dX
        #   * dXmd01
        #   * dXmd02
        # ********************************************************************************
        assert layer.N == X.shape[0]
        assert \
            layer.dX.dtype == TYPE_FLOAT and \
            layer.dX.shape == (N, M)

        assert \
            layer.dXmd01.dtype == TYPE_FLOAT and \
            layer.dXmd01.shape == (N, M)

        assert \
            layer.dXmd02.dtype == TYPE_FLOAT and \
            layer.dXmd02.shape == (N, M)
        assert layer.total_rows_processed == N

        # ********************************************************************************
        # Constraint: total_rows_processed = times_of_invocations * N
        # ********************************************************************************
        for i in range(np.random.randint(1, 100)):
            layer.function(
                X,
                numexpr_enabled=numexpr_enabled,
                numba_enabled=numba_enabled
            )
            assert layer.total_rows_processed == TYPE_INT(N * (i + 2))

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_bn_method_function_multi_invocations_to_succeed():
    """
    Objective:
        Verify the layer class instance function method
    Expected:
        Layer method calculate expected values.
    """
    def objective(x: np.ndarray):
        """Dummy objective function"""
        return np.sum(x)

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

        X = np.random.rand(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        if np.random.uniform() < 0.5:
            eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-10))
        else:
            eps = TYPE_FLOAT(0.0)

        # ********************************************************************************
        # Constraint:
        #   layer needs to reallocate X related storages upon X.shape[0] change.
        # ********************************************************************************
        layer = BatchNormalization(
            name=name,
            num_nodes=M,
            momentum=momentum,
            eps=eps,
            log_level=logging.DEBUG
        )
        layer.objective = objective

        for i in range(np.random.randint(1, 100)):
            layer.function(
                X,
                numexpr_enabled=numexpr_enabled,
                numba_enabled=numba_enabled
            )

        total_rows_processed = layer.total_rows_processed
        ru = layer.RU
        rsd = layer.RSD

        while True:
            Z = np.random.rand(np.random.randint(1, NUM_MAX_BATCH_SIZE), M).astype(TYPE_FLOAT)
            if Z.shape[0] != N:
                break

        layer.function(
            Z,
            numexpr_enabled=numexpr_enabled,
            numba_enabled=numba_enabled
        )

        # ********************************************************************************
        # Constraint: Xsd, U, Xmd, SD should match those of Z
        # ********************************************************************************
        _validate_storage_allocation(layer, Z)
        _validate_layer_values(layer, Z, eps=eps)

        # ********************************************************************************
        # Constraint: Statistics is updated with Z
        # ********************************************************************************
        assert layer.total_rows_processed == total_rows_processed + Z.shape[0]
        _validate_layer_running_statistics(
            layer=layer, previous_ru=ru, previous_rsd=rsd, X=Z, eps=eps
        )

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_bn_method_function_validate_with_frederik_kratzert():
    """
    Objective:
        Verify the layer class instance function method calculates expected values
    Expected:
        Layer method calculate expected values.
    """
    def objective(x: np.ndarray):
        """Dummy objective function"""
        return np.sum(x)

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

        X = np.random.rand(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        if np.random.uniform() < 0.5:
            eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-10))
        else:
            eps = TYPE_FLOAT(0.0)

        layer = BatchNormalization(
            name=name,
            num_nodes=M,
            momentum=momentum,
            eps=eps,
            log_level=logging.DEBUG
        )
        layer.objective = objective

        u = GRADIENT_DIFF_ACCEPTANCE_VALUE
        out, cache = batchnorm_forward(
            x=X, gamma=layer.gamma, beta=layer.beta, eps=eps
        )
        xhat, gamma, xmu, norm, sd, var, eps = cache

        Y = layer.function(
            X,
            numexpr_enabled=numexpr_enabled,
            numba_enabled=numba_enabled
        )

        # ********************************************************************************
        # Constraint: Xsd, X-U, Xmd, SD should match those of frederik_kratzert
        # ********************************************************************************
        assert np.allclose(Y, out, atol=u), \
            "Y=\n%s\nout=\n%s\ndiff=\n%s\n" \
            % (Y, out, (out-Y))

        assert np.allclose(layer.Xmd, xmu, atol=u), \
            "Xmd=\n%s\nxmu=\n%s\ndiff=\n%s\n" \
            % (layer.Xmd, xmu, (xmu-layer.Xmd))

        assert np.allclose(layer.SD, sd, atol=u), \
            "SD=\n%s\nsd=\n%s\ndiff=\n%s\n" \
            % (layer.SD, sd, (sd-layer.SD))

        assert np.allclose(layer.Xstd, xhat, atol=u), \
            "Xstd=\n%s\nxhat=\n%s\ndiff=\n%s\n" \
            % (layer.Xstd, xhat, (xhat-layer.Xstd))

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_bn_method_gradient_validate_with_frederik_kratzert():
    """
    Objective:
        Verify the layer class instance gradient method calculates expected values
    Expected:
        Layer method calculate expected values.
    """
    if TYPE_FLOAT == np.float32:
        # TODO:
        "Need to investigate/redesign for 32 bit floating"
        return

    def objective(x: np.ndarray):
        """Dummy objective function"""
        return np.sum(x)

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

        X = np.random.rand(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        if np.random.uniform() < 0.5:
            eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-10))
        else:
            eps = TYPE_FLOAT(0.0)

        _layer = BatchNormalization(
            name=name,
            num_nodes=M,
            momentum=momentum,
            eps=eps,
            log_level=logging.DEBUG
        )
        _layer.objective = objective

        u = GRADIENT_DIFF_ACCEPTANCE_VALUE
        dout = np.ones(X.shape)

        # --------------------------------------------------------------------------------
        # Benchmark (frederik_kratzert)
        # --------------------------------------------------------------------------------
        out, cache = batchnorm_forward(
            x=X, gamma=_layer.gamma, beta=_layer.beta, eps=eps
        )
        xhat, gamma, xmu, norm, sd, var, eps = cache
        dx, dgamma, dbeta, dxhat, dvar, dxmu2, dxmu1, dmu = batchnorm_backward(dout, cache)

        # ********************************************************************************
        # Constraint: layer gradients should match those of frederik_kratzert
        # ********************************************************************************
        _layer.function(
            X,
            numexpr_enabled=numexpr_enabled,
            numba_enabled=numba_enabled
        )
        _layer.gradient(
            dY=_layer.tensor_cast(dout, TYPE_FLOAT),
            numexpr_enabled=numexpr_enabled,
            numba_enabled=numba_enabled
        )
        assert np.allclose(_layer.dGamma, dgamma, atol=u), \
            "dGamma=\n%s\ndgamma=\n%s\ndiff=\n%s\n" \
            % (_layer.dGamma, dgamma, (dgamma-_layer.dGamma))

        assert np.allclose(_layer.dBeta, dbeta, atol=u), \
            "dBeta=\n%s\ndbeta=\n%s\ndiff=\n%s\n" \
            % (_layer.dBeta, dbeta, (dbeta - _layer.dBeta))

        assert np.allclose(_layer.dXstd, dxhat, atol=u), \
            "dXstd=\n%s\ndxhat=\n%s\ndiff=\n%s\n" \
            % (_layer.dXstd, dxhat, (dxhat - _layer.dXstd))

        assert np.allclose(_layer.dV, dvar, atol=u), \
            "dV=\n%s\ndvar=\n%s\ndiff=\n%s\n" \
            % (_layer.dV, dvar, (dvar - _layer.dV))

        assert np.allclose(_layer.dXmd01, dxmu2, atol=u), \
            "dXmd01=\n%s\ndxmu2=\n%s\ndiff=\n%s\n" \
            % (_layer.dXmd01, dxmu2, (dxmu2 - _layer.dXmd01))

        assert np.allclose(_layer.dXmd02, dxmu1, atol=u), \
            "dXmd02=\n%s\ndxmu1=\n%s\ndiff=\n%s\n" \
            % (_layer.dXmd02, dxmu1, (dxmu1 - _layer.dXmd02))

        assert np.allclose(_layer.dU, dmu, atol=u), \
            "dU=\n%s\ndmu=\n%s\ndiff=\n%s\n" \
            % (_layer.dU, dmu, (dmu - _layer.dU))

        assert np.allclose(_layer.dX, dx, atol=u), \
            "dX=\n%s\ndx=\n%s\ndiff=\n%s\n" \
            % (_layer.dX, dx, (dx - _layer.dX))

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_020_bn_method_gradient_descent():
    """
    Objective:
        Verify the gradient descent
    Expected:
        The objective decrease with the descents.
    """
    if TYPE_FLOAT == np.float32:
        # TODO:
        "Need to investigate/redesign for 32 bit floating"
        return

    def objective(x: np.ndarray):
        """Dummy objective function"""
        return np.sum(x)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        numexpr_enabled = bool(np.random.randint(0, 2))

        # For BN which works on statistics on per-feature basis,
        # no sense if M = 1 or N = 1.
        N: int = np.random.randint(2, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)

        # DO not use np.random.rand as test fails for 32 bit float
        X = np.random.rand(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        if np.random.uniform() < 0.5:
            eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-10))
        else:
            eps = TYPE_FLOAT(0.0)
        layer = BatchNormalization(
            name=name,
            num_nodes=M,
            momentum=momentum,
            eps=eps,
            log_level=logging.DEBUG
        )
        layer.objective = objective

        u = GRADIENT_DIFF_ACCEPTANCE_VALUE
        for _ in range(np.random.randint(1, 10)):
            dout = np.random.uniform(-1, 1, size=X.shape).astype(TYPE_FLOAT)

            Y = layer.function(
                X,
                numexpr_enabled=numexpr_enabled,
            )
            # pylint: disable=not-callable
            layer.objective(Y)
            layer.gradient(
                dY=dout,
                numexpr_enabled=numexpr_enabled,
            )
            dGamma, dBeta = layer.update()

            # ********************************************************************************
            # Constraint: expected gradients match with actual
            # ********************************************************************************
            expected_dGamma = np.sum(dout * layer.Xstd, axis=0)
            expected_dBeta = np.sum(dout, axis=0)
            assert np.allclose(expected_dGamma, dGamma, atol=u), \
                "Need dGamma\n%s\nbut\n%s\ndiff=\n%s\n" \
                % (expected_dGamma, dGamma, expected_dGamma-dGamma)
            assert np.allclose(expected_dBeta, dBeta, atol=u), \
                "Need dBeta\n%s\nbut\n%s\ndiff=\n%s\n" \
                % (expected_dBeta, dBeta, expected_dBeta-dBeta)


def test_020_bn_method_predict():
    """
    Objective:
        Verify the prediction function
    Expected:
        The objective
    """
    def objective(x: np.ndarray):
        """Dummy objective function"""
        return np.sum(x)

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

        X = np.random.rand(N, M).astype(TYPE_FLOAT)
        momentum = TYPE_FLOAT(np.random.uniform(0.7, 0.99))
        if np.random.uniform() < 0.5:
            eps = TYPE_FLOAT(np.random.uniform(1e-12, 1e-8))
        else:
            eps = TYPE_FLOAT(0.0)

        layer = BatchNormalization(
            name=name,
            num_nodes=M,
            momentum=momentum,
            eps=eps,
            log_level=logging.DEBUG
        )
        layer.objective = objective
        Y = layer.function(
            X,
            numexpr_enabled=numexpr_enabled,
            numba_enabled=numba_enabled
        )
        # ********************************************************************************
        # Constraint: With only 1 invocation, predict should be the same with Y.
        # RU = momentum * RU + (1 - momentum) * U
        # After the 1st invocation, RU==U. Then momentum * U + (1 - momentum) * U -> U
        # ********************************************************************************
        assert np.allclose(Y, layer.predict(X), atol=TYPE_FLOAT(1e-9), rtol=TYPE_FLOAT(0))

        # ********************************************************************************
        # Constraint: At 2nd invocation, predict should be the same with
        #
        # ********************************************************************************
        Z = np.random.rand(N, M).astype(TYPE_FLOAT)
        standardized, mean, sd, deviation = standardize(Z, eps=eps, keepdims=False)
        expected_RU = layer.RU * momentum + mean * (TYPE_FLOAT(1)-momentum)
        expected_RSD = layer.RSD * momentum + sd * (TYPE_FLOAT(1)-momentum)
        layer.function(
            Z,
            numexpr_enabled=numexpr_enabled,
            numba_enabled=numba_enabled
        )
        assert np.allclose(layer.RU, expected_RU, atol=TYPE_FLOAT(1e-10), rtol=TYPE_FLOAT(0))
        assert np.allclose(layer.RSD, expected_RSD, atol=TYPE_FLOAT(1e-10), rtol=TYPE_FLOAT(0))
