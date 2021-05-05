"""Network base test cases"""
import cProfile
import copy
import logging

import numpy as np

import layer
import optimizer as optimiser
from common.constant import (
    TYPE_FLOAT,
)
from common.function import (
    softmax,
    relu,
    transform_X_T,
    softmax_cross_entropy_log_loss,
    compose,
)
from . config_test_010_sequential import (
    valid_network_specification_mao,
    valid_network_specification_mamao,
    valid_network_specification_sfmbambamamo,
    _N,
    _M,
    _D,
    _lr,
    _l2,
    invalid_network_specification_with_duplicated_names,
    multilayer_network_specification_bn_to_fail
)

from data import (
    linear_separable_sectors
)
from layer.constants import (
    _OPTIMIZER,
    _NUM_NODES,
    _PARAMETERS,
    _COMPOSITE_LAYER_SPEC
)
from network.sequential import (
    SequentialNetwork
)
from testing.layer import (
    expected_gradient_from_log_loss,
    validate_relu_neuron_round_trip,
    validate_against_expected_loss,
    validate_against_expected_gradient
)
from testing.utilities import (
    build_matmul_relu_objective
)

Logger = logging.getLogger(__name__)


def _must_fail(
        network_specification,
        message
):
    try:
        network = SequentialNetwork.build(
            specification=network_specification,
        )
        raise RuntimeError(message)
    except AssertionError:
        pass


def _must_succeed(
        network_specification,
        message
):
    try:
        network = SequentialNetwork.build(
            specification=network_specification,
        )
        return network
    except Exception as e:
        raise RuntimeError(message)


def test_010_sequential_instantiation_to_fail():
    name = "test_010_base_instantiation_to_fail"

    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    lr = np.random.uniform()
    l2 = np.random.uniform()
    composite_layer_spec = valid_network_specification_mamao[_COMPOSITE_LAYER_SPEC]
    composite_layer_spec["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["lr"] = lr
    composite_layer_spec["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["l2"] = l2
    network = _must_succeed(
        network_specification=valid_network_specification_mamao,
        message="SequentialNetwork() must succeed with %s"
                % valid_network_specification_mamao
    )
    inference_layer: layer.Sequential = network.layer_inference
    matmul_layer: layer.Matmul = inference_layer.layers[0]
    assert matmul_layer.optimizer.lr == lr
    assert matmul_layer.optimizer.l2 == l2

    # ********************************************************************************
    # Constraint: 0 < lr < 1 and 0 < l2 < 1
    # ********************************************************************************
    msg = "SequentialNetwork() must fail with invalid lr value"
    network_specification = copy.deepcopy(valid_network_specification_mamao)
    composite_layer_spec = network_specification[_COMPOSITE_LAYER_SPEC]
    composite_layer_spec["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["lr"] = \
        -np.random.uniform()
    _must_fail(network_specification=network_specification, message=msg)

    msg = "SequentialNetwork() must fail with invalid l2 value"
    network_specification = copy.deepcopy(valid_network_specification_mamao)
    composite_layer_spec = network_specification[_COMPOSITE_LAYER_SPEC]
    composite_layer_spec["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["l2"] = \
        -np.random.uniform()
    _must_fail(network_specification=network_specification, message=msg)

    # ********************************************************************************
    # Constraint: must match
    # 1. Number of outputs from the inference
    # 2. Number of inputs = number of outputs of the objective layer
    # 3. num_nodes parameter of SequentialNetwork(arg)
    # ********************************************************************************
    msg = "SequentialNetwork() must fail when num_nodes does not match that of last inference layer"
    network_specification = copy.deepcopy(valid_network_specification_mamao)
    composite_layer_spec = network_specification[_COMPOSITE_LAYER_SPEC]
    composite_layer_spec["activation02"][_PARAMETERS][_NUM_NODES] = (_M-1)
    _must_fail(network_specification=network_specification, message=msg)

    msg = "SequentialNetwork() must fail when num_nodes does not match that of the objective layer"
    network_specification = copy.deepcopy(valid_network_specification_mamao)
    composite_layer_spec = network_specification[_COMPOSITE_LAYER_SPEC]
    composite_layer_spec["objective"][_PARAMETERS][_NUM_NODES] = (_M-1)
    _must_fail(network_specification=network_specification, message=msg)

    msg = "SequentialNetwork() must fail when there is duplicated layer names"
    _must_fail(
        network_specification=invalid_network_specification_with_duplicated_names,
        message=msg
    )
    D = 10
    M01 = 32
    M02 = 32
    M = 10
    _must_fail(
        network_specification=multilayer_network_specification_bn_to_fail(D, M01, M02, M),
        message=msg
    )


def test_010_validate_sequential_matmul_relu_training():
    # --------------------------------------------------------------------------------
    # Network using sequential
    # --------------------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    network = _must_succeed(
        network_specification=valid_network_specification_mao,
        message="SequentialNetwork() must succeed with %s"
                % valid_network_specification_mao
    )
    inference_layer: layer.Sequential = network.layer_inference
    sequential_matmul_layer: layer.Matmul = inference_layer.layers[0]

    # --------------------------------------------------------------------------------
    # Network without sequential
    # Use the same W from the sequential one.
    # --------------------------------------------------------------------------------
    W = sequential_matmul_layer.W
    W_non_sequential = copy.deepcopy(W)

    optimizer_non_sequential = optimiser.SGD(lr=_lr, l2=_l2)
    *network_components, = build_matmul_relu_objective(
        _M,
        _D,
        W=W_non_sequential,
        optimizer=optimizer_non_sequential,
        log_loss_function=softmax_cross_entropy_log_loss,
        log_level=logging.DEBUG
    )
    matmul, activation, loss = network_components

    # Set objective functions at each layer
    activation.objective = loss.function
    matmul.objective = compose(activation.function, loss.function)
    objective = compose(matmul.function, matmul.objective)

    # ********************************************************************************
    # Constraint:
    # ********************************************************************************
    assert np.array_equal(matmul.W, sequential_matmul_layer.W)

    # --------------------------------------------------------------------------------
    # Training data
    # --------------------------------------------------------------------------------
    X, T, V = linear_separable_sectors(n=_N, d=_D, m=_M)
    X, T = transform_X_T(X, T)
    assert X.shape == (_N, _D)

    num_epochs = 100

    loss.T = T
    for i in range(num_epochs):
        # pylint: disable=not-callable
        L = objective(X)

        # --------------------------------------------------------------------------------
        # Expected outputs and gradients from the non-sequential
        # --------------------------------------------------------------------------------
        P = softmax(relu(np.matmul(matmul.X, matmul.W.T)))
        EDA = expected_gradient_from_log_loss(P=P, T=T, N=_N)
        Y, A, EDY, EDW, EDX, dY, dX, dW = validate_relu_neuron_round_trip(
            matmul=matmul,
            activation=activation,
            X=X,
            dA=EDA
        )

        # --------------------------------------------------------------------------------
        # Run sequential
        # --------------------------------------------------------------------------------
        network.T = T
        A_sequential = network.function(X)
        # pylint: disable=not-callable
        L_sequential = network.objective(A_sequential)
        dX_sequential = network.gradient(TYPE_FLOAT(1))

        # ********************************************************************************
        # Constraint:
        #   sequential layer output and gradients must match with those from without
        #   sequential.
        # ********************************************************************************
        assert validate_against_expected_loss(expected=A, actual=A_sequential), \
            "Expected A is \n%s\nactual is \n%s\ndiff %s\n" % (A, A_sequential, A-A_sequential)
        assert validate_against_expected_loss(expected=L, actual=L_sequential), \
            "Expected L is \n%s\nactual is %s\ndiff %s\n" % (L, L_sequential, L-L_sequential)
        assert validate_against_expected_gradient(expected=dX, actual=dX_sequential), \
            "Expected dX is \n%s\nactual is %s\ndiff %s\n" % (dX, dX_sequential, dX-dX_sequential)

        # --------------------------------------------------------------------------------
        # gradient descent and get the analytical dL/dX, dL/dW
        # --------------------------------------------------------------------------------
        matmul.update()  # dL/dX, dL/dW
        dS = network.update()

        print(L_sequential)


def test_010_sequential_train2():
    # --------------------------------------------------------------------------------
    # Network without sequential layer
    # --------------------------------------------------------------------------------
    num_epochs = 100

    # --------------------------------------------------------------------------------
    # Network using sequential
    # --------------------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    network = _must_succeed(
        network_specification=valid_network_specification_sfmbambamamo,
        message="SequentialNetwork() must succeed with %s"
                % valid_network_specification_sfmbambamamo
    )
    inference_layer: layer.Sequential = network.layer_inference
    sequential_matmul_layer: layer.Matmul = inference_layer.layers[0]

    # --------------------------------------------------------------------------------
    # Training data
    # --------------------------------------------------------------------------------
    X, T, V = linear_separable_sectors(n=_N, d=_D, m=_M)
    X, T = transform_X_T(X, T)
    assert X.shape == (_N, _D)

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(num_epochs):
        network.train(X=X, T=T)

    profiler.disable()
    profiler.print_stats(sort="cumtime")

    for loss in network.history:
        print(loss)


def test():
    from layer.constants import (
        _WEIGHTS,
        _NAME,
        _SCHEME,
        _OPTIMIZER,
        _NUM_NODES,
        _NUM_FEATURES,
        _LOG_LEVEL,
        _PARAMETERS
    )
    from layer import (
        Matmul,
        CrossEntropyLogLoss
    )
    from optimizer import (
        SGD
    )
    from common.function import (
        sigmoid_cross_entropy_log_loss
    )
    from data import (
        linear_separable,
        linear_separable_sectors,
    )
    M = 1
    N = 500  # Number of plots
    D = 2  # Number of features
    sigmoid_classifier_specification = {
        _NAME: "softmax_classifier",
        _NUM_NODES: M,
        _LOG_LEVEL: logging.ERROR,
        _COMPOSITE_LAYER_SPEC: {
            "matmul01": Matmul.specification(
                name="matmul",
                num_nodes=M,
                num_features=D,
                weights_initialization_scheme="he",
                weights_optimizer_specification=SGD.specification(
                    lr=TYPE_FLOAT(0.2),
                    l2=TYPE_FLOAT(1e-3)
                )
            ),
            "loss": CrossEntropyLogLoss.specification(
                name="loss",
                num_nodes=M,
                loss_function=sigmoid_cross_entropy_log_loss.__qualname__
            )
        }
    }
    logistic_classifier = SequentialNetwork.build(
        specification=sigmoid_classifier_specification,
    )
    MAX_TEST_TIMES = 50
    X_Bin, T_Bin, V_Bin = linear_separable(d=D, n=N)
    for i in range(MAX_TEST_TIMES):
        logistic_classifier.train(X=X_Bin, T=T_Bin)
