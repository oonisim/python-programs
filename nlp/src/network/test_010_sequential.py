"""Network base test cases"""
import cProfile
import copy
import logging

import numpy as np

import layer
import optimizer as optimiser
from common.constants import (
    TYPE_FLOAT,
)
from common.function import (
    softmax,
    relu,
    transform_X_T,
    softmax_cross_entropy_log_loss,
    compose,
)
from data import (
    linear_separable_sectors
)
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS,
    _LOSS_FUNCTION,
    _COMPOSITE_LAYER_SPEC,
    _LOG_LEVEL
)
from network.sequential import (
    SequentialNetwork
)
from test.layer_validations import (
    expected_gradient_from_log_loss,
    validate_relu_neuron_round_trip
)
from test.utilities import (
    build_matmul_relu_objective
)
from config_test_010_sequential import (
    valid_network_specification_mao,
    valid_network_specification_mamao,
    _N,
    _M,
    _D,
    _lr,
    _l2,
    invalid_network_specification_with_duplicated_names,
    multilayer_network_specification_bn_to_fail
)

Logger = logging.getLogger(__name__)


def _must_fail(
        network_specification,
        message
):
    try:
        network = SequentialNetwork(
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
        network = SequentialNetwork(
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
        L_sequential = network.objective(A_sequential)
        dX_sequential = network.gradient(TYPE_FLOAT(1))

        # ********************************************************************************
        # Constraint:
        #   sequential layer output and gradients must match with those from without
        #   sequential.
        # ********************************************************************************
        assert np.allclose(A_sequential, A, atol=0.0, rtol=0.01), \
            "Expected A is \n%s\nactual is \n%s\ndiff %s\n" % (A, A_sequential, A-A_sequential)
        assert np.allclose(L_sequential, L, atol=0.0, rtol=0.01), \
            "Expected L is \n%s\nactual is %s\ndiff %s\n" % (L, L_sequential, L-L_sequential)
        assert np.allclose(dX_sequential, dX, atol=0.0, rtol=0.01), \
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
    from config_test_010_sequential import (
        valid_network_specification_mbambamamo,
        _N,
        _M,
        _D,
    )

    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    network = _must_succeed(
        network_specification=valid_network_specification_mbambamamo,
        message="SequentialNetwork() must succeed with %s"
                % valid_network_specification_mbambamamo
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

