"""Network base test cases"""
import cProfile
import copy
import logging
from typing import (
    List,
    Callable
)
import numpy as np
from common.constants import (
    TYPE_FLOAT,
)
from common.functions import (
    softmax,
    relu,
    transform_X_T,
    softmax_cross_entropy_log_loss,
    compose,
)
import layer
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS,
    _LOSS_FUNCTION,
)
from layer.utilities import (
    forward_outputs,
    backward_outputs,
)
import optimizer as optimiser
from network.sequential import (
    SequentialNetwork
)
from test.utilities import (
    build_matmul_relu_objective
)
from test.layer_validations import (
    validate_against_expected_gradient,
    validate_against_numerical_gradient,
    expected_gradient_from_log_loss,
    expected_gradients_from_relu_neuron,
    validate_relu_neuron_round_trip
)
from data import (
    linear_separable,
    linear_separable_sectors
)

Logger = logging.getLogger(__name__)
# Logger.setLevel(logging.DEBUG)


def _must_fail(
        name,
        M,
        network_specification,
        message
):
    try:
        network = SequentialNetwork(
            name=name,
            num_nodes=M,    # number of the last layer output,
            specification=network_specification,
            log_level=logging.DEBUG
        )
        raise RuntimeError(message)
    except AssertionError:
        pass


def _must_succeed(
        name,
        num_nodes,
        specification,
        log_level,
        message
):
    try:
        network = SequentialNetwork(
            name=name,
            num_nodes=num_nodes,    # number of the last layer output,
            specification=specification,
            log_level=log_level
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
    from config_test_010_sequential_config import (
        valid_network_specification_mamao,
        M
    )
    lr = np.random.uniform()
    l2 = np.random.uniform()
    valid_network_specification_mamao["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["lr"] = lr
    valid_network_specification_mamao["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["l2"] = l2
    network = _must_succeed(
        name="test_010_base_instantiation_to_fail",
        num_nodes=M,  # number of the last layer output,
        specification=valid_network_specification_mamao,
        log_level=logging.DEBUG,
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
    network_specification["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["lr"] = \
        -np.random.uniform()
    _must_fail(name=name, M=M, network_specification=network_specification, message=msg)

    msg = "SequentialNetwork() must fail with invalid l2 value"
    network_specification = copy.deepcopy(valid_network_specification_mamao)
    network_specification["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["l2"] = \
        -np.random.uniform()
    _must_fail(name=name, M=M, network_specification=network_specification, message=msg)

    # ********************************************************************************
    # Constraint: must match
    # 1. Number of outputs from the inference
    # 2. Number of inputs = number of outputs of the objective layer
    # 3. num_nodes parameter of SequentialNetwork(arg)
    # ********************************************************************************
    msg = "SequentialNetwork() must fail when num_nodes does not match that of last inference layer"
    network_specification = copy.deepcopy(valid_network_specification_mamao)
    network_specification["activation02"][_PARAMETERS][_NUM_NODES] = (M-1)
    _must_fail(name=name, M=M, network_specification=network_specification, message=msg)

    msg = "SequentialNetwork() must fail when num_nodes does not match that of the objective layer"
    network_specification = copy.deepcopy(valid_network_specification_mamao)
    network_specification["objective"][_PARAMETERS][_NUM_NODES] = (M-1)
    _must_fail(name=name, M=M, network_specification=network_specification, message=msg)


def test_010_validate_sequential_matmul_relu_training():
    # --------------------------------------------------------------------------------
    # Network using sequential
    # --------------------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    from config_test_010_sequential_config import (
        valid_network_specification_mao,
        M,
        N,
        D,
        _lr,
        _l2
    )
    network = _must_succeed(
        name="test_010_base_instantiation_to_fail",
        num_nodes=M,  # number of the last layer output,
        specification=valid_network_specification_mao,
        log_level=logging.DEBUG,
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
        M,
        D,
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
    X, T, V = linear_separable_sectors(n=N, d=D, m=M)
    X, T = transform_X_T(X, T)
    assert X.shape == (N, D)

    num_epochs = 1000

    loss.T = T
    for i in range(num_epochs):
        L = objective(X)

        # --------------------------------------------------------------------------------
        # Expected outputs and gradients from the non-sequential
        # --------------------------------------------------------------------------------
        P = softmax(relu(np.matmul(matmul.X, matmul.W.T)))
        EDA = expected_gradient_from_log_loss(P=P, T=T, N=N)
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
    from config_test_010_sequential_config import (
        valid_network_specification_mbambamamo,
        N,
        M,
        D,
    )

    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    network = _must_succeed(
        name="test_010_base_instantiation_to_fail",
        num_nodes=M,  # number of the last layer output,
        specification=valid_network_specification_mbambamamo,
        log_level=logging.ERROR,
        message="SequentialNetwork() must succeed with %s"
                % valid_network_specification_mbambamamo
    )
    inference_layer: layer.Sequential = network.layer_inference
    sequential_matmul_layer: layer.Matmul = inference_layer.layers[0]

    # --------------------------------------------------------------------------------
    # Training data
    # --------------------------------------------------------------------------------
    X, T, V = linear_separable_sectors(n=N, d=D, m=M)
    X, T = transform_X_T(X, T)
    assert X.shape == (N, D)

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(num_epochs):
        network.train(X=X, T=T)

    profiler.disable()
    profiler.print_stats(sort="cumtime")

    for loss in network.history:
        print(loss)

