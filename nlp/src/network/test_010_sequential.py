"""Network base test cases"""
import cProfile
import copy
import logging
from typing import (
    List,
    Callable
)
import numpy as np
import common.weights as weights
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
Logger.setLevel(logging.DEBUG)


def test_010_sequential_instantiation_to_fail():
    lr = np.random.uniform()
    l2 = np.random.uniform()

    valid_network_specification = {
        "matmul01": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 8,
                _NUM_FEATURES: 2,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                },
                _OPTIMIZER: {
                    _SCHEME: optimiser.SGD.__qualname__,
                    _PARAMETERS: {
                        "lr": lr,
                        "l2": l2
                    }
                }
            },
        },
        "activation01": {
            _SCHEME: layer.ReLU.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 8
            }
        },
        "matmul02": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 3,
                _NUM_FEATURES: 8,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                }
            }
        },
        "activation02": {
            _SCHEME: layer.ReLU.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 3
            }
        },
        "objective": {
            _SCHEME: layer.CrossEntropyLogLoss.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 3,
                _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
            }
        }
    }

    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    try:
        network = SequentialNetwork(
            name="test_010_base_instantiation_to_fail",
            num_nodes=3,    # number of the last layer output,
            specification=valid_network_specification,
            log_level=logging.DEBUG
        )

        inference_layer: layer.Sequential = network.layer_inference
        matmul_layer: layer.Matmul = inference_layer.layers[0]
        assert matmul_layer.optimizer.lr == lr
        assert matmul_layer.optimizer.l2 == l2

    except Exception as e:
        raise RuntimeError(
            "SequentialNetwork() must succeed with %s"
            % valid_network_specification
        )

    network_specification = copy.deepcopy(valid_network_specification)
    network_specification["matmul01"][_PARAMETERS][_OPTIMIZER][_PARAMETERS]["lr"] = \
        -np.random.uniform()
    try:
        network = SequentialNetwork(
            name="test_010_base_instantiation_to_fail",
            num_nodes=3,    # number of the last layer output,
            specification=network_specification,
            log_level=logging.DEBUG
        )
        raise RuntimeError("SequentialNetwork() must fail with invalid lr value")
    except AssertionError:
        pass


def test_010_sequential_train():
    # --------------------------------------------------------------------------------
    # Network without sequential layer
    # --------------------------------------------------------------------------------
    N = 10
    D = 2
    M: int = 3
    lr = 0.01
    l2 = 1e-3

    # --------------------------------------------------------------------------------
    # Network using sequential
    # --------------------------------------------------------------------------------
    valid_network_specification = {
        "matmul01": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: M,
                _NUM_FEATURES: D,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                },
                _OPTIMIZER: {
                    _SCHEME: optimiser.SGD.__qualname__,
                    _PARAMETERS: {
                        "lr": lr,
                        "l2": l2
                    }
                }
            },
        },
        "activation01": {
            _SCHEME: layer.ReLU.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: M
            }
        },
        "objective": {
            _SCHEME: layer.CrossEntropyLogLoss.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: M,
                _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
            }
        }
    }
    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    try:
        network = SequentialNetwork(
            name="test_010_base_instantiation_to_fail",
            num_nodes=M,    # number of the last layer output,
            specification=valid_network_specification,
            log_level=logging.DEBUG
        )

        inference_layer: layer.Sequential = network.layer_inference
        sequential_matmul_layer: layer.Matmul = inference_layer.layers[0]
        assert sequential_matmul_layer.optimizer.lr == lr
        assert sequential_matmul_layer.optimizer.l2 == l2

    except Exception as e:
        raise RuntimeError(
            "SequentialNetwork() must succeed with %s"
            % valid_network_specification
        )

    # --------------------------------------------------------------------------------
    # Network without sequential
    # Use the same W from the sequential one.
    # --------------------------------------------------------------------------------
    W = sequential_matmul_layer.W
    W_non_sequential = copy.deepcopy(W)

    optimizer_non_sequential = optimiser.SGD(lr=lr, l2=l2)
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
    N = 16
    D = 2
    M: int = 3
    lr = 0.1
    l2 = 1e-3
    num_epochs = 100

    # --------------------------------------------------------------------------------
    # Network using sequential
    # --------------------------------------------------------------------------------
    valid_network_specification = {
        "matmul01": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 16,
                _NUM_FEATURES: 2,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                },
                _OPTIMIZER: {
                    _SCHEME: optimiser.SGD.__qualname__,
                    _PARAMETERS: {
                        "lr": lr,
                        "l2": l2
                    }
                }
            },
        },
        "bn01": {
            _SCHEME: layer.BatchNormalization.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 16,
                "momentum": 0.9,
            }
        },
        "activation01": {
            _SCHEME: layer.ReLU.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 16
            }
        },
        "matmul02": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 32,
                _NUM_FEATURES: 16,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                },
                _OPTIMIZER: {
                    _SCHEME: optimiser.SGD.__qualname__,
                    _PARAMETERS: {
                        "lr": lr,
                        "l2": l2
                    }
                }
            },
        },
        "bn02": {
            _SCHEME: layer.BatchNormalization.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 32,
            }
        },
        "activation02": {
            _SCHEME: layer.ReLU.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 32
            }
        },
        "matmul03": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 16,
                _NUM_FEATURES: 32,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                },
                _OPTIMIZER: {
                    _SCHEME: optimiser.SGD.__qualname__,
                    _PARAMETERS: {
                        "lr": lr,
                        "l2": l2
                    }
                }
            },
        },
        "activation03": {
            _SCHEME: layer.ReLU.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 16
            }
        },
        "matmul04": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: 3,
                _NUM_FEATURES: 16,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                }
            }
        },
        "objective": {
            _SCHEME: layer.CrossEntropyLogLoss.__qualname__,
            _PARAMETERS: {
                _NUM_NODES: M,
                _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
            }
        }
    }

    # ----------------------------------------------------------------------
    # Validate the correct specification.
    # NOTE: Invalidate one parameter at a time from the correct one.
    # Otherwise not sure what you are testing.
    # ----------------------------------------------------------------------
    try:
        network = SequentialNetwork(
            name="test_010_base_instantiation_to_fail",
            num_nodes=M,    # number of the last layer output,
            specification=valid_network_specification,
            log_level=logging.DEBUG
        )

        inference_layer: layer.Sequential = network.layer_inference
        sequential_matmul_layer: layer.Matmul = inference_layer.layers[0]
        assert sequential_matmul_layer.optimizer.lr == lr
        assert sequential_matmul_layer.optimizer.l2 == l2

    except Exception as e:
        raise RuntimeError(
            "SequentialNetwork() must succeed with %s"
            % valid_network_specification
        )

    # --------------------------------------------------------------------------------
    # Training data
    # --------------------------------------------------------------------------------
    X, T, V = linear_separable_sectors(n=N, d=D, m=M)
    X, T = transform_X_T(X, T)
    assert X.shape == (N, D)

    for i in range(num_epochs):
        # --------------------------------------------------------------------------------
        # Run sequential
        # --------------------------------------------------------------------------------
        network.T = T
        A_sequential = network.function(X)
        L_sequential = network.objective(A_sequential)
        dX_sequential = network.gradient(TYPE_FLOAT(1))

        dS = network.update()
        print(L_sequential)
