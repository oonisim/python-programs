from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable
)
import logging
import numpy as np
from common.constants import (
    TYPE_FLOAT
)
from common.function import (
    softmax_cross_entropy_log_loss,
    compose
)
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _LOG_LEVEL
)
from layer.matmul import (
    Matmul
)
from layer.activation import (
    ReLU,
    Sigmoid
)
from layer.objective import (
    CrossEntropyLogLoss,
)
from optimizer import (
    Optimizer,
    SGD
)


def build_matmul_relu_objective(
        M: int,
        D: int,
        W: np.ndarray = None,
        optimizer: Optimizer = SGD(),
        log_loss_function: Callable = softmax_cross_entropy_log_loss,
        log_level=logging.ERROR
):
    """
    Responsibility:
        Generate instances of Matmul, ReLU, and Log loss without wiring them.
        Wiring is the function of the Sequential layer to test

    Args:
        M: Number of Matmul nodes
        D: Number of feature in the input X into Matmul without bias.
           Bias is added by the Matmul itself internally.
        W: Weight
        optimizer: Optimizer
        log_loss_function: Cross entropy log loss function for objective layer
        log_level: log level

    Returns:
        matmul, relu, objective
    """
    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    objective = CrossEntropyLogLoss(
        name="loss",
        num_nodes=M,
        log_loss_function=log_loss_function,
        log_level=log_level
    )

    # --------------------------------------------------------------------------------
    # Instantiate the 2nd ReLU layer
    # --------------------------------------------------------------------------------
    activation = ReLU(
        name="relu",
        num_nodes=M,
        log_level=log_level
    )

    # --------------------------------------------------------------------------------
    # Instantiate the 2nd Matmul layer
    # --------------------------------------------------------------------------------
    if W is None:
        matmul_spec = {
            _NAME: "matmul",
            _NUM_NODES: M,
            _NUM_FEATURES: D,
            _WEIGHTS: {
                _SCHEME: "he",
                _NUM_NODES: M,
                _NUM_FEATURES: D + 1
            },
            _OPTIMIZER: {
                _SCHEME: "SGD"
            },
            _LOG_LEVEL: logging.ERROR
        }
        matmul = Matmul.build(matmul_spec)
    else:
        matmul = Matmul(
            name="matmul",
            num_nodes=M,
            W=W,
            optimizer=optimizer,
            log_level=log_level
        )

    return matmul, activation, objective
