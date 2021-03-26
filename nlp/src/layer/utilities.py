from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable
)
import logging
import copy
import numpy as np
from common.functions import (
    softmax_cross_entropy_log_loss
)
from layer import (
    Layer,
    Matmul,
    ReLU,
    CrossEntropyLogLoss
)
from optimizer import (
    Optimizer,
    SGD
)
Logger = logging.getLogger("functions")


def forward_outputs(layers: List[Layer], X):
    outputs = []
    for __layers in layers:
        Y = __layers.function(X)
        outputs.append(Y)
        X = Y
    return outputs


def backward_outputs(layers: List[Layer], dY):
    outputs = []
    for __layers in layers[::-1]:
        dX = __layers.gradient(dY)
        outputs.append(dX)
        dY = dX
    return outputs


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
            "name": "matmul",
            "num_nodes": M,
            "num_features": D,
            "weight": {
                "scheme": "he",
                "num_nodes": M,
                "num_features": D + 1
            },
            "optimizer": {
                "scheme": "SGD"
            },
            "log_level": logging.ERROR
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