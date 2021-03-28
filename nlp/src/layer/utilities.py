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
from common.constants import (
    TYPE_FLOAT
)
from common.functions import (
    softmax_cross_entropy_log_loss,
    compose
)
from layer.base import (
    Layer
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
from layer.normalization import (
    Standardization,
    BatchNormalization
)
from optimizer import (
    Optimizer,
    SGD
)
Logger = logging.getLogger("functions")


# ================================================================================
# Dictionaries of layer per purpose
# ================================================================================
FEATURE_LAYERS = (
    Matmul,
)
ACTIVATION_LAYERS = (
    Sigmoid,
    ReLU,
)
NORMALIZATION_LAYERS = (
    Standardization,
    BatchNormalization
)

# --------------------------------------------------------------------------------
# Inference layers
# --------------------------------------------------------------------------------
FUNCTION_LAYERS = \
    FEATURE_LAYERS + \
    NORMALIZATION_LAYERS + \
    ACTIVATION_LAYERS

FUNCTION_LAYER_SCHEMES = {}
for __layer in FUNCTION_LAYERS:
    FUNCTION_LAYER_SCHEMES[__layer.__qualname__.lower()] = __layer

# --------------------------------------------------------------------------------
# Objective layers
# --------------------------------------------------------------------------------
OBJECTIVE_LAYERS = (
    CrossEntropyLogLoss,
)
OBJECTIVE_LAYER_SCHEMES = {}
for __layer in OBJECTIVE_LAYERS:
    OBJECTIVE_LAYER_SCHEMES[__layer.__qualname__.lower()] = __layer

# All layers
SCHEMES = {
    **FUNCTION_LAYER_SCHEMES,
    **OBJECTIVE_LAYER_SCHEMES
}
assert SCHEMES


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


def compose_sequential_layer_interface(layers: List[Layer]):
    """
    Responsibility:
        Generate function composition F=(fn-1 o ... o f0) from layers.
        fi = layer(i).function
    Args:
        layers: List of layers
    Returns:
        F: Composed function F=(fn-1 o ... o f0)
    """
    assert len(layers) > 0

    function: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
    predict: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
    gradient: [[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None

    if len(layers) == 1:
        Logger.warning("Only 1 layer provided.")
        function = layers[0].function
        predict = layers[0].predict
        gradient = layers[0].gradient
    else:
        # Layer function F=(fn-1 o ... o f0)
        function = compose(*[__layer.function for __layer in layers])
        # Gradient function G=(g0 o g1 o ... o gn-1)
        gradient = compose(*[__layer.gradient for __layer in layers[::-1]])

        # --------------------------------------------------------------------------------
        # Prediction function P=(fn-2 o ... o f0) excluding the last layer if it is
        # an activation.
        # TODO: Understand why including the last activation layer make the prediction fail.
        # --------------------------------------------------------------------------------
        if isinstance(layers[-1], ACTIVATION_LAYERS):
            predict = compose(*[__layer.predict for __layer in layers])
        else:
            predict = compose(*[__layer.predict for __layer in layers[:-1]])

    return function, predict, gradient


def compose_sequential_layer_objective(
        layers: List[Layer],
        objective: Callable
) -> Callable:
    """Build the objective function of a sequential layer
    Args:
        layers: layers in the sequence
        objective: objective function of the last layer
    """
    assert callable(objective)

    for __layer in layers[::-1]:
        __layer.objective = objective
        objective = compose(*[__layer.function, objective])

    return objective


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
