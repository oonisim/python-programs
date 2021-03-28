import logging
from typing import (
    Union,
    List,
    Callable
)

import numpy as np

from common.constants import (
    TYPE_FLOAT
)
from common.functions import (
    compose
)
from layer.base import (
    Layer
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


def compose_sequential_layer_interface(
        layers: List[Layer],
        omit_last_activation_for_prediction: bool = False
):
    """
    Responsibility:
        Generate function composition F=(fn-1 o ... o f0) from layers.
        fi = layer(i).function
    Args:
        layers: List of layers
        omit_last_activation_for_prediction:
            flag to omit the last activation at compositing prediction function.
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
        if omit_last_activation_for_prediction and isinstance(layers[-1], ACTIVATION_LAYERS):
            predict = compose(*[__layer.predict for __layer in layers[:-1]])
        else:
            predict = compose(*[__layer.predict for __layer in layers])

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


