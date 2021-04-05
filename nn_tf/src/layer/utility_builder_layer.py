"""Tools to use from composite layer classes
DO NOT import composite layer classes from this Python module.

Layer specification:
    A layer specification is a dictionary object which identifies:
    1. a layer class (use class.__qualname__)
    2. **kwargs parameters for __init__

    Format:
    {
        _SCHEME: <>,
        _PARAMETERS: {**kwargs __init__}
    }

    Example: (Matmul)
    {
        _SCHEME: layer.Matmul.__qualname__,
        _PARAMETERS: {
            _NAME: "matmul01",
            _NUM_NODES: 8,
            _NUM_FEATURES: 2,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: "he"
            },
            _OPTIMIZER: {
                _SCHEME: optimiser.optimizer.SGD.__qualname__,
                _PARAMETERS: {
                    "lr": 1e-2,
                    "l2": 1e-3
                }
            }
        },
    }

    Example: Sequential composite layer.

    {
        _SCHEME: layer.Sequential.__qualname__,
        _PARAMETERS: {
            _NAME: "sequential01",
            _NUM_NODES: 8,
            _LAYERS: <composite layer specification>
        }
    }


Composite layer specification:
    A composite layer specification is a dictionary of layer specifications
    to define multiple layers to instantiate a composite layer.
    A member layer is specified as "<layer id>: <layer_specification>".

    It can include another composite specification to create a nested structure.

    Example for matmul-activation-matmul-objective composite layer.
    {
        "matmul01": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NAME: "matmul01",
                _NUM_NODES: 8,
                _NUM_FEATURES: 2,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                },
                _OPTIMIZER: {
                    _SCHEME: optimiser.optimizer.SGD.__qualname__,
                    _PARAMETERS: {
                        "lr": _lr,
                        "l2": _l2
                    }
                }
            },
        },
        "activation01": {
            _SCHEME: layer.ReLU.__qualname__,
            _PARAMETERS: {
                _NAME: "relu01",
                _NUM_NODES: 8
            }
        },
        "matmul02": {
            _SCHEME: layer.Matmul.__qualname__,
            _PARAMETERS: {
                _NAME: "matmul02",
                _NUM_NODES: 3,
                _NUM_FEATURES: 8,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                }
            }
        },
        "objective": {
            _SCHEME: layer.CrossEntropyLogLoss.__qualname__,
            _PARAMETERS: {
                _NAME: "relu01",
                _NUM_NODES: M,
                _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
            }
        }
    }

"""
import copy
import logging
from typing import (
    List,
    Callable,
    Union
)

import numpy as np

from common.constants import (
    TYPE_FLOAT,
)
from common.function import (
    compose
)
from layer.base import (
    Layer,
)
from layer.constants import (
    _NAME,
    _SCHEME,
    _NUM_NODES,
    _PARAMETERS,
    _LOG_LEVEL
)
from layer.schemes import (  # Non composite layer classes only
    SCHEMES
)

Logger = logging.getLogger(__name__)


def compose_sequential_layer_interface(
        layers: List[Layer],
):
    """
    Responsibility:
        Generate function composition F=(fn-1 o ... o f0) from layers.
        fi = layer(i).function
    Args:
        layers: List of layers
            flag to omit the last activation at compositing prediction function.
    Returns:
        F: Composed function F=(fn-1 o ... o f0)
    """
    assert len(layers) > 0

    function: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
    predict: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None
    gradient: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = None

    if len(layers) == 1:
        Logger.info(
            "compose_sequential_layer_interface: Only 1 layer in sequence. "
            "Layer names=%s", [_layer.name for _layer in layers])

        function = layers[0].function
        predict = layers[0].predict
        gradient = layers[0].gradient
    else:
        Logger.debug("Layer names=%s", [_layer.name for _layer in layers])
        # Layer function F=(fn-1 o ... o f0)
        function = compose(*[__layer.function for __layer in layers])
        # Gradient function G=(g0 o g1 o ... o gn-1)
        gradient = compose(*[__layer.gradient for __layer in layers[::-1]])

        # --------------------------------------------------------------------------------
        # Prediction function P=(fn-2 o ... o f0) excluding the last layer if it is
        # an activation.
        # TODO: Understand why including the last activation layer make the prediction fail.
        # --------------------------------------------------------------------------------
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


def map_parallel_layer_interface(
        layers: List[Layer],
):
    """
    Responsibility:
        Generate function composition F=(fn-1 o ... o f0) from layers.
        fi = layer(i).function
    Args:
        layers: List of layers
            flag to omit the last activation at compositing prediction function.
    Returns:
        F: Composed function F=(fn-1 o ... o f0)
    """
    assert len(layers) > 0

    function: List[
        Callable[
            [Union[np.ndarray, TYPE_FLOAT]],
            Union[np.ndarray, TYPE_FLOAT]
        ]
    ] = None
    predict: List[
        Callable[
            [Union[np.ndarray, TYPE_FLOAT]],
            Union[np.ndarray, TYPE_FLOAT]
        ]
    ] = None
    gradient: List[
        Callable[
            [Union[np.ndarray, TYPE_FLOAT]],
            Union[np.ndarray, TYPE_FLOAT]
        ]
    ] = None

    if len(layers) == 1:
        Logger.info(
            "map_parallel_layer_interface: Only 1 layer in sequence. "
            "Layer names=%s", [_layer.name for _layer in layers])

        function = [layers[0].function]
        predict = [layers[0].predict]
        gradient = [layers[0].gradient]
    else:
        def apply_function(_layer: Layer):
            return _layer.function

        def apply_gradient(_layer: Layer):
            return _layer.gradient

        def apply_sum(x):
            return np.sum(x, axis=0)

        def apply_predict(_layer: Layer):
            return _layer.predict

        function = list(map(apply_function, layers))
        # Back propagation of a parallel layer is sum(gradients, axis=0)
        gradient = list(map(apply_gradient, layers))
        gradient = compose(gradient, apply_sum)
        predict = list(map(apply_predict, layers))

    return function, predict, gradient


def map_parallel_layer_objective(
        layers: List[Layer],
        objective: Callable
) -> List[Callable]:
    """Build the objective function of a sequential layer
    Args:
        layers: layers in the sequence
        objective: objective function of the last layer
    """
    assert callable(objective)

    def apply(_layer: Layer):
        _layer.objective = objective
        return _layer.objective

    return list(map(apply, layers))


def inspect_layer_specification(
        layer_specification: dict,
        log_level: int
):
    """
    Responsibility:
        Validate a layer specification and return a cloned specification
        to avoid unexpected mutation.

    Args:
        layer_specification: spec
        log_level: default log level to set

    Returns:
        inspected_specification: cloned inspected specification
    """
    spec = layer_specification
    # --------------------------------------------------------------------------------
    # Layer class to instantiate
    # --------------------------------------------------------------------------------
    assert \
        _SCHEME in spec and isinstance(spec[_SCHEME], str),\
        "%s mandatory but missing in %s" % (_SCHEME, spec)

    assert \
        spec[_SCHEME] in SCHEMES, \
        "%s must be one of %s but %s" % (_SCHEME, SCHEMES, spec[_SCHEME])

    # --------------------------------------------------------------------------------
    # Parameters to pass to the __init__() method of the class
    # --------------------------------------------------------------------------------
    assert \
        _PARAMETERS in spec and isinstance(spec[_PARAMETERS], dict), \
        "%s is mandatory in a layer specification.\nspec is %s" \
        % (_PARAMETERS, spec)
    parameters = spec[_PARAMETERS]

    assert \
        _NAME in parameters and isinstance(parameters[_NAME], str), \
        "%s is mandatory in %s" % (_NAME, _PARAMETERS)
    parameter_name = parameters[_NAME].lower()

    assert \
        _NUM_NODES in parameters, "%s is mandatory in %s" % (_NUM_NODES, _PARAMETERS)
    log_level = log_level \
        if _LOG_LEVEL not in parameters else parameters[_LOG_LEVEL]

    # --------------------------------------------------------------------------------
    # Clone the only required elements
    # --------------------------------------------------------------------------------
    inspected = {
        _SCHEME: copy.deepcopy(spec[_SCHEME]),
        _PARAMETERS: copy.deepcopy(spec[_PARAMETERS]),
    }
    inspected[_PARAMETERS][_LOG_LEVEL] = log_level
    inspected[_PARAMETERS][_NAME] = parameter_name
    return inspected


def build_layer_from_layer_specification(
        specification: dict,
        log_level: int,
        inspect_layer_spec: bool = False
) -> Layer:
    spec = inspect_layer_specification(
        layer_specification=specification, log_level=log_level
    ) if inspect_layer_spec else specification

    scheme: Layer = SCHEMES[spec[_SCHEME]]
    return scheme.build(spec[_PARAMETERS])


def build_layers_from_composite_layer_specification(
        specification: dict,
        log_level: int,
        inspect_layer_spec=True
) -> List[Layer]:
    assert isinstance(specification, dict) and len(specification) > 0, \
        "composite specification must have elements. \n%s\n" % specification
    layers = []
    for layer_id, layer_spec in specification.items():
        try:
            layers.append(
                build_layer_from_layer_specification(
                    specification=layer_spec,
                    log_level=log_level,
                    inspect_layer_spec=inspect_layer_spec
                )
            )
        except AssertionError as e:
            raise \
                AssertionError(
                    "%s has invalid layer specification %s"
                    % (layer_id, layer_spec)
                ) from e

    assert \
        len(layers) > 0 and \
        all([isinstance(_layer, Layer) for _layer in layers])
    return layers
