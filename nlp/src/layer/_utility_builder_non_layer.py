"""
Responsibility:
    Factory to create non-layer instances e.g. weights, optimizer.

Consumer:
    Layer classes.

Note:
    DO NOT import this file from those classes that are used from
    the layer classes.
"""
from typing import (
    Dict,
)

import numpy as np

import common.weights as weights
import optimizer as optimiser
from layer.constants import (
    _WEIGHTS,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS
)


def build_optimizer_from_layer_parameters(layer_parameters: Dict):
    if _OPTIMIZER not in layer_parameters:
        _optimizer = optimiser.SGD()    # Use SGD() as default
    else:
        optimizer_spec = layer_parameters[_OPTIMIZER]
        assert \
            _SCHEME in optimizer_spec, \
            "Invalid optimizer specification %s" % optimizer_spec

        optimizer_scheme = optimizer_spec[_SCHEME].lower()
        assert optimizer_scheme in optimiser.SCHEMES, \
            "Invalid optimizer scheme %s. Scheme must be one of %s." \
            % (optimizer_scheme, optimiser.SCHEMES)

        _optimizer = \
            optimiser.SCHEMES[optimizer_scheme].build(optimizer_spec[_PARAMETERS]) \
            if _PARAMETERS in optimizer_spec else optimiser.SGD()
    return _optimizer


def build_weights_from_layer_parameters(parameters: Dict) -> np.ndarray:
    """Build layer.Matmul weights
    Args:
        parameters: layer parameter
    Return: initialized weight

    Weight Specification: {
        "scheme": "weight initialization scheme [he|xavier]"
    }
    """
    assert \
        _NUM_NODES in parameters and \
        _NUM_FEATURES in parameters and \
        _WEIGHTS in parameters

    num_nodes = parameters[_NUM_NODES]
    num_features = parameters[_NUM_FEATURES]
    weight_spec = parameters[_WEIGHTS]

    M = num_nodes
    D = num_features + 1    # Add bias
    scheme = weight_spec[_SCHEME] \
        if _SCHEME in weight_spec else "uniform"
    assert scheme in weights.SCHEMES
    return weights.SCHEMES[scheme](M, D)

