from typing import (
    Dict,
)
import numpy as np
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS
)
import optimizer as optimiser
import common.weights as weights


def build_optimizer_from_layer_specification(specification: Dict):
    spec = specification

    if _OPTIMIZER in spec:
        assert \
            _SCHEME in spec[_OPTIMIZER], \
            "Invalid optimizer spec %s" % spec[_OPTIMIZER]

    if _OPTIMIZER in spec and _SCHEME in spec[_OPTIMIZER]:
        optimizer_spec = spec[_OPTIMIZER]
        optimizer_scheme = optimizer_spec[_SCHEME].lower()
        assert optimizer_scheme in optimiser.SCHEMES, \
            "Invalid optimizer scheme %s. Scheme must be one of %s." \
            % (optimizer_scheme, optimiser.SCHEMES)

        parameters = optimizer_spec[_PARAMETERS] \
            if _PARAMETERS in optimizer_spec else {}
        parameters[_NAME] = parameters[_NAME] \
            if _NAME in parameters else spec[_NAME]
        _optimizer = optimiser.SCHEMES[optimizer_scheme].build(parameters)
    else:
        _optimizer = optimiser.SGD()

    return _optimizer


def build_weights_from_layer_specification(specification: Dict) -> np.ndarray:
    """Build layer.Matmul weights
    Args:
        specification: layer specification
    Return: initialized weight

    Weight Specification: {
        "scheme": "weight initialization scheme [he|xavier]"
    }
    """
    layer_spec = specification
    num_nodes = layer_spec[_NUM_NODES]
    num_features = layer_spec[_NUM_FEATURES]
    weight_spec = layer_spec[_WEIGHTS]

    M = num_nodes
    D = num_features + 1    # Add bias
    scheme = weight_spec[_SCHEME] \
        if _SCHEME in weight_spec else "uniform"
    assert scheme in weights.SCHEMES
    return weights.SCHEMES[scheme](M, D)

