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
from data import (
    linear_separable,
    linear_separable_sectors
)
import layer
import optimizer as optimiser
from test.layer_validations import (
    validate_against_expected_gradient
)
from test.layer_validations import (
    validate_against_numerical_gradient
)
from network.sequential import (
    SequentialNetwork
)

Logger = logging.getLogger(__name__)
Logger.setLevel(logging.DEBUG)


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


def test_010_base_instantiation_to_fail():
    network_specification = {
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
                    "lr": 0.01,
                    "l2": 1e-3
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

    network = SequentialNetwork(
        name="test_010_base_instantiation_to_fail",
        num_nodes=3,    # number of the last layer output,
        specification=network_specification,
        log_level=logging.DEBUG
    )
