import logging

import numpy as np

import layer
import optimizer as optimiser
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT
)
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS,
    _LOSS_FUNCTION,
    _LOG_LEVEL,
    _COMPOSITE_LAYER_SPEC
)

_lr = np.random.uniform()
_l2 = np.random.uniform()
_M = 3  # Number of output/classes
_D = 2
_eps = np.random.uniform(low=0, high=1e-4)
_momentum = TYPE_FLOAT(np.random.uniform(low=0.9, high=1.0))

_composite_layer_specification_template = {
    "matmul01": {
        _SCHEME: layer.Matmul.class_id(),
        _PARAMETERS: {
            _NAME: "matmul01",
            _NUM_NODES: 16,
            _NUM_FEATURES: _D,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: "he"
            },
            _OPTIMIZER: {
                _SCHEME: optimiser.SGD.class_id(),
                _PARAMETERS: {
                    "lr": _lr,
                    "l2": _l2
                }
            }
        },
    },
    "bn01": {
        _SCHEME: layer.BatchNormalization.class_id(),
        _PARAMETERS: {
            _NAME: "bn01",
            _NUM_NODES: 16,
            _OPTIMIZER: {
                _SCHEME: optimiser.SGD.class_id(),
                _PARAMETERS: {
                    "lr": _lr,
                    "l2": _l2
                }
            },
            "momentum": _momentum,
            "eps": _eps,
            "log_level": logging.ERROR
        }
    },
    "activation01": {
        _SCHEME: layer.ReLU.class_id(),
        _PARAMETERS: {
            _NAME: "relu01",
            _NUM_NODES: 16
        }
    },
    "objective": {
        _SCHEME: layer.CrossEntropyLogLoss.class_id(),
        _PARAMETERS: {
            _NAME: "loss",
            _NUM_NODES: _M,
            _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
        }
    }
}

_sequential_composite_layer_specification = {
    _NAME: "valid_network_mao",
    _NUM_NODES: _M,
    _LOG_LEVEL: logging.ERROR,
    _COMPOSITE_LAYER_SPEC: _composite_layer_specification_template
}
