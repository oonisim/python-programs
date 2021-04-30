from layer.constants import (
    _COMPOSITE_LAYER_SPEC,
    _SCHEME,
    _NAME,
    _NUM_NODES,
    _LOG_LEVEL,
    _PARAMETERS,
    _NUM_FEATURES,
    _WEIGHTS,
    _OPTIMIZER,
    _LOSS_FUNCTION,
)

_network_layer_specification_template = {
    "name": "multilayer_network",
    _NUM_NODES: 10,
    _LOG_LEVEL: 40,
    _COMPOSITE_LAYER_SPEC: {
        "std000": {
            _SCHEME: "Standardization",
            _PARAMETERS: {
                "name": "std000",
                _NUM_NODES: 5,
                "momentum": 0.9,
                "eps": 0.0,
                _LOG_LEVEL: 40
            }
        },
        "matmul001": {
            _SCHEME: "Matmul",
            _PARAMETERS: {
                "name": "matmul001",
                _NUM_NODES: 10,
                _NUM_FEATURES: 5,
                _WEIGHTS: {
                    _SCHEME: "he"
                },
                _OPTIMIZER: {
                    _SCHEME: "SGD",
                    _PARAMETERS: {
                        "name": "sgd",
                        "lr": 0.05,
                        "l2": 0.001
                    }
                }
            }
        },
        "bn001": {
            _SCHEME: "BatchNormalization",
            _PARAMETERS: {
                "name": "bn001",
                _NUM_NODES: 10,
                "momentum": 0.9,
                _OPTIMIZER: {
                    _SCHEME: "SGD",
                    _PARAMETERS: {
                        "name": "sgd",
                        "lr": 0.05,
                        "l2": 0.001
                    }
                },
                "eps": 0.0,
                _LOG_LEVEL: 40
            }
        },
        "activation001": {
            _SCHEME: "ReLU",
            _PARAMETERS: {
                "name": "relu001",
                _NUM_NODES: 10,
                "slope": 1e-10
            }
        },
        "matmul": {
            _SCHEME: "Matmul",
            _PARAMETERS: {
                "name": "matmul",
                _NUM_NODES: 10,
                _NUM_FEATURES: 10,
                _WEIGHTS: {
                    _SCHEME: "he"
                },
                _OPTIMIZER: {
                    _SCHEME: "SGD",
                    _PARAMETERS: {
                        "name": "sgd",
                        "lr": 0.05,
                        "l2": 0.001
                    }
                }
            }
        },
        "loss": {
            _SCHEME: "CrossEntropyLogLoss",
            _PARAMETERS: {
                "name": "loss",
                _NUM_NODES: 10,
                _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
            }
        }
    }
}
