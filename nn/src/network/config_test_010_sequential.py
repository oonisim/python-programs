import logging
import numpy as np
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT
)
import layer
from layer import (
    Matmul,
    ReLU,
    CrossEntropyLogLoss
)
import optimizer as optimiser
from optimizer import (
    SGD
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
    _COMPOSITE_LAYER_SPEC,
    _LOG_LEVEL
)

_lr = TYPE_FLOAT(np.random.uniform())
_l2 = TYPE_FLOAT(np.random.uniform())
_N = 10
_M = 3  # Number of output/classes
_D = 2
_eps = TYPE_FLOAT(np.random.uniform(low=TYPE_FLOAT(0), high=TYPE_FLOAT(1e-4)))
_momentum = TYPE_FLOAT(np.random.uniform(low=TYPE_FLOAT(0.9), high=TYPE_FLOAT(1.0)))


valid_network_specification_mamao = {
    _NAME: "network_mamao",
    _NUM_NODES: _M,
    _LOG_LEVEL: logging.ERROR,
    _COMPOSITE_LAYER_SPEC: {
        "matmul01": {
            _SCHEME: layer.Matmul.class_id(),
            _PARAMETERS: {
                _NAME: "matmul01",
                _NUM_NODES: 8,
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
        "activation01": {
            _SCHEME: layer.ReLU.class_id(),
            _PARAMETERS: {
                _NAME: "relu01",
                _NUM_NODES: 8
            }
        },
        "matmul02": {
            _SCHEME: layer.Matmul.class_id(),
            _PARAMETERS: {
                _NAME: "matmul02",
                _NUM_NODES: _M,
                _NUM_FEATURES: 8,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: "he"
                }
            }
        },
        "activation02": {
            _SCHEME: layer.ReLU.class_id(),
            _PARAMETERS: {
                _NAME: "relu02",
                _NUM_NODES: _M
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
}

valid_network_specification_mao = {
    _NAME: "valid_network_mao",
    _NUM_NODES: _M,
    _LOG_LEVEL: logging.ERROR,
    _COMPOSITE_LAYER_SPEC: {
        "matmul01": {
            _SCHEME: layer.Matmul.class_id(),
            _PARAMETERS: {
                _NAME: "matmul01",
                _NUM_NODES: _M,
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
        "activation01": {
            _SCHEME: layer.ReLU.class_id(),
            _PARAMETERS: {
                _NAME: "relu01",
                _NUM_NODES: _M
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
}


composite_layer_specification_sfmbambamamo = {
    "std01": {
        _SCHEME: layer.Standardization.class_id(),
        _PARAMETERS: {
            _NAME: "std01",
            _NUM_NODES: _D,
            "momentum": _momentum,
            "eps": _eps,
            "log_level": logging.ERROR
        }
    },
    "fss01": {
        _SCHEME: layer.FeatureScaleShift.class_id(),
        _PARAMETERS: {
            _NAME: "fss01",
            _NUM_NODES: _D,
            "log_level": logging.ERROR,
            _OPTIMIZER: {
                _SCHEME: optimiser.SGD.class_id(),
                _PARAMETERS: {
                    "lr": _lr,
                    "l2": _l2
                }
            }
        }
    },
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
    "matmul02": {
        _SCHEME: layer.Matmul.class_id(),
        _PARAMETERS: {
            _NAME: "matmul02",
            _NUM_NODES: 32,
            _NUM_FEATURES: 16,  # NOT including bias
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
    "bn02": {
        _SCHEME: layer.BatchNormalization.class_id(),
        _PARAMETERS: {
            _NAME: "bn02",
            _NUM_NODES: 32,
        }
    },
    "activation02": {
        _SCHEME: layer.ReLU.class_id(),
        _PARAMETERS: {
            _NAME: "relu02",
            _NUM_NODES: 32
        }
    },
    "matmul03": {
        _SCHEME: layer.Matmul.class_id(),
        _PARAMETERS: {
            _NAME: "matmul03",
            _NUM_NODES: 16,
            _NUM_FEATURES: 32,  # NOT including bias
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
    "activation03": {
        _SCHEME: layer.ReLU.class_id(),
        _PARAMETERS: {
            _NAME: "relu03",
            _NUM_NODES: 16
        }
    },
    "matmul04": {
        _SCHEME: layer.Matmul.class_id(),
        _PARAMETERS: {
            _NAME: "matmul04",
            _NUM_NODES: _M,
            _NUM_FEATURES: 16,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: "he"
            }
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

valid_network_specification_sfmbambamamo = {
    _NAME: "valid_network_mao",
    _NUM_NODES: _M,
    _LOG_LEVEL: logging.ERROR,
    _COMPOSITE_LAYER_SPEC: composite_layer_specification_sfmbambamamo
}


def invalid_network_specification_with_duplicated_names():
    N = 10
    M = 4
    D = 2
    M01 = 8
    M02: int = M  # Number of categories to classify

    MAX_TEST_TIMES = 3

    X = np.random.rand(N, 2)
    T = np.random.randint(0, 4, N)

    sequential_layer_specification = {
        "matmul01": layer.Matmul.specification(
            name="matmul01",
            num_nodes=M01,
            num_features=D,
            weights_initialization_scheme="he",
            weights_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            )
        ),
        "bn01": layer.BatchNormalization.specification(
            name="bn01",
            num_nodes=M01,
            gamma_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            ),
            beta_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3),
            ),
            momentum=TYPE_FLOAT(0.9)
        ),
        "relu01": layer.ReLU.specification(
            name="relu01",
            num_nodes=M01,
        ),
        "matmul02": layer.Matmul.specification(
            name="matmul02",
            num_nodes=M02,
            num_features=M01,
            weights_initialization_scheme="he",
            weights_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            )
        ),
        "bn02": layer.BatchNormalization.specification(
            name="bn01",
            num_nodes=M02,
            gamma_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            ),
            beta_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3),
            ),
            momentum=TYPE_FLOAT(0.9)
        ),
        "loss": layer.CrossEntropyLogLoss.specification(
            name="loss001", num_nodes=M
        )
    }

    network_specification = {
        _NAME: "two_layer_classifier_with_batch_normalization",
        _NUM_NODES: M,
        _LOG_LEVEL: logging.ERROR,
        _COMPOSITE_LAYER_SPEC: sequential_layer_specification
    }


def multilayer_network_specification_bn_to_fail(D, M01, M02, M):
    sequential_layer_specification_bn_to_fail = {
        "matmul01": layer.Matmul.specification(
            name="matmul01",
            num_nodes=M01,
            num_features=D,
            weights_initialization_scheme="he",
            weights_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            )
        ),
        "bn01": layer.BatchNormalization.specification(
            name="bn01",
            num_nodes=M01,
            gamma_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            ),
            beta_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3),
            ),
            momentum=TYPE_FLOAT(0.9)
        ),
        "relu01": layer.ReLU.specification(
            name="relu01",
            num_nodes=M01,
        ),
        "matmul02": layer.Matmul.specification(
            name="matmul01",
            num_nodes=M02,
            num_features=M01,
            weights_initialization_scheme="he",
            weights_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            )
        ),
        "bn02": layer.BatchNormalization.specification(
            name="bn02",
            num_nodes=M02,
            gamma_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            ),
            beta_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3),
            ),
            momentum=TYPE_FLOAT(0.9)
        ),
        "relu02": layer.ReLU.specification(
            name="relu02",
            num_nodes=M02,
        ),
        "matmul03": layer.Matmul.specification(
            name="matmul03",
            num_nodes=M,
            num_features=M02,
            weights_initialization_scheme="he",
            weights_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            )
        ),
        "bn03": layer.BatchNormalization.specification(
            name="bn03",
            num_nodes=M,
            gamma_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3)
            ),
            beta_optimizer_specification=optimiser.SGD.specification(
                lr=TYPE_FLOAT(0.05),
                l2=TYPE_FLOAT(1e-3),
            ),
            momentum=TYPE_FLOAT(0.9)
        ),
        "loss": CrossEntropyLogLoss.specification(
            name="loss001", num_nodes=M
        )
    }

    return {
        _NAME: "two_layer_classifier_with_batch_normalization",
        _NUM_NODES: M,
        _LOG_LEVEL: logging.ERROR,
        _COMPOSITE_LAYER_SPEC: sequential_layer_specification_bn_to_fail
    }
