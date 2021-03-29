import numpy as np
import layer
import optimizer as optimiser
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
_lr = np.random.uniform()
_l2 = np.random.uniform()
M = 3  # Number of output/classes
N = 10
D = 2

valid_network_specification_mamao = {
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
            _NUM_NODES: M
        }
    },
    "objective": {
        _SCHEME: layer.CrossEntropyLogLoss.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: M,
            _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
        }
    }
}

valid_network_specification_mao = {
    "matmul01": {
        _SCHEME: layer.Matmul.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: M,
            _NUM_FEATURES: D,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: "he"
            },
            _OPTIMIZER: {
                _SCHEME: optimiser.SGD.__qualname__,
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
            _NUM_NODES: M
        }
    },
    "objective": {
        _SCHEME: layer.CrossEntropyLogLoss.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: M,
            _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
        }
    }
}

valid_network_specification_mbambamamo = {
    "matmul01": {
        _SCHEME: layer.Matmul.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 16,
            _NUM_FEATURES: 2,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: "he"
            },
            _OPTIMIZER: {
                _SCHEME: optimiser.SGD.__qualname__,
                _PARAMETERS: {
                    "lr": _lr,
                    "l2": _l2
                }
            }
        },
    },
    "bn01": {
        _SCHEME: layer.BatchNormalization.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 16,
            "momentum": 0.9,
        }
    },
    "activation01": {
        _SCHEME: layer.ReLU.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 16
        }
    },
    "matmul02": {
        _SCHEME: layer.Matmul.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 32,
            _NUM_FEATURES: 16,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: "he"
            },
            _OPTIMIZER: {
                _SCHEME: optimiser.SGD.__qualname__,
                _PARAMETERS: {
                    "lr": _lr,
                    "l2": _l2
                }
            }
        },
    },
    "bn02": {
        _SCHEME: layer.BatchNormalization.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 32,
        }
    },
    "activation02": {
        _SCHEME: layer.ReLU.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 32
        }
    },
    "matmul03": {
        _SCHEME: layer.Matmul.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 16,
            _NUM_FEATURES: 32,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: "he"
            },
            _OPTIMIZER: {
                _SCHEME: optimiser.SGD.__qualname__,
                _PARAMETERS: {
                    "lr": _lr,
                    "l2": _l2
                }
            }
        },
    },
    "activation03": {
        _SCHEME: layer.ReLU.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 16
        }
    },
    "matmul04": {
        _SCHEME: layer.Matmul.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: 3,
            _NUM_FEATURES: 16,  # NOT including bias
            _WEIGHTS: {
                _SCHEME: "he"
            }
        }
    },
    "objective": {
        _SCHEME: layer.CrossEntropyLogLoss.__qualname__,
        _PARAMETERS: {
            _NUM_NODES: M,
            _LOSS_FUNCTION: "softmax_cross_entropy_log_loss"
        }
    }
}