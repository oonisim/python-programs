from layer.base import Layer
from layer.normalization import (
    BatchNormalization,
    Standardization
)
from layer.matmul import Matmul
from layer.activation import (
    Sigmoid,
    ReLU
)
from layer.objective import (
    CrossEntropyLogLoss
)
from layer.sequential import (
    Sequential
)

FUNCTION_LAYERS = (
    Matmul,
    Sigmoid,
    ReLU,
    Standardization,
    BatchNormalization
)
FUNCTION_LAYER_SCHEMES = {}
for __layer in FUNCTION_LAYERS:
    FUNCTION_LAYER_SCHEMES[__layer.__qualname__.lower()] = __layer

OBJECTIVE_LAYERS = (
    CrossEntropyLogLoss,
)
OBJECTIVE_LAYER_SCHEMES = {}
for __layer in OBJECTIVE_LAYERS:
    OBJECTIVE_LAYER_SCHEMES[__layer.__qualname__.lower()] = __layer

SCHEMES = {
    **FUNCTION_LAYER_SCHEMES,
    **OBJECTIVE_LAYER_SCHEMES
}
assert SCHEMES
