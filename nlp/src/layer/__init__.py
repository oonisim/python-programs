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

SCHEMES = {}
FUNCTION_LAYERS = (
    Matmul,
    Sigmoid,
    ReLU,
    Standardization,
    BatchNormalization
)
assert FUNCTION_LAYERS
for __layer in FUNCTION_LAYERS:
    SCHEMES[__layer.__qualname__.lower()] = __layer

OBJECTIVE_LAYERS = (
    CrossEntropyLogLoss,
)
assert OBJECTIVE_LAYERS
for __layer in OBJECTIVE_LAYERS:
    SCHEMES[__layer.__qualname__.lower()] = __layer

assert SCHEMES
