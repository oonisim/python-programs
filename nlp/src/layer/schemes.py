from layer.matmul import (
    Matmul
)
from layer.activation import (
    ReLU,
    Sigmoid
)
from layer.objective import (
    CrossEntropyLogLoss,
)
from layer.normalization import (
    Standardization,
    BatchNormalization
)
from layer.sequential import (
    Sequential
)


# ================================================================================
# Dictionaries of layer per purpose
# ================================================================================
FEATURE_LAYERS = (
    Matmul,
)
ACTIVATION_LAYERS = (
    Sigmoid,
    ReLU,
)
NORMALIZATION_LAYERS = (
    Standardization,
    BatchNormalization
)
SEQUENTIAL_LAYERS = (
    Sequential,
)

# --------------------------------------------------------------------------------
# Inference layers
# --------------------------------------------------------------------------------
FUNCTION_LAYERS = \
    FEATURE_LAYERS + \
    NORMALIZATION_LAYERS + \
    ACTIVATION_LAYERS

FUNCTION_LAYER_SCHEMES = {}
for __a_layer in FUNCTION_LAYERS:
    FUNCTION_LAYER_SCHEMES[__a_layer.__qualname__.lower()] = __a_layer

# --------------------------------------------------------------------------------
# Objective layers
# --------------------------------------------------------------------------------
OBJECTIVE_LAYERS = (
    CrossEntropyLogLoss,
)
OBJECTIVE_LAYER_SCHEMES = {}
for __a_layer in OBJECTIVE_LAYERS:
    OBJECTIVE_LAYER_SCHEMES[__a_layer.__qualname__.lower()] = __a_layer

# All layers
SCHEMES = {
    **FUNCTION_LAYER_SCHEMES,
    **OBJECTIVE_LAYER_SCHEMES
}
assert SCHEMES
