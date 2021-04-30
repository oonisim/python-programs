"""Non composite layer class information"
DO NOT import composite layer classes in this file.
"""
from layer.matmul import (
    Matmul
)
from layer.sum import (
    Sum
)
from layer.preprocessing import (
    EventIndexing
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
    FeatureScaleShift,
    BatchNormalization
)

# DO NOT import these
# from layer.sequential import (
#     Sequential
# )

# ================================================================================
# Dictionaries of layer per purpose
# ================================================================================
PREPROCESS_LAYERS = (
    EventIndexing,
)
FEATURE_LAYERS = (
    Matmul,
    Sum
)
ACTIVATION_LAYERS = (
    Sigmoid,
    ReLU,
)
NORMALIZATION_LAYERS = (
    Standardization,
    FeatureScaleShift,
    BatchNormalization
)

# --------------------------------------------------------------------------------
# Inference layers
# --------------------------------------------------------------------------------
FUNCTION_LAYERS = \
    FEATURE_LAYERS + \
    NORMALIZATION_LAYERS + \
    ACTIVATION_LAYERS

FUNCTION_LAYER_SCHEMES = {}
for _layer in FUNCTION_LAYERS:
    FUNCTION_LAYER_SCHEMES[_layer.class_id()] = _layer

# --------------------------------------------------------------------------------
# Objective layers
# --------------------------------------------------------------------------------
OBJECTIVE_LAYERS = (
    CrossEntropyLogLoss,
)
OBJECTIVE_LAYER_SCHEMES = {}
for _layer in OBJECTIVE_LAYERS:
    OBJECTIVE_LAYER_SCHEMES[_layer.class_id()] = _layer

# All layers
SCHEMES = {
    **FUNCTION_LAYER_SCHEMES,
    **OBJECTIVE_LAYER_SCHEMES
}
assert SCHEMES
