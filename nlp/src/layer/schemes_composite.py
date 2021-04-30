from layer.sequential import (
    Sequential
)
COMPOSITE_LAYERS = (
    Sequential,
)
COMPOSITE_LAYER_SCHEMES = {}
for _layer in COMPOSITE_LAYERS:
    COMPOSITE_LAYER_SCHEMES[_layer.class_id()] = _layer
