from layer.sequential import (
    Sequential
)
from layer.parallel import (
    Parallel
)
COMPOSITE_LAYERS = (
    Sequential,
    Parallel,
)
COMPOSITE_LAYER_SCHEMES = {}
for _layer in COMPOSITE_LAYERS:
    COMPOSITE_LAYER_SCHEMES[_layer.__qualname__] = _layer
