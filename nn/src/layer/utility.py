import logging
from typing import (
    List
)

from layer.base import (
    Layer
)

Logger = logging.getLogger(__name__)


def forward_outputs(layers: List[Layer], X):
    outputs = []
    for __layers in layers:
        Y = __layers.function(X)
        outputs.append(Y)
        X = Y
    return outputs


def backward_outputs(layers: List[Layer], dY):
    outputs = []
    for __layers in layers[::-1]:
        dX = __layers.gradient(dY)
        outputs.append(dX)
        dY = dX
    return outputs


