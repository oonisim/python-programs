"""Adapter layer implementation
Responsibility: Bridge between two layers as an adapter.
    Transform a layer output to match the input of another layer.
    a. anterior_layer.function -> adapter.function -> posterior_layer.function
    b. anterior_layer.gradient <- adapter.gradient <- posterior_layer.gradient

Instantiation:
    Provide layer function(), gradient() and update() methods and set

Function signature:
    def function(X: TYPE_TENSOR, adapter: Layer):
    def gradient(dY: TYPE_TENSOR, adapter: Layer):
    def update(adapter: Layer):
"""
import logging
from typing import (
    List,
    Dict,
    Any,
    Union,
    Callable,
)

from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR
)
from layer.base import Layer


class Adapter(Layer):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def build(parameters: Dict):
        raise NotImplementedError("TBD")

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            num_nodes: TYPE_INT,
            function: Callable[[Any, Layer], Any],
            gradient: Callable[[Any, Layer], Any],
            update: Callable[[Layer], List[Any]] = lambda: [],
            log_level: int = logging.ERROR
    ):
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert callable(function) and callable(gradient)
        self._function = function
        self._gradient = gradient
        self._update = update

    def function(self, X: TYPE_TENSOR) -> TYPE_TENSOR:
        self.X = X
        self._Y = self._function(X, self)
        return self.Y

    def gradient(self, dY: Union[TYPE_TENSOR, TYPE_FLOAT]) -> Union[TYPE_TENSOR, TYPE_FLOAT]:
        self.dY = dY
        self._dX = self._gradient(dY, self)
        return self.dX
