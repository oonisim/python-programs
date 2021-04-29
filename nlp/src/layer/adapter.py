"""Adapter layer implementation
Responsibility: Bridge between two layers as an adapter.
    Transform a layer output to match the input of another layer.
    a. anterior_layer.function -> adapter.function -> posterior_layer.function
    b. anterior_layer.gradient <- adapter.gradient <- posterior_layer.gradient
"""
import copy
import logging
from typing import (
    Optional,
    Union,
    Callable, Dict
)

import numpy as np
from memory_profiler import profile as memory_profile

from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR,
    EVENT_VECTOR_SIZE,
    EVENT_META_ENTITIES
)
from common.function import (
    numerical_jacobian,
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
            function: Callable,
            gradient: Callable,
            log_level: int = logging.ERROR
    ):
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert callable(function) and callable(gradient)
        self._function = function
        self._gradient = gradient

    def function(self, X: TYPE_TENSOR) -> TYPE_TENSOR:
        self._Y = self._function(X)
        return self.Y

    def gradient(self, dY: Union[TYPE_TENSOR, TYPE_FLOAT]) -> Union[TYPE_TENSOR, TYPE_FLOAT]:
        self.dY = self._gradient(dY)
        return self.dY

