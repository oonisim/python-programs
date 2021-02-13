"""Activation layer implementation
Note:
    Python passes a pointer to an object storage if it is mutable.
    Matrices can take up lage storage, hence to avoid temporary copies,
    directly update the target storage area.
"""
from typing import (
    List,
    Dict,
    Optional,
    Final,
    Generator,
    Iterator,
    Callable
)
import logging
import numpy as np
from . base import Layer
from .. optimizer import (
    Optimizer,
    SGD,
)
from .. common.functions import (
    sigmoid
)


class Relu(Layer):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.mask = None    # To zero clear the outputs where x <= 0
        self.A = None       # Activation

    def function(self, X):
        self.mask = (X <= 0)
        A = X.copy()
        A[self.mask] = 0

        return A

    def gradient(self, dA):
        dA[self.mask] = 0
        dX = dA

        return dX


class Sigmoid(Layer):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.A = None   # Activation

    def function(self, X):
        A = sigmoid(X)
        self.A = A
        return A

    def gradient(self, dA):
        dX = dA * (1.0 - self.A) * self.A

        return dX
