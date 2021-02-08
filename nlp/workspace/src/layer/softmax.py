"""Softmax layer implementation
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
from .base import Layer
from ..optimizer import (
    Optimizer,
    SGD,
)
from .. common.functions import (
    cross_entropy_error,
    softmax
)


class Softmax:
    """Softmax class to calculate the probability
    """
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(self):
        """Initialize a softmax layer"""
        self.P = None  # Probability
        self.T = None  # 教師データ

    def forward(self, X, T):
        """Calculate the node values (probabilities)
        Args:
            X: Input of shape (N, M) to calculate the probabilities for the M nodes.
            T: Labels for input X. Shape(N,M) for OHE label and shape(M) for index label.
        Returns:
            P: Probabilities of shape (N, M)
        """
        self.T = T
        self.P = softmax(X)
        return self.P

    def backward(self, dP=1):
        """Calculate the gradient dL/dX, the impact on L by the dA form the anterior
        Args:
            dP: Gradient dL/dP, impact on L by dP, given from the post layer.
        Returns:
            Gradient dL/dX
        """
        batch_size = self.T.shape[0]
        if self.T.size == self.P.size:  # 教師データがone-hot-vectorの場合
            dX = (self.P - self.T)
        else:
            dX = self.P.copy()
            dX[np.arange(batch_size), self.T] -= 1
            dX = dX

        return dX