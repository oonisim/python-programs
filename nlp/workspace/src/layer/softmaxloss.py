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


class SoftmaxWithLogLoss(Layer):
    """Softmax cross entropy log loss class
    Combined with the log loss because calculating gradients separately is not simple.
    When combined, the dL/dX, impact on L by input delta dX, is (P - T)/N.
    """
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(self, name: str):
        """Initialize a softmax layer
        Args
            name: Instance ID
        """
        super().__init__(name=name)

        self.P = None  # Probabilities of shape (N, M)
        self.T = None  # Labels of shape (N, M) for OHE or (N,) for index labels.
        self.L = None  # Loss of shape ()/scalar.

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
        self.L = cross_entropy_error(self.P, self.T)
        return self.L

    def backward(self, dP=1):
        """Calculate the gradient dL/dX, the impact on L by the dX form the anterior layer
        Args:
            dP: Gradient dL/dP, impact on L by dP, given from the post layer.
        Returns:
            Gradient dL/dX of shape (N, M)
        """
        N = batch_size = self.T.shape[0]
        if self.T.size == self.P.size:  # 教師データがone-hot-vectorの場合
            dX = (self.P - self.T) / N
        else:
            dX = self.P.copy()
            dX = dP * dX if dP != 1 else dP
            dX[np.arange(batch_size), self.T] -= 1

        return dX / N
