"""Gradient descent algorithm implementations"""
import numpy as np
from . base import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent """
    def update(self, W, dW, out=None) -> np.ndarray:
        """Default method to update the weight matrix W
        Args:
            W: weight matrix to update
            dW: gradient of dL/dW, the impact on L by dW
            out: location into which the result is stored
        Return:
            W: A reference to out if specified or a np array allocated.
        """
        return np.subtract(W, self.lr * dW, out=out)
