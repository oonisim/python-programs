"""Gradient descent algorithm implementations"""
import numpy as np
from . base import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent """
    def update(self, W, dW) -> None:
        """Default method to update the weight matrix W
        Args:
            W: weight matrix to update
            dW: gradient of dL/dW, the impact on L by dW
        """
        np.subtract(W, self.lr * dW, out=W)
