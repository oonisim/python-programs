"""Gradient descent algorithm implementations"""
import logging
import numpy as np
from . base import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent """
    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(self, name="SGD", lr=0.01, l2: float = 1e-3, log_level=logging.WARNING):
        super().__init__(name=name, lr=lr, l2=l2, log_level=log_level)

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    def update(self, W, dW, out=None) -> np.ndarray:
        """Default method to update the weight matrix W
        Args:
            W: weight matrix to update
            dW: gradient of dL/dW, the impact on L by dW
            out: location into which the result is stored
        Return:
            W: A reference to out if specified or a np array allocated.
        """
        if self.logger.level == logging.WARNING and np.all(dW < dW / 100.0):
            self.logger.warning(
                "update(): Gradient descent potentially stalling with dW < 1% of W."
            )

        regularization = dW * self.l2
        return np.subtract(W, self.lr * (dW + regularization), out=out)
