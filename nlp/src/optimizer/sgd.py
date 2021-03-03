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
        if np.all(np.abs(dW) < np.abs(W / 100.0)):
            self.logger.warning(
                "SGD[%s].update(): Gradient descent potentially stalling with dW < 1% of W.",
                self.name
            )

        # --------------------------------------------------------------------------------
        # Why excluding the bias weight from the regularization?
        # TODO: Remove w0 from the regularization (not include bias weight)
        # --------------------------------------------------------------------------------
        # Overfitting is when the model is sensitive to changes in the input.
        # Bias is fixed (x0=1), hence no change, hence no point to include it
        # --------------------------------------------------------------------------------
        regularization = dW * self.l2
        return np.subtract(W, self.lr * (dW + regularization), out=out)
