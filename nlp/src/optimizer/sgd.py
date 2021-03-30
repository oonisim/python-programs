"""Gradient descent algorithm implementations"""
from typing import (
    Dict
)
import logging
import numpy as np
import numexpr as ne
from . base import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent """
    # ================================================================================
    # Class initialization
    # ================================================================================
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def template(specification: Dict):
        return {
            "scheme": SGD.__qualname__,
            "parameters": {
                "lr": 0.01,
                "l2": 1e-3
            }
        }

    @staticmethod
    def build(parameters: Dict):
        """Build an optimizer based on the specification.
        Spec example:
        {
            "name": "sgd",  # optional
            "lr": 0.1,
            "l2": 0.1
        }
        """
        return SGD(**parameters)

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(self, name="SGD", lr=0.01, l2: float = 1e-3, log_level=logging.ERROR):
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
        # --------------------------------------------------------------------------------
        # Gradient can be zero. e.g for a Batch Normalization layer, when a feature xi
        # in a batch has the same value, such as a specific pixel in images is all black
        # then the standardized value xi_std = 0 -> dL/dGamma = sum(dL/dY * xi_std) = 0.
        # --------------------------------------------------------------------------------
        if np.all(np.abs(dW) < np.abs(W / 100.0)):
            self.logger.warning(
                "SGD[%s].update(): Gradient descent potentially stalling with dW < W/100.",
                self.name
            )

        # --------------------------------------------------------------------------------
        # Why excluding the bias weight from the regularization?
        # TODO: Remove w0 from the regularization (not include bias weight)
        # --------------------------------------------------------------------------------
        # Overfitting is when the model is sensitive to changes in the input.
        # Bias is fixed (x0=1), hence no change, hence no point to include it
        # --------------------------------------------------------------------------------
        l2 = self.l2
        lr = self.lr
        # return np.subtract(W, dW * (1 + l2), out=out)
        return ne.evaluate("W - lr * dW * (1 + l2)", out=out)
