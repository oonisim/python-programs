"""Gradient descent algorithm implementations"""
from typing import (
    Dict
)
import logging
import numpy as np
import numexpr as ne
from common.constant import (
    TYPE_FLOAT,
    TYPE_TENSOR
)
from . base import Optimizer

_SGD_NAME_DEFAULT = "sgd"
_SGD_LR_DEFAULT = TYPE_FLOAT(1e-2)
_SGD_L2_DEFAULT = TYPE_FLOAT(1e-3)


class SGD(Optimizer):
    """Stochastic gradient descent """
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return SGD.specification()

    @staticmethod
    def specification(
            name: str = _SGD_NAME_DEFAULT,
            lr: TYPE_FLOAT = _SGD_LR_DEFAULT,
            l2: TYPE_FLOAT = _SGD_L2_DEFAULT
    ):
        """Generate SGD specification
        Args:
            name: optimizer name
            lr: learning rate
            l2: L2 regularization parameter
        Returns:
            specification
        """
        return {
            "scheme": SGD.class_id(),
            "parameters": {
                "name": name,
                "lr": lr,
                "l2": l2
            }
        }

    @staticmethod
    def build(parameters: Dict):
        """Build an optimizer based on the specification.
        """
        if "lr" in parameters:
            parameters["lr"] = TYPE_FLOAT(parameters["lr"])
            parameters["l2"] = TYPE_FLOAT(parameters["l2"])
        return SGD(**parameters)

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name=_SGD_NAME_DEFAULT,
            lr: TYPE_FLOAT = TYPE_FLOAT(0.01),
            l2: TYPE_FLOAT = TYPE_FLOAT(1e-3),
            log_level=logging.ERROR
    ):
        super().__init__(name=name, lr=lr, l2=l2, log_level=log_level)

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def differential(self, dW: TYPE_TENSOR, W: TYPE_TENSOR = None, out: TYPE_TENSOR = None):
        """Calculate the differential to update W"""
        if out is not None:
            raise NotImplementedError("out is not supported for TF")
        return self.multiply(x=dW, y=self.lr * (TYPE_FLOAT(1.0) + self.l2))

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
        if np.all(np.abs(dW) < np.abs(W / TYPE_FLOAT(100.0))):
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
        scale = TYPE_FLOAT(1.0) + l2
        # return np.subtract(W, dW * scale, out=out)
        return ne.evaluate("W - lr * dW * scale", out=out)
