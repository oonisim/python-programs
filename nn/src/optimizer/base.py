"""Gradient descent base"""
import logging
from typing import (
    Union
)

import numpy as np

from common.constant import (
    TYPE_FLOAT,
    TYPE_TENSOR
)
import function.nn.tf as nn


class Optimizer(nn.Function):
    """Gradient descent optimization base class implementation"""
    # ================================================================================
    # Class
    # ================================================================================
    @classmethod
    def class_id(cls):
        """Identify the class
        Avoid using Python implementation specific __qualname__

        Returns:
            Class identifier
        """
        return cls.__qualname__

    # ================================================================================
    # Instance
    # ================================================================================
    def __init__(
            self,
            name,
            lr=TYPE_FLOAT(0.01),
            l2: TYPE_FLOAT = TYPE_FLOAT(1e-3),
            log_level=logging.WARNING
    ):
        """
        Args:
            lr: learning rate of the gradient descent
            l2: L2 regularization hyper parameter, e.g. 1e-3, set to 0 not to use it
        """
        super().__init__(name=name, log_level=log_level)
        assert isinstance(lr, TYPE_FLOAT) and isinstance(l2, TYPE_FLOAT)
        self.lr = lr
        self.l2 = l2

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Learning rate of the gradient descent"""
        assert isinstance(self._name, str) and len(self._name) > 0
        return self._name

    @property
    def lr(self) -> Union[TYPE_FLOAT, np.ndarray]:
        """Learning rate of the gradient descent"""
        return self._lr

    @lr.setter
    def lr(self, lr: Union[TYPE_FLOAT, np.ndarray]):
        """Set Learning rate"""
        # assert self.is_float_scalar(lr) and (TYPE_FLOAT(0) < lr < TYPE_FLOAT(1))
        assert self.is_float_scalar(lr) and (TYPE_FLOAT(0) < lr)
        self._lr = lr

    @property
    def l2(self) -> Union[TYPE_FLOAT, np.ndarray]:
        """L2 regularization hyper parameter"""
        return self._l2

    @l2.setter
    def l2(self, l2: Union[TYPE_FLOAT, np.ndarray]):
        """Set L2 regularization"""
        assert self.is_float_scalar(l2) and (TYPE_FLOAT(0) < l2 < TYPE_FLOAT(1))
        self._l2 = l2

    @property
    def logger(self) -> logging.Logger:
        """Instance logger"""
        assert isinstance(self._logger, logging.Logger), \
            "logger is not initialized"
        return self._logger

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def differential(self, dW: TYPE_TENSOR, W: TYPE_TENSOR = None, out: TYPE_TENSOR = None):
        """Calculate the differential to update W"""
        raise NotImplementedError("TBD")

    def update(self, W, dW, out=None) -> np.ndarray:
        """Default method to update the weight matrix W
        Args:
            W: weight matrix to update
            dW: gradient of dL/dW, the impact of dW on the system output L
            out: Location into which the result is stored
        Returns:
            Updated W
        """
        raise NotImplementedError("TBD")



