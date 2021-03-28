"""Gradient descent base"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Final,
    Generator,
    Iterator,
    Callable
)
import logging
import numpy as np
from common.constants import (
    TYPE_FLOAT
)


class Optimizer:
    """Gradient descent optimization base class implementation"""
    # ================================================================================
    # Instance
    # ================================================================================
    def __init__(self, name, lr=0.01, l2: TYPE_FLOAT = 1e-3, log_level=logging.WARNING):
        """
        Args:
            lr: learning rate of the gradient descent
            l2: L2 regularization hyper parameter, e.g. 1e-3, set to 0 not to use it
        """
        self._name = name
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
        assert \
            (
                isinstance(lr, TYPE_FLOAT) or
                (isinstance(lr, np.ndarray) and lr.dtype == TYPE_FLOAT)
            ) and (0.0 < lr < 1.0)

        self._lr = lr

    @property
    def l2(self) -> Union[TYPE_FLOAT, np.ndarray]:
        """L2 regularization hyper parameter"""
        return self._l2

    @l2.setter
    def l2(self, l2: Union[TYPE_FLOAT, np.ndarray]):
        """Set L2 regularization"""
        assert \
            (
                isinstance(l2, TYPE_FLOAT) or
                (isinstance(l2, np.ndarray) and l2.dtype == TYPE_FLOAT)
            ) and (0.0 < l2 < 1.0)

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
    def update(self, W, dW, out=None) -> np.ndarray:
        """Default method to update the weight matrix W
        Args:
            W: weight matrix to update
            dW: gradient of dL/dW, the impact of dW on the system output L
            out: Location into which the result is stored
        Returns:
            Updated W
        """
        pass



