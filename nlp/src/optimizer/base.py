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


class Optimizer:
    """Gradient descent optimization base class implementation"""
    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(self, name, lr=0.01, l2: float = 1e-3, log_level=logging.WARNING):
        """
        Args:
            lr: learning rate of the gradient descent
            l2: L2 regularization hyper parameter, e.g. 1e-3, set to 0 not to use it
        """
        self._name = "optimizer"
        self._lr: Union[float, np.ndarray] = lr
        self._l2: Union[float, np.ndarray] = l2

        self._logger = logging.getLogger(name)
        self._logger.setLevel(log_level)

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def lr(self) -> Union[float, np.ndarray]:
        """Learning rate of the gradient descent"""
        assert self._lr and self._lr >= 0
        return self._lr

    @property
    def l2(self) -> Union[float, np.ndarray]:
        """L2 regularization hyper parameter"""
        assert self._l2 and self._l2 >= 0
        return self._l2

    @property
    def logger(self) -> logging.Logger:
        """Instance logger"""
        assert self._logger, "logger is not initialized"
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



