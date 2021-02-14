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
import numpy as np


class Optimizer:
    """Gradient descent optimization base class implementation"""
    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(self, lr=0.01, l2: float = 1e-3):
        """
        Args:
            lr: learning rate of the gradient descent
            l2: L2 regularization hyper parameter, e.g. 1e-3, set to 0 not to use it
        """
        self._lr: Union[float, np.ndarray] = lr
        self._l2: Union[float, np.ndarray] = l2

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



