"""Activation layer implementation
Note:
    Python passes a pointer to an object storage if it is mutable.
    Matrices can take up lage storage, hence to avoid temporary copies,
    directly update the target storage area.
"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Final,
    Generator,
    Iterator,
    Callable
)
import logging
import copy
import numpy as np
from base import Layer
from common.functions import (
    sigmoid
)


class Relu(Layer):
    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(self, name: str, num_nodes: int, log_level: int = logging.ERROR):
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        self.mask: np.ndarray = np.empty(())    # To zero clear the outputs where x <= 0
        self._A: np.ndarray = np.empty(())      # Activation
        self._M = num_nodes                     # Number of nodes alias

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def M(self) -> int:
        """Batch size"""
        assert self._M > 0, "M is not initialized"
        return self._M

    @property
    def A(self) -> np.ndarray:
        """Activation"""
        assert self._A.size > 0, "A is not initialized"
        return self._A

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X) -> Union[np.ndarray, float]:
        X = np.array(X).reshape((1, -1)) if isinstance(X, float) else X
        assert X.shape[1] == self.M, \
            f"Number of node X {X.shape[1] } does not match {self.M}."

        self.X = X
        self.mask = (X <= 0)
        A = copy.deepcopy(X)
        A[self.mask] = 0

        return A

    def gradient(self, dA: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate gradient dL/dX=(dL/dA * dA/dX) to back-propagate
        to the previous layer. dA has the same shape (N,M) with A as L is scalar.

        Args
            dA: dL/dA of shape (N,M), impact on L by the activation dA.
        Returns:
            dL/dX: impact on L by the layer input dX
        """
        dA = np.array(dA).reshape((1, -1)) if isinstance(dA, float) else dA
        assert dA.shape == (self.N, self.M), \
            f"dA shape should be {(self.N, self.M)} but {dA.shape}."
        dX: np.ndarray = copy.deepcopy(dA)
        dX[self.mask] = 0
        return dX


class Sigmoid(Layer):
    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(self, name: str, num_nodes: int, log_level: int = logging.ERROR):
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        self.mask: np.ndarray = np.empty(())    # To zero clear the outputs where x <= 0
        self._A: np.ndarray = np.empty(())       # Activation
        self._M = num_nodes                     # Number of nodes alias

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def M(self) -> int:
        """Batch size"""
        assert self._M > 0, "M is not initialized"
        return self._M

    @property
    def A(self) -> np.ndarray:
        """Activation"""
        assert self._A.size > 0, "A is not initialized"
        return self._A

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        X = np.array(X).reshape((1, -1)) if isinstance(X, float) else X
        assert X.shape[1] == self.M, \
            f"Number of node X {X.shape[1] } does not match {self.M}."

        self._X = X
        self._A = sigmoid(X)
        return self.A

    def gradient(self, dA: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate gradient dL/dX=(dL/dA * dA/dX) to back-propagate
        to the previous layer. dA has the same shape (N,M) with A as L is scalar.

        Args
            dA: dL/dA of shape (N,M), impact on L by the activation dA.
        Returns:
            dL/dX: impact on L by the layer input dX
        """
        dA = np.array(dA).reshape((1, -1)) if isinstance(dA, float) else dA
        assert dA.shape == (self.N, self.M), \
            f"dA shape should be {(self.N, self.M)} but {dA.shape}."
        dX = dA * (1.0 - self.A) * self.A
        return dX
