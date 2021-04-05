"""Activation layer implementation
Note:
    Python passes a pointer to an object storage if it is mutable.
    Matrices can take up lage storage, hence to avoid temporary copies,
    directly update the target storage area.
"""
import logging
from typing import (
    Union,
    Dict
)

import numpy as np
import numexpr as ne
from common.constants import (
    TYPE_FLOAT,
    ENABLE_NUMEXPR
)
from common.function import (
    sigmoid
)
from layer.base import Layer
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS,
    _LOG_LEVEL
)


class ReLU(Layer):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return ReLU.specification(name="relu001", num_nodes=3)

    @staticmethod
    def specification(
            name: str,
            num_nodes: int,
    ):
        """Generate ReLU specification
        Args:
            name: layer name
            num_nodes: number of nodes (outputs) in the layer
        """
        return {
            _SCHEME: ReLU.__qualname__,
            _PARAMETERS: {
                _NAME: name,
                _NUM_NODES: num_nodes
            }
        }

    @staticmethod
    def build(parameters: Dict):
        assert (
            _NAME in parameters and
            _NUM_NODES in parameters
        )

        return ReLU(
            name=parameters[_NAME],
            num_nodes=parameters[_NUM_NODES],
            log_level=parameters[_LOG_LEVEL] 
            if _LOG_LEVEL in parameters else logging.ERROR
        )

    # ================================================================================
    # Instance
    # ================================================================================
    def __init__(self, name: str, num_nodes: int, log_level: int = logging.ERROR):
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        self.mask: np.ndarray = np.empty(())    # To zero clear the outputs where x <= 0
        self._A: np.ndarray = np.empty(())      # Activation
        self._M = num_nodes                     # Number of nodes alias

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X) -> Union[np.ndarray, float]:
        X = np.array(X).reshape((1, -1)) if isinstance(X, float) else X
        assert X.shape[1] == self.M, \
            f"Number of node X {X.shape[1] } does not match {self.M}."

        self.X = X
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty(X.shape, dtype=TYPE_FLOAT)

        self.mask = (X <= 0)
        # Y = np.copy(X)
        # Y[self.mask] = 0
        # self._Y = Y
        np.copyto(self._Y, X)
        self._Y[self.mask] = 0
        return self.Y

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
        self._dY = dA

        # dX = np.copy(dA)
        # dX[self.mask] = 0
        # self._dX = dX
        np.copyto(self._dX, dA)
        self._dX[self.mask] = 0
        return self.dX


class Sigmoid(Layer):
    # ================================================================================
    # Class initialization
    # ================================================================================
    @staticmethod
    def build(parameters: Dict):
        assert (
            _NAME in parameters and
            _NUM_NODES in parameters
        )

        return Sigmoid(
            name=parameters[_NAME],
            num_nodes=parameters[_NUM_NODES],
            log_level=parameters[_LOG_LEVEL] if _LOG_LEVEL in parameters else logging.ERROR
        )

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

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        X = np.array(X).reshape((1, -1)) if isinstance(X, float) else X
        assert X.shape[1] == self.M, \
            f"Number of node X {X.shape[1] } does not match {self.M}."

        self._X = X
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty(X.shape, dtype=TYPE_FLOAT)

        self._Y = sigmoid(X, out=self._Y)
        return self.Y

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

        self._dY = dA
        if ENABLE_NUMEXPR:
            Y = self.Y
            ne.evaluate("dA * (1.0 - Y) * Y", out=self._dX)
        else:
            self._dX = dA * (1.0 - self.Y) * self.Y

        return self.dX

