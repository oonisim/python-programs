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

import numexpr as ne
import numpy as np

from common.constant import (
    TYPE_FLOAT,
    TYPE_TENSOR,
    ENABLE_NUMEXPR
)
from common.function import (
    sigmoid
)
from layer.base import Layer
from layer.constants import (
    _NAME,
    _SCHEME,
    _NUM_NODES,
    _PARAMETERS,
    _LOG_LEVEL,
    RELU_LEAKY_SLOPE
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
            slope: TYPE_FLOAT = RELU_LEAKY_SLOPE
    ):
        """Generate ReLU specification
        Args:
            name: layer name
            num_nodes: number of nodes (outputs) in the layer
            slope: leaky slope value
        """
        return {
            _SCHEME: ReLU.class_id(),
            _PARAMETERS: {
                _NAME: name,
                _NUM_NODES: num_nodes,
                "slope": slope
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
            slope=parameters["slope"] if "slope" in parameters else TYPE_FLOAT(0),
            log_level=parameters[_LOG_LEVEL] if _LOG_LEVEL in parameters else logging.ERROR
        )

    # ================================================================================
    # Instance
    # ================================================================================
    @property
    def slope(self) -> Union[TYPE_FLOAT, np.ndarray]:
        """Leaky ReLU slope value"""
        return self._slope

    def __init__(
            self,
            name: str,
            num_nodes: int,
            slope: TYPE_FLOAT = RELU_LEAKY_SLOPE,
            log_level:
            int = logging.ERROR
    ):
        """
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            slope: leaky slope value
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        self.mask: np.ndarray = np.empty(0)    # To zero clear the outputs where x <= 0
        self._A: np.ndarray = np.empty(0)      # Activation
        self._M = num_nodes                     # Number of nodes alias
        self._slope: TYPE_FLOAT = slope
        assert self.slope >= TYPE_FLOAT(0), "Leaky ReLU slope value needs to be positive."

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X) -> Union[np.ndarray, TYPE_FLOAT]:
        X = np.array(X).reshape((1, -1)) if isinstance(X, TYPE_FLOAT) else X
        assert X.shape[1] == self.M, \
            f"Number of node X {X.shape[1] } does not match {self.M}."

        self.X = X
        # --------------------------------------------------------------------------------
        # Allocate array storage for np.func(out=) for Y but not dY.
        # gradient() need to validate the dY shape is (N,M)
        # --------------------------------------------------------------------------------
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty(X.shape, dtype=TYPE_FLOAT)

        np.copyto(self._Y, X)
        self.mask = (X <= TYPE_FLOAT(0.0))
        # --------------------------------------------------------------------------------
        # Leaky ReLU When slope > 0
        # --------------------------------------------------------------------------------
        # self._Y[self.mask] = 0
        self._Y[self.mask] *= self.slope

        return self.Y

    def gradient(self, dA: Union[np.ndarray, TYPE_FLOAT]) -> Union[np.ndarray, TYPE_FLOAT]:
        """Calculate gradient dL/dX=(dL/dA * dA/dX) to back-propagate
        to the previous layer. dA has the same shape (N,M) with A as L is scalar.

        Args
            dA: dL/dA of shape (N,M), impact on L by the activation dA.
        Returns:
            dL/dX: impact on L by the layer input dX
        """
        dA = np.array(dA).reshape((1, -1)) if isinstance(dA, TYPE_FLOAT) else dA
        assert dA.shape == (self.N, self.M), \
            f"dA shape should be {(self.N, self.M)} but {dA.shape}."
        self._dY = dA

        np.copyto(self._dX, dA)
        # --------------------------------------------------------------------------------
        # Leaky ReLU When slope > 0
        # --------------------------------------------------------------------------------
        # self._dX[self.mask] = 0
        self._dX[self.mask] = self.slope
        return self.dX


class Sigmoid(Layer):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return Sigmoid.specification(name="relu001", num_nodes=3)

    @staticmethod
    def specification(
            name: str,
            num_nodes: int,
    ):
        """Generate Sigmoid specification
        Args:
            name: layer name
            num_nodes: number of nodes (outputs) in the layer
        """
        return {
            _SCHEME: Sigmoid.class_id(),
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
        self.mask: np.ndarray = np.empty(0)    # To zero clear the outputs where x <= 0
        self._A: np.ndarray = np.empty(0)       # Activation
        self._M = num_nodes                     # Number of nodes alias

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[TYPE_FLOAT, np.ndarray]) -> Union[TYPE_FLOAT, np.ndarray]:
        X = np.array(X).reshape((1, -1)) if isinstance(X, TYPE_FLOAT) else X
        assert X.shape[1] == self.M, \
            f"Number of node X {X.shape[1] } does not match {self.M}."

        self.X = X
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty(X.shape, dtype=TYPE_FLOAT)

        self._Y = sigmoid(X, out=self._Y)
        return self.Y

    def gradient(self, dA: TYPE_TENSOR) -> TYPE_TENSOR:
        """Calculate gradient dL/dX=(dL/dA * dA/dX) to back-propagate
        to the previous layer. dA has the same shape (N,M) with A as L is scalar.

        Args
            dA: dL/dA of shape (N,M), impact on L by the activation dA.
        Returns:
            dL/dX: impact on L by the layer input dX
        """
        if isinstance(dA, TYPE_FLOAT):
            dA = np.array(dA, dtype=TYPE_FLOAT).reshape((1, -1))

        assert dA.shape == (self.N, self.M), \
            f"dA shape should be {(self.N, self.M)} but {dA.shape}."

        self._dY = dA
        if ENABLE_NUMEXPR:
            Y = self.Y
            ne.evaluate("dA * (1.0 - Y) * Y", out=self._dX)
        else:
            self._dX = dA * (TYPE_FLOAT(1.0) - self.Y) * self.Y

        return self.dX
