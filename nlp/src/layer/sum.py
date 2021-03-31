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

from common.constants import (
    TYPE_FLOAT
)
from layer.base import Layer
from layer.constants import (
    _NAME,
    _NUM_NODES,
    _LOG_LEVEL,
)


class Sum(Layer):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def build(parameters: Dict):
        assert (
            _NAME in parameters and
            _NUM_NODES in parameters
        )

        return Sum(
            name=parameters[_NAME],
            num_nodes=parameters[_NUM_NODES],
            log_level=parameters[_LOG_LEVEL] if _LOG_LEVEL in parameters else logging.ERROR
        )

    # ================================================================================
    # Instance
    # ================================================================================
    @property
    def is_averaged(self) -> bool:
        """Flat if the output Y is averaged by N"""
        return self._is_averaged

    @property
    def axis(self) -> int:
        """axis to sum along"""
        return self._axis

    def __init__(
            self,
            name: str,
            num_nodes: int,
            axis: int = 0,
            is_averaged: bool = False,
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: layer name
            num_nodes: number of nodes in the layer
            axis: axis along which to sum
            is_averaged: flag to average the output
            log_level: logging level
        """
        assert isinstance(axis, int) and axis == 0, "Currently axis=0 is only supported"
        assert isinstance(is_averaged, bool)

        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        self.mask: np.ndarray = np.empty(())    # To zero clear the outputs where x <= 0
        self._M: int = num_nodes                     # Number of nodes alias
        self._shape_X = ()                      # Original shape of X
        self._is_averaged: bool = is_averaged
        self._axis: int = axis

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X) -> np.ndarray:
        self._shape_X = X.shape
        self.X = X

        if self._Y.size <= 0:
            shape_Y = X[0].shape if X.ndim > 0 else ()
            self._Y = np.empty(shape_Y, dtype=TYPE_FLOAT)

        np.sum(X, axis=self.axis, out=self._Y) \
            if self.is_averaged \
            else np.sum(X/self.N, out=self._Y)

        return self.Y

    def gradient(self, dY: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate gradient dL/dX=(dL/dY * dY/dX) to back-propagate
        to the previous layer. dY has the same shape (N,M) with A as L is scalar.

        Args
            dY: dL/dY of shape (1,M) or (M,), impact on L by the activation dY.
        Returns:
            dL/dX: impact on L by the layer input dX

        TODO: How to restore the shape when axis != 0. Need to insert np.newaxis?
        """
        dY = np.array(dY).reshape((1, -1)) if isinstance(dY, float) else dY
        assert dY.shape == self.Y.shape, \
            f"dY shape should be {self.Y.shape} but {dY.shape}."

        self._dX[:] = (TYPE_FLOAT(1) / self.N) \
            if self.is_averaged else TYPE_FLOAT(1)
        np.multiply(self.dX, dY, out=self._dX)

        assert self.dX.shape == self._shape_X, \
            "dX.shape %s must match original X shape %s" \
            % (self.dX.shape, self._shape_X)
        return self.dX
