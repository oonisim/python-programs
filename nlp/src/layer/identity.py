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


class Identity(Layer):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def build(parameters: Dict):
        assert (
            _NAME in parameters and
            _NUM_NODES in parameters
        )

        return Identity(
            name=parameters[_NAME],
            num_nodes=parameters[_NUM_NODES],
            log_level=parameters[_LOG_LEVEL] if _LOG_LEVEL in parameters else logging.ERROR
        )

    # ================================================================================
    # Instance
    # ================================================================================
    def __init__(self, name: str, num_nodes: int, log_level: int = logging.ERROR):
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        self.mask: np.ndarray = np.empty(())    # To zero clear the outputs where x <= 0
        self._M = num_nodes                     # Number of nodes alias

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X) -> np.ndarray:
        self.X = X
        return self.X

    def gradient(self, dY: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate gradient dL/dX=(dL/dY * dY/dX) to back-propagate
        to the previous layer. dY has the same shape (N,M) with A as L is scalar.

        Args
            dY: dL/dY of shape (1,M) or (M,), impact on L by the activation dY.
        Returns:
            dL/dX: impact on L by the layer input dX
        """
        return dY
