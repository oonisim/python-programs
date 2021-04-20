import logging
import numpy as np
import numexpr as ne
from common.constant import (
    TYPE_FLOAT,
    BOUNDARY_SIGMOID,
    ENABLE_NUMEXPR
)
import function.common.base as base
import logging

import numexpr as ne
import numpy as np

import function.common.base as base
from common.constant import (
    TYPE_FLOAT,
    BOUNDARY_SIGMOID,
    ENABLE_NUMEXPR
)

Logger = logging.getLogger(__name__)


class Function(base.Function):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def sigmoid(
            X,
            boundary: TYPE_FLOAT = BOUNDARY_SIGMOID,
            out=None
    ):
        """Sigmoid activate function
        Args:
            X: Domain value
            boundary: The lower boundary of acceptable X value.
            out: A location into which the result is stored

        NOTE:
            epsilon to prevent causing inf e.g. log(X+e) has a consequence of clipping
            the derivative which can make numerical gradient unstable. For instance,
            due to epsilon, log(X+e+h) and log(X+e-h) will get close or same, and
            divide by 2*h can cause catastrophic cancellation.

            To prevent such instability, limit the value range of X with boundary.
        """
        X = super().assure_float_tensor(X)
        boundary = BOUNDARY_SIGMOID \
            if (boundary is None or boundary <= TYPE_FLOAT(0)) else boundary
        assert boundary > 0

        if np.all(np.abs(X) <= boundary):
            _X = X
        else:
            Logger.warning(
                "sigmoid: X value exceeded the boundary %s, hence clipping.", boundary
            )
            if isinstance(X, np.ndarray):
                _X = np.copy(X)
                _X[X > boundary] = boundary
                _X[X < -boundary] = -boundary
            else:  # Scalar
                assert isinstance(X, TYPE_FLOAT)
                _X = np.sign(X) * boundary

        if ENABLE_NUMEXPR:
            Y = ne.evaluate("1 / (1 + exp(-1 * _X))", out=out)
        else:
            if out is not None:
                assert _X.shape == out.shape, f"Output shape must match X shape {_X.shape}"
                np.exp(-1 * _X, out=out)
                np.add(1, out, out=out)
                Y = np.divide(1, out, out=out)
            else:
                Y = 1 / (1 + np.exp(-1 * _X))

        return Y

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: ID name
        """
        super().__init__(name=name, log_level=log_level)
