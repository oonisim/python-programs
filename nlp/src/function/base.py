from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable,
    Any,
    NoReturn,
    Final
)
import logging
import numpy as np
import numexpr as ne
from common.constant import (
    TYPE_FLOAT,
    TYPE_LABEL,
    BOUNDARY_SIGMOID,
    ENABLE_NUMEXPR
)

Logger = logging.getLogger(__name__)


class Function:
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def is_float_scalar(X) -> bool:
        """Confirm if X is float scalar()
        """
        return np.issubdtype(type(X), np.floating)

    @staticmethod
    def is_float_tensor(X) -> bool:
        """Confirm if X is float tensor
        The implementation depends on the framework e.g. numpy, tensorflow.
        """
        return \
            isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating)

    @staticmethod
    def to_float_tensor(X) -> object:
        return np.ndarray(X, dtype=TYPE_FLOAT)

    @staticmethod
    def assure_float_tensor(X) -> np.ndarray:
        if Function.is_float_tensor(X):
            pass
        elif Function.is_float_scalar(X):
            X = Function.to_float_tensor(X)
        else:
            raise AssertionError(f"Float compatible type expected but {str(X)}")

        return X

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
        X = Function.assure_float_tensor(X)
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
            Y = 1 / (1 + np.exp(-1 * _X))

        return Y

    # ================================================================================
    # Instance
    # ================================================================================

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A unique name to identify a layer"""
        return self._name

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
        assert name
        self._name: str = name

        # number of nodes in the layer
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])
