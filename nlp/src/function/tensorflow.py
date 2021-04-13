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
import tensorflow as tf
from common.constant import (
    TYPE_FLOAT,
    TYPE_LABEL,
    BOUNDARY_SIGMOID,
)


Logger = logging.getLogger(__name__)


class Function:
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def is_np_float(X) -> bool:
        """Confirm if X is numpy float
        """
        return \
            np.issubdtype(type(X), np.floating) or \
            isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating)

    @staticmethod
    def is_float_tensor(X) -> bool:
        return tf.is_tensor(X) and X.dtype.is_floating

    @staticmethod
    def to_float_tensor(X):
        return tf.convert_to_tensor(X, dtype=TYPE_FLOAT)

    @staticmethod
    def assure_float_tensor(X) -> tf.Tensor:
        if Function.is_float_tensor(X):
            pass
        elif Function.is_np_float(X):
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
                "sigmoid: X value exceeded the boundary %s, hence clipping.",
                boundary
            )
            if isinstance(X, np.ndarray):
                _X = tf.identity(X)
                tf.where((X > boundary), boundary, X)
                tf.where((X > boundary), boundary, X)
                _X[X > boundary] = boundary
                _X[X < -boundary] = -boundary
            else:  # Scalar
                assert isinstance(X, TYPE_FLOAT)
                _X = np.sign(X) * boundary

        Y = tf.nn.sigmoid(x=_X)
        return Y

    @staticmethod
    @tf.function
    def _softmax(X):
        # --------------------------------------------------------------------------------
        # exp(x-c) to prevent the infinite exp(x) for a large value x, with c = max(x).
        # keepdims=True to be able to broadcast.
        # --------------------------------------------------------------------------------
        C = tf.math.reduce_max(X, axis=1, keepdims=True)
        exp = tf.math.exp(X - C)
        P = tf.math.divide(exp, tf.math.reduce_sum(exp, axis=-1, keepdims=True))
        return P

    @staticmethod
    def softmax(X, out=None) -> object:
        """Softmax P = exp(X) / sum(exp(X))
        Args:
            X: batch input data of shape (N,M).
                N: Batch size
                M: Number of nodes
            out: A location into which the result is stored
        Returns:
            P: Probability of shape (N,M)

        Note:
            https://stackoverflow.com/questions/48824351 for
            Type 'Variable' doesn't have expected attribute '__sub__'
        """
        X = Function.assure_float_tensor(X)
        P = Function._softmax(X)
        return P

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
