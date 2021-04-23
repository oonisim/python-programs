import logging
from typing import (
    Union,
    Callable
)

import numpy as np
import tensorflow as tf

import function.nn.base as base
from common.constant import (
    TYPE_FLOAT,
    TYPE_TENSOR,
    BOUNDARY_SIGMOID,
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
        X = super(Function, Function).assure_float_tensor(X)
        boundary = BOUNDARY_SIGMOID \
            if (boundary is None or boundary <= TYPE_FLOAT(0)) else boundary
        assert boundary > 0

        if tf.reduce_all(np.abs(X) <= boundary):
            _X = X
        else:
            Logger.warning(
                "sigmoid: X value exceeded the boundary %s, hence clipping.",
                boundary
            )
            if isinstance(X, np.ndarray):
                _X = tf.Variable(X)
                _X.assign(tf.where((X > boundary), boundary, X))
                _X.assign(tf.where((X < -boundary), -boundary, X))
            else:  # Scalar
                assert super().is_float_scalar(X)
                _X = tf.constant(tf.math.sign(X) * boundary)

        Y = tf.nn.sigmoid(x=_X)
        return Y.numpy()

    @staticmethod
    def softmax(X: Union[TYPE_FLOAT, TYPE_TENSOR], axis=None, out=None) -> Union[TYPE_FLOAT, TYPE_TENSOR]:
        """Softmax P = exp(X) / sum(exp(X))
        Args:
            X: batch input data of shape (N,M).
                N: Batch size
                M: Number of nodes
            axis: The dimension softmax would be performed on.
            out: A location into which the result is stored
        Returns:
            P: Probability of shape (N,M)

        Note:
            https://stackoverflow.com/questions/48824351 for
            Type 'Variable' doesn't have expected attribute '__sub__'
        """
        X = super(Function, Function).assure_float_tensor(X)
        P = tf.nn.softmax(logits=X, axis=axis)
        return P

    @staticmethod
    def einsum(equation, *inputs, **kwargs) -> TYPE_TENSOR:
        return tf.einsum(equation, *inputs, **kwargs).numpy()

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
