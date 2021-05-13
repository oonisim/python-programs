import logging

import numexpr as ne
import numpy as np

import function.nn.base as base
from common.constant import (
    TYPE_FLOAT,
    TYPE_TENSOR,
    BOUNDARY_SIGMOID,
    ENABLE_NUMEXPR
)

Logger = logging.getLogger(__name__)


class Function(base.Function):
    # ================================================================================
    # Class
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Tensors
    # --------------------------------------------------------------------------------
    @staticmethod
    def ones(shape=None, dtype=TYPE_FLOAT):
        return np.ones(
            shape=shape, dtype=dtype
        )

    # --------------------------------------------------------------------------------
    # Tensor validations
    # --------------------------------------------------------------------------------
    @staticmethod
    def is_finite(X) -> bool:
        return np.all(np.isfinite(X))

    @staticmethod
    def all(x, axis=None) -> bool:
        """Check if all true"""
        return np.all(x, axis=axis)

    @staticmethod
    def all_close(x, y):
        return np.allclose(
            x, y, atol=TYPE_FLOAT(1e-5)
        )

    @staticmethod
    def all_equal(x, y):
        return np.array_equal(x, y)

    @staticmethod
    def in1d(target, source, invert: bool = False):
        """
        Return boolean tensor of the target shape that tells if an element of
        'target' b is in 'source'.

        Example:
            target = np.array([1,2,3,4,5])
            source = np.array([1,3,5])
            is_in(target, source)
            -----
            [ True, False,  True, False,  True]

        Args:
            target: Target 1D array to test
            source: list of elements to check if in 'target'
            invert: If True, the values in the returned array are inverted
        """
        return np.in1d(target, source, invert=invert)

    # --------------------------------------------------------------------------------
    # Operations - Statistics
    # --------------------------------------------------------------------------------
    @staticmethod
    def min(x, axis=None, keepdims=False):
        """Max value in x
        """
        return np.amin(
            x, axis=axis, keepdims=keepdims
        )

    @staticmethod
    def max(x, axis=None, keepdims=False):
        """Max value in x
        """
        return np.amax(
            x, axis=axis, keepdims=keepdims
        )

    @staticmethod
    def unique(x, out_idx=None):
        values = np.unique(x=x)
        return values

    # --------------------------------------------------------------------------------
    # Operations - indices
    # --------------------------------------------------------------------------------
    @staticmethod
    def argmin(x, axis=None):
        return np.argmin(x, axis=axis)

    @staticmethod
    def argmax(x, axis=None):
        return np.argmax(x, axis=axis)

    @staticmethod
    def argsort(x, axis: int = -1, direction: str = 'ASCENDING'):
        direction = direction.upper()
        if direction == 'ASCENDING':
            return np.argsort(x, axis=axis)
        elif direction == 'DESCENDING':
            return np.argsort(-x, axis=axis)
        else:
            raise AssertionError(f"Unexpected direction {direction}")

    # --------------------------------------------------------------------------------
    # Operations - Select
    # --------------------------------------------------------------------------------
    @staticmethod
    def mask(x, mask):
        """Select elements from array with boolean indices"""
        return x[mask]

    # --------------------------------------------------------------------------------
    # Operations - Update
    # --------------------------------------------------------------------------------
    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    # --------------------------------------------------------------------------------
    # Operations - Transformation
    # --------------------------------------------------------------------------------
    @staticmethod
    def concat(values, axis=0):
        return np.concatenate(values, axis=axis)

    # --------------------------------------------------------------------------------
    # Operations - Math
    # --------------------------------------------------------------------------------
    @staticmethod
    def add(x, y, out=None):
        assert out is None, "out is not supported for TF"
        return np.add(x, y, out=out)

    @staticmethod
    def sum(x, axis=None, keepdims=False):
        return np.sum(
            x, axis=axis, keepdims=keepdims
        )

    @staticmethod
    def multiply(x, y, out=None) -> TYPE_TENSOR:
        assert out is None, "out is not supported for TF"
        return np.multiply(x, y, out=out)

    @staticmethod
    def divide(x, y, out=None):
        return np.divide(x, y, out)

    @staticmethod
    def sqrt(X) -> TYPE_TENSOR:
        return np.sqrt(X)

    @staticmethod
    def pow(x, y):
        return np.pow(x, y)

    @staticmethod
    def einsum(equation, *inputs, out=None) -> TYPE_TENSOR:
        return np.einsum(equation, *inputs, out=out)

    @staticmethod
    def random_bool_tensor(shape: tuple, num_trues: int):
        """Generate a bool tensor where num_trues elements are set to True
        Args:
            shape: shape of the tensor to generate
            num_trues: number of True to randomly set to the tensor
        Returns: tensor of 'shape' where num_trues elements are set to True
        """
        size = np.multiply.reduce(a=shape, axis=None)  # multiply.reduce(([])) -> 1
        assert len(shape) > 0 <= num_trues <= size

        indices = np.random.choice(a=np.arange(size), size=num_trues, replace=False)
        flatten = np.zeros(size)
        flatten[indices] = 1

        return np.reshape(flatten, shape).astype(np.bool_)

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
