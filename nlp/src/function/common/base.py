import logging

import numpy as np
import tensorflow as tf

from common.constant import (
    TYPE_FLOAT,
    TYPE_TENSOR
)

Logger = logging.getLogger(__name__)


class Function:
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def is_np_float_scalar(X) -> bool:
        """Confirm if X is numpy float scalar
        1. np primitive e.g. np.float, np.float32
        2. np ndarray of shape () dtype is sub type of np.floating
        """
        return \
            np.issubdtype(type(X), np.floating) or \
            isinstance(X, np.ndarray) and X.ndim == 0 and np.issubdtype(X.dtype, np.floating)

    @staticmethod
    def is_np_float_tensor(X) -> bool:
        """Confirm if X is numpy float tensor
        """
        return \
            isinstance(X, np.ndarray) and X.ndim > 0 and np.issubdtype(X.dtype, np.floating)

    @staticmethod
    def is_float_scalar(X) -> bool:
        """Confirm if X is float scalar of shape ()
        """
        return \
            Function.is_np_float_scalar(X) or \
            (tf.is_tensor(X) and X.shape == () and X.dtype.is_floating)

    @staticmethod
    def is_tensor(X):
        """Check if X is rank > 1 tensor
        Scalar and vector are tensor, but "tensor" is re-defined as rank 2 or larger,
        so as to handle scalar, and vector respectively.
        """
        return \
            (isinstance(X, np.ndarray) and X.ndim > 0) or \
            (tf.is_tensor(X) and tf.rank(X) > 0)

    @staticmethod
    def is_float_tensor(X) -> bool:
        """Confirm if X is float tensor of dimension > 0
        The implementation depends on the framework e.g. numpy, tensorflow.
        """
        return \
            (isinstance(X, np.ndarray) and X.ndim > 0 and np.issubdtype(X.dtype, np.floating)) or \
            (tf.is_tensor(X) and X.dtype.is_floating)

    @staticmethod
    def is_finite(X) -> bool:
        return np.isfinite(X)

    @staticmethod
    def tensor_shape(X):
        """The shape of a tensor
        """
        assert Function.is_tensor(X)
        return tuple(X.get_shape()) if tf.is_tensor(X) else X.shape

    @staticmethod
    def tensor_rank(X):
        """The rank of a tensor
        Rank is the number of indices required to uniquely select an element of the tensor.
        Rank is also known as "order", "degree", or "ndims.

        The rank of a tensor is not the same as the rank of a matrix.
        Dimension may match with Rank but not always.
        """
        assert Function.is_tensor(X)
        return tf.rank(X) if tf.is_tensor(X) else X.ndim

    @staticmethod
    def tensor_size(X):
        """The size of a tensor
        """
        assert Function.is_tensor(X)
        return tf.size(X) if tf.is_tensor(X) else X.size

    @staticmethod
    def tensor_dtype(X):
        """The dtype of a tensor
        """
        assert Function.is_tensor(X)
        return X.dtype

    @staticmethod
    def to_tensor(X, dtype=None) -> TYPE_TENSOR:
        """Convert to numpy array
        Use numpy array as TF handles a numpy array as tensor
        """
        if dtype is None:
            return np.array(X)
        else:
            return np.array(X, dtype=dtype)

    @staticmethod
    def to_float_tensor(X, dtype=TYPE_FLOAT) -> TYPE_TENSOR:
        """
        Use numpy array as TF handles a numpy array as tensor
        """
        return np.array(X, dtype=dtype)

    @staticmethod
    def assure_tensor(X) -> np.ndarray:
        if Function.is_tensor(X):
            pass
        else:
            X = Function.to_tensor(X=X)

        return X

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
    def reshape(X, shape):
        return np.reshape(X, shape)

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
