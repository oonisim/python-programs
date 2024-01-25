import logging
from typing import (
    Tuple
)

import numpy as np
import tensorflow as tf

import function.nn.base as base
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR,
    BOUNDARY_SIGMOID,
    TYPE_NN_FLOAT,
)

Logger = logging.getLogger(__name__)


class Function(base.Function):
    # ================================================================================
    # Class
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Types
    # --------------------------------------------------------------------------------
    @staticmethod
    def TYPE_FLOAT(x, dtype=TYPE_NN_FLOAT):
        return tf.constant(x, dtype=dtype)

    # --------------------------------------------------------------------------------
    # Tensors
    # --------------------------------------------------------------------------------
    @staticmethod
    def ones(shape=None, dtype=TYPE_NN_FLOAT):
        return tf.ones(
            shape=shape, dtype=dtype, name=None
        )

    @staticmethod
    def zeros(shape=None, dtype=TYPE_NN_FLOAT):
        return tf.zeros(
            shape=shape, dtype=dtype, name=None
        )

    @staticmethod
    def full(shape, value, dtype=TYPE_NN_FLOAT):
        return tf.fill(dims=shape, value=tf.constant(value, dtype=dtype))

    # --------------------------------------------------------------------------------
    # Tensor validations
    # --------------------------------------------------------------------------------
    @staticmethod
    def is_finite(X) -> bool:
        return tf.reduce_all(tf.math.is_finite(X))

    @staticmethod
    def all(x, axis=None) -> bool:
        """Check if all true"""
        return tf.reduce_all(x, axis=axis)

    @staticmethod
    def all_close(expected, actual, msg=None):
        try:
            tf.debugging.assert_near(
                expected, actual,
                rtol=tf.constant(0.1, dtype=TYPE_FLOAT),
                atol=tf.constant(1e-2, dtype=TYPE_FLOAT),
                message=msg,
                summarize=None
            )
            return True
        except tf.errors.InvalidArgumentError as e:
            return False

    @staticmethod
    def all_equal(x, y):
        return tf.reduce_all(tf.math.equal(x, y))

    @staticmethod
    def in1d(target, source, invert: bool = False):
        """
        Return boolean tensor of the target shape that tells if an element of
        'target' b is in 'source'.

        Example:
            target = tf.constant([1,2,3,4,5])
            source = tf.constant([1,3,5])
            is_in(target, source)
            -----
            [ True, False,  True, False,  True]

        Args:
            target: Target 1D array to test
            source: list of elements to check if in 'target'
            invert: If True, the values in the returned array are inverted
       """
        assert Function.tensor_rank(target) == Function.tensor_rank(source) == 1
        mask = tf.reduce_any(tf.equal(tf.reshape(source, shape=(-1, 1)), target), axis=0)
        return tf.math.logical_not(mask) if invert else mask

    # --------------------------------------------------------------------------------
    # Operations - Statistics
    # --------------------------------------------------------------------------------
    @staticmethod
    def min(x, axis=None, keepdims=False):
        """Max value in x
        """
        return tf.math.reduce_min(
            x, axis=axis, keepdims=keepdims
        )

    @staticmethod
    def max(x, axis=None, keepdims=False):
        """Max value in x
        """
        return tf.math.reduce_max(
            x, axis=axis, keepdims=keepdims
        )

    @staticmethod
    def unique(x, out_idx=tf.as_dtype(TYPE_INT)):
        values, indices = tf.unique(x=x)
        return values

    # --------------------------------------------------------------------------------
    # Operations - indices
    # --------------------------------------------------------------------------------
    @staticmethod
    def argmin(x, axis: int = 0):
        return tf.math.argmin(x, axis=axis, output_type=tf.as_dtype(TYPE_INT))

    @staticmethod
    def argmax(x, axis: int = 0):
        return tf.math.argmax(x, axis=axis, output_type=tf.as_dtype(TYPE_INT))

    @staticmethod
    def argsort(x, axis: int = -1, direction: str = 'ASCENDING'):
        return tf.argsort(x, axis=axis, direction=direction.upper())

    # --------------------------------------------------------------------------------
    # Operations - Select
    # --------------------------------------------------------------------------------
    @staticmethod
    def mask(x, mask):
        """Select elements from array with boolean indices
        array[mask] in numpy where the mask is boolean nd.array
        """
        return tf.boolean_mask(tensor=x, mask=mask)

    # --------------------------------------------------------------------------------
    # Operations - Update
    # --------------------------------------------------------------------------------
    @staticmethod
    def where(condition, x, y):
        return tf.where(condition, x, y)

    # --------------------------------------------------------------------------------
    # Operations - Transformation
    # --------------------------------------------------------------------------------
    @staticmethod
    def concat(values, axis=0):
        return tf.concat(values, axis=axis)

    # --------------------------------------------------------------------------------
    # Operations - Math
    # --------------------------------------------------------------------------------
    @staticmethod
    def add(x, y, out=None):
        assert out is None, "out is not supported for TF"
        return tf.math.add(x, y)

    @staticmethod
    def sum(x, axis=None, keepdims=False):
        return tf.math.reduce_sum(
            x, axis=axis, keepdims=keepdims
        )

    @staticmethod
    def multiply(x, y, out=None) -> TYPE_TENSOR:
        """
        https://stackoverflow.com/a/54255819/4281353
        TF Matmul expects float32, and causes errors otherwise.
        'InvalidArgumentError: cannot compute Mul as input #1(zero-based)
        was expected to be a float tensor but is a double tensor [Op:Mul]'
        """
        assert out is None, "out is not supported for TF"
        if TYPE_FLOAT == np.float64:
            return np.multiply(x, y)
        else:
            return tf.math.multiply(x, y)

    @staticmethod
    def divide(x, y):
        return tf.math.divide(x, y)

    @staticmethod
    def sqrt(X) -> TYPE_TENSOR:
        return tf.math.sqrt(X)

    @staticmethod
    def pow(x, y):
        return tf.math.pow(x, y)

    @staticmethod
    def einsum(equation, *inputs, **kwargs) -> TYPE_TENSOR:
        assert "out" not in kwargs, "out is not supported for TF"
        return tf.einsum(equation, *inputs, **kwargs)

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
        return Y

    @staticmethod
    def softmax(X: TYPE_TENSOR, axis=None, out=None) -> TYPE_TENSOR:
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
    def sigmoid_cross_entropy_log_loss(
            X: TYPE_TENSOR,
            T: TYPE_TENSOR
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cross entropy log loss for sigmoid activation -( T*log(Z) + (1-T)*log(1-Z) )
        where Z = sigmoid(X).

        Formula:
            Solution to avoid rounding errors and subtract cancellation by Reza Bonyadi.
            -----
            Let z=1/(1+p), p= e^(-x), then log(1-z)=log(p)-log(1+p), which is more stable
            in terms of rounding errors (we got rid of division, which is the main issue
            in numerical instabilities).
            -----

            J = (1-T)X + np.log(1 + np.exp(-X))

        Args:
            X: Input data of shape (N,1) to go through sigmoid where:
                N is Batch size
                Number of nodes M is always 1 for binary 0/1 classification.
            T: label in the index format of shape (N,1).

        Returns:
            J: Loss value of shape () for scalar or (N,) a loss value per batch.
            P: Activation value sigmoid(X)
        """
        name = "sigmoid_cross_entropy_log_loss"
        shape_X = super(Function, Function).tensor_shape(X)
        shape_T = super(Function, Function).tensor_shape(T)
        assert super(Function, Function).tensor_rank(X) == 2 and shape_X[1] == 1
        assert super(Function, Function).tensor_rank(T) == 2 and shape_T[1] == 1
        assert shape_X[0] == shape_T[0]

        # --------------------------------------------------------------------------------
        # DO NOT squeeze X or P as the caller expects the original shape from X.
        # e.g. CrossEntropyLogLoss.function() where the shape of P:(N,M) is
        # checked if it has the same shape of X:(N,M)
        #
        # To handle both OHE label format and index label format, the data
        # X and T for OHE/logistic classification both have the shape (N,1).
        # as the number of input M into the logistic log loss layer = 1.
        # Because the expected loss J shape is (N,) to match with the output
        # number of output from the loss layer (N,), one loss value for each
        # batch, J is squeezed, but not P.
        # --------------------------------------------------------------------------------
        # X = tf.squeeze(X, axis=-1)
        # T = tf.squeeze(T, axis=-1)
        J = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(T, dtype=X.dtype),     # Label dtype must be the same with X
            logits=X
        )
        J = tf.squeeze(J, axis=-1)  # Shape from (N,M) to (N,), one loss per batch
        P = tf.nn.sigmoid(x=X)
        assert tf.reduce_all(tf.math.is_finite(J))
        assert tf.reduce_all(tf.math.is_finite(P))

        return J, P

    @staticmethod
    def softmax_cross_entropy_log_loss(
            X: TYPE_TENSOR,
            T: TYPE_TENSOR,
            need_softmax: bool = True,
            out_P=None,
            out_J=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cross entropy log loss for softmax activation.
        Returns:
            J: Loss value of shape (N,), a loss value per batch.
            P: Activation value softmax(X)
        """
        assert tf.reduce_all(tf.math.is_finite(X)), f"NaN or inf detected \n{X}\n"
        assert out_J is None and out_P is None, "out is not supported"

        name = "softmax_cross_entropy_log_loss"
        N = super(Function, Function).tensor_shape(X)[0]
        rows = tf.range(N)  # (N,)
        cols = T            # Same shape (N,) with rows
        assert super(Function, Function).tensor_shape(rows) == super(Function, Function).tensor_shape(cols), \
            "np P indices need the same shape"

        J = tf.nn.softmax_cross_entropy_with_logits(
            labels=T, logits=X, axis=-1
        )
        shape_J = super(Function, Function).tensor_shape(J)
        P = tf.nn.softmax(X=X) if need_softmax else tf.constant(np.empty(X.shape), dtype=TYPE_NN_FLOAT)

        if not super(Function, Function).is_finite(J):
            raise RuntimeError(f"{name}: Invalid loss J:\n{J}.")

        assert super(Function, Function).tensor_rank(J) == 1 and (0 < N == shape_J[0]), \
            f"Need J shape ({N},) but {shape_J}."

        Logger.debug("%s: J is [%s] J.shape %s\n", name, J, shape_J)

        return J, P

    @staticmethod
    def random_choice(a, size):
        """Random choice from 'a' based on 'size' without duplicates
        Tensorflow 2.0 does not have equivalent.
        See https://github.com/tensorflow/tensorflow/issues/8496

        Args:
            a: Tensor
            size: int or shape as tuple of ints e.g., (m, n, k).
        Returns: Tensor of the shape specified with 'size' arg.

        Examples:
            X = tf.constant([[1,2,3],[4,5,6]])
            random_choice(X, (2,1,2))
            -----
            [
              [
                [5 4]
              ],
              [
                [1 2]
              ]
            ]
        """
        is_size_scalar: bool = \
            isinstance(size, int) or np.issubdtype(type(a), np.integer) or\
            (tf.is_tensor(a) and a.shape == () and a.dtype.is_integer)
        if is_size_scalar:
            shape = (size,)
        elif isinstance(size, tuple) and len(size) > 0:
            shape = size
        else:
            raise AssertionError(f"Unexpected size arg {size}")

        sample_size = tf.math.reduce_prod(size, axis=None)
        assert sample_size > 0

        # --------------------------------------------------------------------------------
        # Select elements from a flat array
        # --------------------------------------------------------------------------------
        a = tf.reshape(a, (-1))
        length = tf.size(a)
        assert sample_size <= length

        # --------------------------------------------------------------------------------
        # Shuffle a sequential numbers (0, ..., length-1) and take size.
        # To select 'sample_size' elements from a 1D array of shape (length,),
        # TF Indices needs to have the shape (sample_size,1) where each index
        # has shape (1,),
        # --------------------------------------------------------------------------------
        indices = tf.reshape(
            tensor=tf.random.shuffle(tf.range(0, length, dtype=tf.int32))[:sample_size],
            shape=(-1, 1)   # Convert to the shape:(sample_size,1)
        )
        return tf.reshape(tensor=tf.gather_nd(a, indices), shape=shape)

    @staticmethod
    def random_bool_tensor(shape: tuple, num: int):
        """Generate bool tensor where num elements are set to True
        Args:
            shape: shape of the tensor to generate
            num: number of 'True' elements in the result tensor
        Returns: tensor of shape where num elements are set to True
        """
        size = tf.math.reduce_prod(shape, axis=None)
        num = tf.cast(num, tf.int32)
        # Must check len(shape) as reduce_prod(([])) -> 1
        # https://stackoverflow.com/questions/67351236
        assert len(shape) > 0 <= num <= size

        # --------------------------------------------------------------------------------
        # TF Indices to update a 1D array of shape (size,).
        # Indices has the shape (size,1) where each index has shape (1,)
        # --------------------------------------------------------------------------------
        indices = tf.reshape(
            tensor=tf.random.shuffle(
                tf.range(0, size, dtype=tf.int32)
            )[:num],  # Shuffle a sequential indices and take 'num' indices
            shape=(-1, 1)  # Convert to the shape:(size,1)
        )
        updates = tf.ones(shape=(num,), dtype=tf.int32)
        X = tf.tensor_scatter_nd_update(
            tensor=tf.zeros(shape=(size,), dtype=tf.int32),
            indices=indices,
            updates=updates
        )

        return tf.cast(tf.reshape(X, shape), dtype=tf.bool)

    @staticmethod
    def numerical_jacobian(
            f,
            X,
            condition=None
    ) -> np.ndarray:
        """Calculate gradients dX via Tensorflow Autodiff

        Args:
            f: Y=f(X) where Y is a scalar or shape() array.
            X: input of shame (N, M), or (N,) or ()
            condition: boolean indices to select which elements to calculate
        Returns:
            dX: Jacobian matrix that has the same shape of X.
        """
        name = "numerical_jacobian"
        X = tf.Variable(X, trainable=True)
        assert X.dtype.is_floating

        with tf.GradientTape() as tape:
            tape.watch(X)  # Start recording the history of operations applied to `X`

            # --------------------------------------------------------------------------------
            # Forward path
            # --------------------------------------------------------------------------------
            loss = f(X)

            # --------------------------------------------------------------------------------
            # Backward path/Autodiff
            # --------------------------------------------------------------------------------
            dX = tape.gradient(loss, X)

        assert \
            dX is not None and \
            super(Function, Function).tensor_shape(dX) == super(Function, Function).tensor_shape(X)
        if not tf.reduce_all(tf.math.is_finite(dX)):
            raise ValueError(f"{name} caused Nan or Inf \n%s\n" % dX.numpy())

        return dX

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
