"""DNN functions
Responsibility:
    Functions to construct the neural network.

NOT:
    This is not the place to define non-neural network related utilities.
    For common utilities, use utility.py.

NOTE:
    Those marked as "From deep-learning-from-scratch" is copied from the github.
    https://github.com/oreilly-japan/deep-learning-from-scratch
"""
import logging
from typing import (
    Optional,
    Union,
    Tuple,
    Callable
)

import numexpr as ne
import numpy as np

from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL,
    OFFSET_DELTA,
    OFFSET_LOG,
    OFFSET_MODE_ELEMENT_WISE,
    BOUNDARY_SIGMOID,
    GRADIENT_SATURATION_THRESHOLD,
    ENABLE_NUMEXPR
)
from test.config import (
    ENFORCE_STRICT_ASSERT,
)

Logger = logging.getLogger("functions")


def identity(x: np.ndarray):
    return x


def standardize(
    X: Union[np.ndarray, TYPE_FLOAT],
        eps: TYPE_FLOAT = 0.0,
        keepdims=False,
        out=None,
        out_mean=None,
        out_md=None,
        out_sd=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize X per-feature basis.
    Each feature is independent from other features, hence standardize per feature.
    1. Calculate the mean per each column, not entire matrix.
    2. Calculate the variance per each column
    3. Standardize mean/sqrt(variance+eps) where small positive eps prevents sd from being zero.

    TODO:
        separate logic for numexpr and numpy
        implement numba option

    Args:
        X: Input data to standardize per feature basis.
        eps: A small positive value to assure sd > 0 for deviation/sd will not be div-by-zero.
             Allow eps=0 to simulate or compare np.std().
        keepdims: bool, optional to control numpy keepdims option.
        out: Output storage for the standardized X
        out_mean: Output storage for the mean
        out_md: Output storage for the MD (Mean Deviation) = X-mean
        out_sd: Output storage for the SD

    Returns:
        standardized: standardized X
        mean: mean of X
        sd: standard deviation of X
        deviation: X-mean
    """
    assert (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT and X.size > 0)
    if X.ndim <= 1:
        X = X.reshape(1, -1)

    N = X.shape[0]
    ddof = 1 if N > 1 else 0    # Bessel's correction

    # --------------------------------------------------------------------------------
    # Calculate mean/deviation/sd per feature
    # --------------------------------------------------------------------------------
    mean = np.mean(X, axis=0, keepdims=keepdims, out=out_mean)
    deviation = ne.evaluate("X - mean", out_md) \
        if ENABLE_NUMEXPR else np.subtract(X, mean, out=out_md)

    # --------------------------------------------------------------------------------
    # Using MD instead of SD
    # --------------------------------------------------------------------------------
    # TODO remove from here
    # md = np.sum(np.abs(deviation), axis=0) / N
    # mask = (md < 1e-8)
    # if np.any(mask):
    #     md[mask] = TYPE_FLOAT(1.0)
    #     standardized = np.divide(deviation, md, out)
    # else:
    #     standardized = np.divide(deviation, md, out)
    #
    # return standardized, mean, md, deviation
    # # To here
    # --------------------------------------------------------------------------------

    if eps > 0:
        # Re-use the storage of buffer for standardized.
        buffer = ne.evaluate("deviation ** 2")
        sd = np.sum(buffer, axis=0, keepdims=keepdims)
        sd = ne.evaluate("sqrt( (sd / (N - ddof)) + eps )", out=sd)
        standardized = ne.evaluate("deviation / sd", out=buffer)
    else:
        sd = np.std(X, axis=0, ddof=ddof, keepdims=keepdims, out=out_sd)

        # --------------------------------------------------------------------------------
        # TODO: See if below works. If it does not, implement the original BN which uses
        # variance = sqrt(variance + eps) / (N -ddos)
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        # NOTE:
        # Even when a feature has the same values e.g. -3.29686744, its mean may not be 0.
        # https://stackoverflow.com/questions/66728134
        #
        # X = np.array([
        #     [-1.11793447, -3.29686744, -3.50615096],
        #     [-1.11793447, -3.29686744, -3.50615096],
        #     [-1.11793447, -3.29686744, -3.50615096],
        #     [-1.11793447, -3.29686744, -3.50615096],
        #     [-1.11793447, -3.29686744, -3.50615096]
        # ])
        #
        # X-mean is
        # [[0.0000000e+00 4.4408921e-16 4.4408921e-16]
        #  [0.0000000e+00 4.4408921e-16 4.4408921e-16]
        #  [0.0000000e+00 4.4408921e-16 4.4408921e-16]
        #  [0.0000000e+00 4.4408921e-16 4.4408921e-16]
        #  [0.0000000e+00 4.4408921e-16 4.4408921e-16]]
        #
        # SD is
        # [0.0000000e+00 4.4408921e-16 4.4408921e-16]
        #
        # Hence regard the SD as 0 when the value is less than a small value k. e.g. 1e-8.
        # If the distribution is such skewed around the mean, regard it as (X-mean) == 0.
        # --------------------------------------------------------------------------------
        mask = (sd < 1e-8)
        if np.any(mask):
            # Temporary replace the zero elements with one
            sd[mask] = TYPE_FLOAT(1.0)

            # standardize and zero clear the mask elements
            standardized = np.divide(deviation, sd, out)

            # --------------------------------------------------------------------------------
            # sd == 0 means variance == 0, which happens when (x-mu) == 0
            # v=sum(square(x-mu)) / (n-1).
            # Then elements of (X-mean) where sd ==0 should be 0.
            # Hence there should be no need to zero-clear those elements.
            # --------------------------------------------------------------------------------
            # out[::, mask] = 0.0

            # --------------------------------------------------------------------------------
            # Leaving "sd[mask] = 1.0" should be OK because (X-mean)/sd -> 0
            # for those element where sd == 0. Then the BN output
            # "gamma * ((X-mean) / sd) + beta" -> beta
            # --------------------------------------------------------------------------------
            # restore sd
            # sd[mask] = 0.0

        else:
            standardized = np.divide(deviation, sd, out)

    assert np.all(sd != TYPE_FLOAT(0.0))
    assert np.all(np.isfinite(standardized))
    return standardized, mean, sd, deviation


def logarithm(
        X: Union[np.ndarray, TYPE_FLOAT],
        offset: Optional[Union[np.ndarray, TYPE_FLOAT]] = OFFSET_LOG,
        out=None
) -> Union[np.ndarray, TYPE_FLOAT]:
    """Wrapper for np.log(x) to set the hard-limit for x
    Args:
        X: domain value for log
        offset: The lower boundary of acceptable X value.
        out: A location into which the result is stored
    Returns:
        np.log(X)
    """
    if isinstance(X, TYPE_FLOAT):
        return np.log(X + offset, out=out)

    assert (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT)
    offset = OFFSET_LOG if (offset is None or offset <= 0.0) else offset   # offset > 0
    assert offset > 0.0
    if OFFSET_MODE_ELEMENT_WISE:
        # --------------------------------------------------------------------------------
        # Clip the element value only when it is below the offset as log(k), not log(x+k).
        # Note: Checking all element can take time and deepcopy everytime cost memory.
        # --------------------------------------------------------------------------------
        selections = (X < offset)
        if np.any(selections):
            if out is None:
                _X = np.copy(X)
            else:
                assert out.shape == X.shape
                np.copyto(out, X)
                _X = out

            _X[selections] = offset
        else:
            _X = X

        Y = ne.evaluate("log(_X)", out=out) \
            if ENABLE_NUMEXPR else np.log(_X, out=out)
    else:
        # --------------------------------------------------------------------------------
        # Adding the offset value to all elements as log(x+k) to avoid log(0)=-inf.
        # --------------------------------------------------------------------------------
        Y = ne.evaluate("log(X + offset)", out=out) \
            if ENABLE_NUMEXPR else np.log(X+offset, out=out)
    assert \
        np.all(np.isfinite(Y)), f"log(X) caused nan for X \nX={X}."

    return Y


def sigmoid_reverse(y):
    """
    Args:
        y: y=sigmoid(x)
    Returns:
        x: x that gives y=sigmoid(x)
    """
    return np.log(y/(1-y))


def sigmoid(
    X: Union[TYPE_FLOAT, np.ndarray],
    boundary: Optional[Union[np.ndarray, TYPE_FLOAT]] = BOUNDARY_SIGMOID,
    out=None
) -> Union[TYPE_FLOAT, np.ndarray]:
    """Sigmoid activate function
    Args:
        X: > domain value for log
        boundary: The lower boundary of acceptable X value.
        out: A location into which the result is stored

    NOTE:
        epsilon to prevent causing inf e.g. log(X+e) has a consequence of clipping
        the derivative which can make numerical gradient unstable. For instance,
        due to epsilon, log(X+e+h) and log(X+e-h) will get close or same, and
        divide by 2*h can cause catastrophic cancellation.

        To prevent such instability, limit the value range of X with boundary.
    """
    assert (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT) or isinstance(X, TYPE_FLOAT)
    boundary = BOUNDARY_SIGMOID if (boundary is None or boundary <= TYPE_FLOAT(0)) else boundary
    assert boundary > 0

    if np.all(np.abs(X) <= boundary):
        _X = X
    elif isinstance(X, np.ndarray):
        Logger.warning(
            "sigmoid: X value exceeded the boundary %s, hence clipping.", boundary
        )
        _X = np.copy(X)
        _X[X > boundary] = boundary
        _X[X < -boundary] = -boundary
    else:   # Scalar
        assert isinstance(X, TYPE_FLOAT)
        Logger.warning(
            "sigmoid: X value exceeded the boundary %s, hence clipping.", boundary
        )
        _X = np.sign(X) * boundary

    if ENABLE_NUMEXPR:
        Y = ne.evaluate("1 / (1 + exp(-1 * _X))", out=out)
    else:
        Y = 1 / (1 + np.exp(-1 * _X))

    return Y


def sigmoid_gradient(X: Union[TYPE_FLOAT, np.ndarray]) -> Union[TYPE_FLOAT, np.ndarray]:
    """Sigmoid gradient
    For X: [10,15, 20, 25, 30], sigmoid(X) gets saturated as:
    0.999954602131298
    0.999999694097773
    0.999999997938846
    0.999999999986112
    0.999999999999907
    For abs(X) > 30, derivative gets 0.

    Args:
        X:
    Returns: gradient
    """
    assert (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT) or isinstance(X, TYPE_FLOAT)
    Z = sigmoid(X)
    return Z * (TYPE_FLOAT(1.0) - Z)


def relu(X: Union[TYPE_FLOAT, np.ndarray]) -> Union[TYPE_FLOAT, np.ndarray]:
    """ReLU activation function"""
    assert (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT) or isinstance(X, TYPE_FLOAT)
    return np.maximum(TYPE_FLOAT(0.0), X)


def relu_gradient(X: Union[TYPE_FLOAT, np.ndarray]) -> Union[TYPE_FLOAT, np.ndarray]:
    """ReLU gradient
    Args:
        X:
    Returns: gradient
    """
    assert (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT) or isinstance(X, TYPE_FLOAT)
    grad = np.zeros_like(X)
    grad[X >= TYPE_FLOAT(0.0)] = TYPE_FLOAT(1)
    return grad


def softmax(X: np.ndarray, out=None) -> np.ndarray:
    """Softmax P = exp(X) / sum(exp(X))
    Args:
        X: batch input data of shape (N,M).
            N: Batch size
            M: Number of nodes
        out: A location into which the result is stored
    Returns:
        P: Probability of shape (N,M)
    """
    name = "softmax"
    assert isinstance(X, TYPE_FLOAT) or (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT), \
        "X must be float or ndarray(dtype=TYPE_FLOAT)"

    # --------------------------------------------------------------------------------
    # exp(x-c) to prevent the infinite exp(x) for a large value x, with c = max(x).
    # keepdims=True to be able to broadcast.
    # --------------------------------------------------------------------------------
    C = np.max(X, axis=-1, keepdims=True)
    exp = np.exp(X - C)
    P = np.divide(exp, np.sum(exp, axis=-1, keepdims=True), out=out)
    Logger.debug("%s: X %s exp %s P %s", name, X, exp, P)

    return P


def categorical_log_loss(
        P: np.ndarray, T: np.ndarray, offset: Optional[TYPE_FLOAT] = None
) -> np.ndarray:
    """Categorical cross entropy log loss function for multi class classification.
    Args:
        P: Probabilities e.g. from Softmax
           arg name must be P to be consistent with other log function.
        T: Labels
        offset: small number to avoid np.inf by log(0) by log(0+offset)
    Returns:
        J: Loss value.
    """
    assert np.all(np.isin(T, [0, 1]))
    J: np.ndarray = -T * logarithm(P, offset)
    return J


def logistic_log_loss(
        P: np.ndarray, T: np.ndarray, offset: Optional[TYPE_FLOAT] = None
) -> Union[np.ndarray, TYPE_FLOAT]:
    """Logistic cross entropy log loss function for binary classification.
    Args:
        P: Activations e.g. from Sigmoid
           arg name must be P to be consistent with other log function.
        T: Labels
        offset: small number to avoid np.inf from log(0) by log(0+offset)
    Returns:
        J: Loss value.
    """
    # assert (np.all((P+offset) > 0) and np.all(T == 1)) or\
    #        (np.all((1-P+offset) > 0) and np.all(T == 0))
    # J: np.ndarray = -(T * np.log(P+offset) + (1-T) * np.log(1-P+offset))

    assert np.all(np.isin(T, [0, 1]))
    J = -(T * logarithm(P, offset) + (1.0-T) * logarithm(1.0-P, offset))

    return J


def logistic_log_loss_gradient(X, T, offset: TYPE_FLOAT = BOUNDARY_SIGMOID):
    """Derivative of
    Z = sigmoid(X), dZ/dX = Z(1-Z)
    L = -( T*log(Z) + (1-T) * log(1-Z) ) dL/dZ = -T(1-T)/Z + (1-T)/(1-Z)
   """
    assert np.all(np.isin(T, [0, 1]))
    Z = sigmoid(X)

    # dL/dX = (Z - T)
    return -T * (1-Z) + Z * (1-T)


def transform_X_T(
        X: Union[np.ndarray, TYPE_FLOAT], T: Union[np.ndarray, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate acceptable (X, T) conditions. Identify if T is in OHE or index
    format and transform them accordingly into X:(N,M), T:(N) shapes for T in
    index format or X:(N,M), T(N, M) shapes for T in OHE format.

    NOTE:
        This handling make things way too complex that it should be.
        By being specific with below, this should be avoided.
        1. Index label or OHE label
        2. Binary classification or multi class classifications.

    Background:
        For T,
            - both the OHE (One Hot Encoding) and the index format are accepted.
            - both binary label for logistic regression, and multi categorical
              labels for the multi label classification are accepted.

        This function identifies:
        1. If the format is OHE or Index
        2. If the label is binary or multi labels.

        And transform X, T into:
        1. X:(N,M), T:(N) shapes for T in index format.
        2. X:(N,M), T(N, M) shapes for T in OHE format.

        Then X shape is always in 2D, hence always be able to use X[0], X[1],
        and T shape tells its format.

    Note:
        M == 1 for X: Always binary OHE label (Binary is OHE).
            X=0.9, T=1: Scalar OHE.
            X[0.9], T[0]: Same with scalar OHE X=0.9, T=1.
            X[[0.9], [0.7], [0.2]], T [[1], [1], [0]]: 2D OHE
            X[[0.9], [0.7], [0.2]], T [1, 1, 0]: transform into 2D OHE

        M > 1 for X and X.ndim == T.ndim: OHE labels.
            X[0,9, 0.1], T[1, 0]: Into X[[0,9, 0.1]], T[1, 0]
            X[                     T[
                [0,9, 0.1, 0.3]        [1, 0, 0],
                [0,2, 0.1, 0.7]        [0, 0, 1]
            ]                      ]
            transform into
            X[                     T[
                [0,9, 0.1, 0.3]        0,
                [0,2, 0.1, 0.7]        2
            ]                      ]

        M > 1 for X and X.ndim == T.ndim:
            Transform into index label with T = T.argmax()

    Args:
        X: Input
        T: Labels
    Returns:
        (X, T): reshaped X of shape(N,M), T of shape(M,)
    """
    name = "transform_X_T"
    assert (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT) or isinstance(X, TYPE_FLOAT), \
        f"Type of P must be {TYPE_FLOAT}"
    assert (isinstance(T, np.ndarray) and np.issubdtype(T.dtype, np.integer)) or \
           isinstance(T, int), "Type of T must be integer"

    if isinstance(X, TYPE_FLOAT) or X.ndim == 0:
        # --------------------------------------------------------------------------------
        # When X is scalar, T must be a scalar. This is a binary (0/1) label.
        # --------------------------------------------------------------------------------
        assert isinstance(T, int) or T.ndim == 0
        X = np.array(X).reshape((-1, 1))
        T = np.array(T).reshape((-1, 1))  # Convert into T(N,1) 2D OHE Binary

    else:
        # ================================================================================
        # Hereafter, X and T need be np arrays.
        # Convert int T into ndarray.
        # Convert T in OHE format into index format with T = np.argmax(T). If X is in 1D,
        # convert it into 2D so as to get the log loss with numpy tuple- like indices
        # X[
        #   range(N),
        #   T
        # ]
        # ================================================================================
        assert isinstance(X, np.ndarray)
        T = np.array(T, dtype=TYPE_LABEL) if isinstance(T, int) else T

        if X.ndim == 1:
            # --------------------------------------------------------------------------------
            # X is 1D array, then T dim should be in Set{0, 1}.
            # --------------------------------------------------------------------------------
            _shape = T.shape
            if T.ndim == 0:
                if T.size == X.size == 1:  # M==1
                    # --------------------------------------------------------------------------------
                    # Binary OHE label e.g. T=0, P=[0.94]. Reshape into 2D OHE format
                    # (T.ndim == P.ndim == 2) after transformation.
                    # --------------------------------------------------------------------------------
                    assert np.all(np.isin(T, [0, 1])), "Binary label must be 0/1"
                    X = np.array(X).reshape((-1, 1))
                    T = T.reshape((-1, 1))  # Convert into T(N,1) 2D OHE Binary
                else:  # M>1
                    # --------------------------------------------------------------------------------
                    # T=i is a scalar index label for a 1D X[x0, x1, ..., xd-1] to select Xi.
                    # Convert the scalar index label into single element 1D index label T.
                    # Transfer results in (P.ndim==T.ndim+1 > 1) and (P.shape[0]==T.shape[0]==N).
                    # --------------------------------------------------------------------------------
                    T = T.reshape(-1)
                    X = X.reshape((1, -1))

            elif T.ndim == 1:  # T is OHE label
                if T.size == X.size == 1:  # M==1
                    # --------------------------------------------------------------------------------
                    # Binary OHE label e.g. T=[0], P=[0.94]. Reshape into 2D OHE format
                    # (T.ndim == P.ndim == 2) after transformation.
                    # --------------------------------------------------------------------------------
                    assert np.all(np.isin(T, [0, 1])), "Binary label must be 0/1"
                    X = np.array(X).reshape((-1, 1))
                    T = T.reshape((-1, 1))  # Convert into T(N,1) 2D OHE Binary

                else:
                    # --------------------------------------------------------------------------------
                    # Index label when M>1
                    # T[0,...1,...0] is OHE where T[i]==1 to select X[i] from X[x0, ..., Xi, ..., xd-1].
                    # Then T.shape == X.shape must be true when T is OHE labels.
                    # Convert the OHE labels into a 1D index labels T with np.argmax(T).reshape(-1)
                    # because np.argmax(ndim=1) results in ().
                    # Transfer results in (P.ndim==T.ndim+1 > 1) and (P.shape[0]==T.shape[0]==N).
                    # --------------------------------------------------------------------------------
                    # This format could be binary label as well as multi labels.
                    # [Multi label]:
                    # There are 1 input X which can be one of 4 labels. Probability of each
                    # label is P[0.1, 0.0, 0.98,0.2]. T[0,0,1,0] is 1 truth that tells the 3rd label is
                    # correct. Same with T=2 as index label.
                    #
                    # [Binary 0/1 label]:
                    # There batch size of X is 4 with 4 inputs. T[0,0,1,0] has 4 truth for 4 inputs.
                    # This is the same with P(N,M), T(N,M).
                    # P[              T[
                    #   [0.1],           [0/False],
                    #   [0.0],           [0/False],
                    #   [0.98],         [1/True],
                    #   [0.2],           [0/False]
                    # ]
                    #
                    # We cannot tell if it is binary label or multi labels. The decision is this is multi.
                    # Use X(N,M), T(N,M) format for BINARY 0/1 labeling.
                    # --------------------------------------------------------------------------------
                    assert T.shape == X.shape and np.all(np.isin(T, [0, 1])), \
                        "For (T.ndim == X.ndim == 1), T is OHE and T.shape %s == X.shape %s needs True" \
                        % (T.shape, X.shape)
                    T = np.argmax(T).reshape(-1)
                    X = X.reshape((1, -1))

            else:
                assert False, \
                    "When X.ndim=1, T.ndim should be 0 or 1 but %s" % T.ndim

            Logger.debug("%s: X.shape (%s,) has been converted into %s", name, X.size, X.shape)
            Logger.debug("%s: T.shape %s has been converted into %s", name, _shape, T.shape)

        else:  # X.ndim > 1
            _shape = T.shape
            if T.ndim == 1:
                if X.shape[1] == 1:
                    # --------------------------------------------------------------------------------
                    # M (X.shape[1]==1) then T is binary OHE labels. Convert T into T(N, 1) shape.
                    # Transfer results in (P.ndim==T.ndim+1 > 1) and (P.shape[1]==T.shape[1]==1).
                    # This condition X:(N,1), T(N,1) are in the 2D binary OHE labels.
                    # --------------------------------------------------------------------------------
                    assert np.all(np.isin(T, [0, 1])), "Binary label must be 0/1"
                    T = T.reshape((-1, 1))  # Convert into T(N,1) 2D OHE Binary
                    Logger.debug("%s: T.shape %s has been converted into %s", name, _shape, T.shape)

                elif X.shape[1] > 1:
                    # --------------------------------------------------------------------------------
                    # M (X.shape[1]>1) then T is index labels. (X, T) are in X(N,M), T(N,)
                    # No conversion required.
                    # --------------------------------------------------------------------------------
                    assert X.shape[0] == T.shape[0], \
                        "%s: Index format X(N,M), T(N,) expected but X %s T %s" \
                        % (name, X.shape, T.shape)
                    pass

            elif T.ndim > 1:
                if T.shape[1] == X.shape[1] == 1:
                    # --------------------------------------------------------------------------------
                    # T is binary OHE labels e.g. T[[0]], P[[0.9]] or T[[0],[1],[0]], P[[0.],[0.9],[0.]]
                    # No conversion
                    # --------------------------------------------------------------------------------
                    pass
                else:
                    # --------------------------------------------------------------------------------
                    # T is OHE labels. Convert into index labels
                    # Transfer results in (P.ndim==T.ndim+1 > 1) and (P.shape[0]==T.shape[0]==N).
                    # --------------------------------------------------------------------------------
                    T = T.argmax(axis=-1)
                    Logger.debug("%s: T.shape %s has been converted into %s", name, _shape, T.shape)
            else:
                msg = "transform_X_T(): Invalid state."
                Logger.error(
                    "transform_X_T(): Invalid state X.shape %s X \n%s \nT.shape %s T %s",
                    X.shape, X, T.shape, T
                )
                raise RuntimeError(msg)

    return X, T


def transform_scalar_X_T(X, T):
    """Transform scalar X, T to np.ndarray"""
    X = np.array(X, dtype=TYPE_FLOAT) if isinstance(X, TYPE_FLOAT) else X
    T = np.array(T, dtype=TYPE_LABEL) if isinstance(T, int) else T
    return X, T


def check_categorical_classification_X_T(X, T):
    """Verify if the input data is for categorical classification
    Args:
        X:
        T: label

    X(N, M) and T(N,) in index label format are expected.
    """
    assert (
            isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT and
            X.ndim == (T.ndim+1) and X.shape[1] > 1 and X.size > 0
    ), f"X.shape(N,M>1) expected for categorical (M>1) classification but {X.shape}"

    assert \
        isinstance(T, np.ndarray) and np.issubdtype(T.dtype, np.integer) and \
        T.shape[0] == X.shape[0] and T.size > 0, \
        "X:shape(N, M) and T:shape(N,) in index label format expected but X %s T %s" \
        % (X.shape, T.shape)


def softmax_cross_entropy_log_loss(
        X: Union[np.ndarray],
        T: Union[np.ndarray],
        offset: TYPE_FLOAT = TYPE_FLOAT(0),
        use_reformula: bool = True,
        need_softmax: bool = True,
        out_P=None,
        out_J=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross entropy log loss for softmax activation -T * log(softmax(X))
    NOTE:
        Handle only the label whose value is True. The reason not to use non-labels to
        calculate the loss is TBD.

        Do not accept binary classification data where P is scalar or P.shape[T] = 1.
        Softmax() is for multi label classification, hence M=P.shape[1] > 1.

        P.ndim == 0 or ((1 < P.ndim == T.ndim) and (P.shape[1] == T.shape[1] == 1))
        is for binary classification.
        Run "P, T = transform_X_T(P, T)" to transform (P, T) to transform P in 1D or
        T in OHE format before calling this function.

    Formula:
        Loss J = -log( exp(xi) / sum(exp(X)) ) can be re-formulated as
        sum(exp(X)) - xi, by which division is eliminated. However, exp(X) can be
        a large value or inf. Need to test which is stable, softmax(X) or sum(exp(X).
        xi is the correct input at index i.

        P[                        T[
          [x0,x1,...xi,...xd-1],    i,
          ...                       ...
        ]                         ]
        j = -log(pi) = -log( exp(xi) / sum(exp(x0),...,exp(xd-1) )
          = log(sum(exp(X))) - xi  : X=[x0,x1,...xi,...xd-1]

    Args:
        X: Input data of shape (N,M) to go through softmax where:
            N is Batch size
            M is Number of nodes
        T: label in the index format of shape (N,).
        offset: small number to avoid np.inf by log(0) by log(0+offset)
        use_reformula: Flag to use "J=sum(exp(X)) - xi" or not
        need_softmax: Flag if P=softmax(X) needs to be returned
        out_P: A location into which the result is stored
        out_J: A location into which the result is stored

    Returns:
        J: Loss value of shape (N,), a loss value per batch.
        P: Activation value softmax(X)
    """
    name = "softmax_cross_entropy_log_loss"
    X, T = transform_scalar_X_T(X, T)
    check_categorical_classification_X_T(X, T)

    N = X.shape[0]
    rows = np.arange(N)     # (N,)
    cols = T                # Same shape (N,) with rows
    assert rows.shape == cols.shape, \
        f"np P indices need the same shape but rows {rows.shape} cols {cols.shape}."

    if use_reformula:
        _A = X[rows, cols]
        J = logarithm(X=np.sum(np.exp(X), axis=-1), offset=offset, out=out_J) - _A
        P = softmax(X=X, out=out_P) if need_softmax else np.empty(X.shape)
    else:
        P = softmax(X=X, out=out_P)
        _A = P[rows, cols]
        J = -logarithm(X=_A, offset=offset, out=out_J)

    if not np.all(np.isfinite(J)):
        raise RuntimeError(f"{name}: Invalid loss J:\n{J}.")

    assert (J.ndim > 0) and (0 < N == J.shape[0]), \
        f"Need J shape ({N},) but {J.shape}."

    Logger.debug("%s: J is [%s] J.shape %s\n", name, J, J.shape)

    return J, P


def check_binary_classification_X_T(X, T):
    """Verify if the input data is for binary classification
    Args:
        X:
        T: label
    """
    assert \
        (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT and X.size > 0) and \
        (
                isinstance(T, np.ndarray) and np.issubdtype(T.dtype, np.integer) and
                np.all(np.isin(T, [0, 1]))
        ) and \
        (
            (X.ndim == 0 and T.ndim == 0) or
            ((1 < X.ndim == T.ndim) and (X.shape[1] == T.shape[1] == 1))
        ), f"Unexpected format for binary classification. X=\n{X}"


def sigmoid_cross_entropy_log_loss(
        X: Union[np.ndarray, TYPE_FLOAT],
        T: Union[np.ndarray, int],
        offset: TYPE_FLOAT = TYPE_FLOAT(0)
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
        T: label in the index format of shape (N,).
        offset: small number to avoid np.inf by log(0) by log(0+offset)

    Returns:
        J: Loss value of shape () for scalar or (N,) a loss value per batch.
        P: Activation value sigmoid(X)
    """
    name = "sigmoid_cross_entropy_log_loss"
    # P, T = transform_X_T(P, T)
    # --------------------------------------------------------------------------------
    # X is scalar and T is a scalar binary OHE label, or
    # T is 2D binary OHE labels e.g. T[[0],[1],[0]], X[[0.9],[0.1],[0.3]].
    # T is binary label 0 or 1
    # --------------------------------------------------------------------------------
    X, T = transform_scalar_X_T(X, T)
    check_binary_classification_X_T(X, T)

    Z1 = TYPE_FLOAT(1.0) + np.exp(-X)    # 1/Z where Z = sigmoid(X) = (1/1 + np.exp(-X))
    P = TYPE_FLOAT(1.0) / Z1
    J = np.multiply((TYPE_FLOAT(1.0) - T), X) + np.log(Z1)
    J = np.squeeze(J, axis=-1)    # Shape from (N,M) to (N,)
    assert np.all(np.isfinite(J))

    return J, P


def generic_cross_entropy_log_loss(
        X: Union[np.ndarray, TYPE_FLOAT],
        T: Union[np.ndarray, int],
        activation: Callable = softmax,
        objective: Callable = categorical_log_loss,
        offset: TYPE_FLOAT = TYPE_FLOAT(0)
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the cross entropy log loss as objective(activation(X), T)
    Args:
        X: input data
        T: label
        activation: activation function
        objective: objective function
        offset: small number to avoid np.inf by log(0) by log(0+offset)
    Returns:
        J: loss in shape (N,)
        P: activation
    """
    X, T = transform_scalar_X_T(X, T)
    if activation == sigmoid:
        check_binary_classification_X_T(X, T)
    elif activation == softmax:
        check_categorical_classification_X_T(X, T)
    else:
        assert False, "currently only sigmoid and softmax are supported."

    P = activation(X)
    J = cross_entropy_log_loss(P, T, objective, offset)
    return J, P


def cross_entropy_log_loss(
        P: Union[np.ndarray, TYPE_FLOAT],
        T: Union[np.ndarray, int],
        f: Callable = categorical_log_loss,
        offset: TYPE_FLOAT = OFFSET_LOG
) -> np.ndarray:
    """Cross entropy log loss [ -t(n)(m) * log(p(n)(m)) ] for multi labels.
    Args:
        P: activation or probabilities from an activation function.
        T: labels
        f: Cross entropy log loss function f(P, T) where P is activation, T is label
        offset: small number to avoid np.inf by log(0) by log(0+offset)

    Returns:
        J: Loss value of shape (N,), a loss value per batch.

    NOTE:
        Handle only the label whose value is True. The reason not to use non-labels to
        calculate the loss is TBD.

        See transform_X_T for the format and shape of P and T.
    """
    name = "cross_entropy_log_loss"
    P, T = transform_X_T(P, T)
    if P.ndim == 0:
        assert False, "P.ndim needs (N,M) after transform_X_T(P, T)"
        # --------------------------------------------------------------------------------
        # P is scalar, T is a scalar binary OHE label. Return -t * log(p).
        # --------------------------------------------------------------------------------
        # assert T.ndim == 0, "P.ndim==0 requires T.ndim==0 but %s" % T.shape
        # return f(P, T, offset)

    if (1 < P.ndim == T.ndim) and (P.shape[1] == T.shape[1] == 1):
        # --------------------------------------------------------------------------------
        # This condition X:(N,1), T(N,1) tells T is the 2D binary OHE labels.
        # T is 2D binary OHE labels e.g. T[[0],[1],[0]], P[[0.9],[0.1],[0.3]].
        # Return -T * log(P)
        # --------------------------------------------------------------------------------
        return np.squeeze(f(P=P, T=T, offset=offset), axis=-1)    # Shape from (N,M) to (N,)
        # return f(P=P, T=T, offset=offset).reshape(-1)

    # ================================================================================
    # Calculate Cross entropy log loss -t * log(p).
    # Select an element P[n][t] at each row n which corresponds to the true label t.
    # Use the Numpy tuple indexing. e.g. P[n=0][t=2] and P[n=3][t=4].
    # P[
    #   (0, 3),
    #   (2, 4)     # The tuple sizes must be the same at all axes
    # ]
    #
    # Tuple indexing selects only one element per row.
    # Beware the numpy behavior difference between P[(n),(m)] and P[[n],[m]].
    # https://stackoverflow.com/questions/66269684
    # P[1,1]  and P[(0)(0)] results in a scalar value, HOWEVER, P[[1],[1]] in array.
    #
    # P shape can be (1,1), (1, M), (N, 1), (N, M), hence P[rows, cols] are:
    # P (1,1) -> P[rows, cols] results in a 1D of  (1,).
    # P (1,M) -> P[rows, cols] results in a 1D of  (1,)
    # P (N,1) -> P[rows, cols] results in a 1D of  (N,)
    # P (N,M) -> P[rows, cols] results in a 1D of  (N,)
    #
    # J shape matches with the P[rows, cols] shape.
    # ================================================================================
    N = batch_size = P.shape[0]
    rows = np.arange(N)     # (N,)
    cols = T                # Same shape (N,) with rows
    assert rows.shape == cols.shape, \
        f"np P indices need the same shape but rows {rows.shape} cols {cols.shape}."

    _P = P[rows, cols]
    Logger.debug("%s: N is [%s]", name, N)
    Logger.debug("%s: P.shape %s", name, P.shape)
    Logger.debug("%s: P[rows, cols].shape %s", name, _P.shape)
    Logger.debug("%s: P[rows, cols] is %s", name, _P)

    J = f(P=_P, T=int(1), offset=offset)

    assert not np.all(np.isnan(J)), f"log(x) caused nan for P \n{P}."
    Logger.debug("%s: J is [%s]", name, J)
    Logger.debug("%s: J.shape %s\n", name, J.shape)

    assert (J.ndim > 0) and (0 < N == J.shape[0]), \
        f"Loss J.shape is expected to be ({N},) but {J.shape}"
    return J


def numerical_jacobian(
        f: Callable[[np.ndarray], np.ndarray],
        X: Union[np.ndarray, TYPE_FLOAT],
        delta: Optional[TYPE_FLOAT] = OFFSET_DELTA
) -> np.ndarray:
    """Calculate Jacobian matrix J numerically with (f(X+h) - f(X-h)) / 2h
    Jacobian matrix element Jpq = df/dXpq, the impact on J by the
    small difference to Xpq where p is row index and q is col index of J.

    Note:
        Beware limitations by the float storage size, e.g. loss of significance.
        https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
        https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/Contents/
        https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/02Numerics/Weaknesses/
        https://www.cise.ufl.edu/~mssz/CompOrg/CDA-arith.html

    Args:
        f: Y=f(X) where Y is a scalar or shape() array.
        X: input of shame (N, M), or (N,) or ()
        delta: small delta value to calculate the f value for X+/-h
    Returns:
        J: Jacobian matrix that has the same shape of X.
    """
    name = "numerical_jacobian"
    X = np.array(X, dtype=TYPE_FLOAT) if isinstance(X, (TYPE_FLOAT, int)) else X
    J = np.zeros_like(X, dtype=TYPE_FLOAT)
    delta = OFFSET_DELTA if (delta is None or delta <= 0.0) else delta
    divider = 2 * delta

    # --------------------------------------------------------------------------------
    # (x+h) or (x-h) may cause an invalid value area for the function f.
    # e.g log loss tries to offset x=0 by adding a small value k as log(0+k).
    # However because k=1e-7 << h=1e-5, f(x-h) causes nan due to log(x < 0)
    # as x needs to be > 0 for log.
    #
    # X and tmp must be float, or it will be int causing float calculation fail.
    # e.g. f(1-h) = log(1-h) causes log(0) instead of log(1-h).
    # --------------------------------------------------------------------------------
    assert (X.dtype == TYPE_FLOAT), f"X must be type {TYPE_FLOAT}"
    assert delta > 0.0 and isinstance(delta, TYPE_FLOAT)

    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp: TYPE_FLOAT = X[idx]

        # --------------------------------------------------------------------------------
        # f(x+h)
        # --------------------------------------------------------------------------------
        X[idx] = tmp + delta
        fx1: Union[np.ndarray, TYPE_FLOAT] = f(X)  # f(x+h)
        Logger.debug(
            "%s: idx[%s] x[%s] (x+h)[%s] fx1=[%s]",
            name, idx, tmp, tmp+delta, fx1
        )

        assert \
            ((isinstance(fx1, np.ndarray) and fx1.size == 1) or isinstance(fx1, TYPE_FLOAT)), \
            f"The f function needs to return scalar or shape () but {fx1}"
        assert np.isfinite(fx1), \
            "f(x+h) caused nan for f %s for X %s" % (f, (tmp + delta))

        # --------------------------------------------------------------------------------
        # f(x-h)
        # --------------------------------------------------------------------------------
        X[idx] = tmp - delta
        fx2: Union[np.ndarray, TYPE_FLOAT] = f(X)
        Logger.debug(
            "%s: idx[%s] x[%s] (x-h)[%s] fx2=[%s]",
            name, idx, tmp, tmp-delta, fx2
        )
        assert \
            ((isinstance(fx2, np.ndarray) and fx2.size == 1) or isinstance(fx2, TYPE_FLOAT)), \
            f"The f function needs to return scalar or shape () but {fx2}"
        assert np.isfinite(fx2), \
            "f(x-h) caused nan for f %s for X %s" % (f, (tmp - delta))

        # --------------------------------------------------------------------------------
        # When f(x+k) and f(x-k) are relatively too close, subtract between them can ben
        # too small, and the precision error 1/f(x) relative to f(x+k) and f(x-k) is much
        # larger relative to that of f(x+k) or f(x-k), hence the result can be unstable.
        # Prevent the subtract df(x) from being too small to f(x) by assuring df(x)/dx is
        # greater than GN_DIFF_ACCEPTANCE_RATIO.
        #
        # e.g. For logistic log loss function f(x) with log(+1e-7) to avoid log(0)/inf.
        # x[14.708627877981929] (x+h)[14.708627878981929] fx1=[14.708628288297405]
        # x[14.708627877981929] (x-h)[14.708627876981929] fx2=[14.708628286670217]
        # (fx1-fx2)=[1.6271872738116144e-09]
        # (fx1-fx2) / fxn < 1e-10. The difference is relatively too small to f(x).
        #
        #
        # If the gradient of f(x) at x is nearly zero, or saturation, then reconsider
        # if using the numerical gradient is fit for the purpose. Prevent the gradient
        # from being too close to zero by f(x+k)-f(x-k) > GN_DIFF_ACCEPTANCE_VALUE
        # --------------------------------------------------------------------------------
        difference = (fx1 - fx2)
        Logger.debug("%s: (fx1-fx2)=[%s]", name, difference)

        derivative_saturation_condition = (difference == 0.0)
        if derivative_saturation_condition:
            fmt = "%s: derivative saturation fx1=fx2=%s detected.\n"
            args = tuple([name, fx1])
            Logger.warning(fmt, *args)
            assert ENFORCE_STRICT_ASSERT, fmt % args

        # subtract_cancellation_condition = (fx1 != fx2) and (
        #         (difference < (fx1 * GN_DIFF_ACCEPTANCE_RATIO)) or
        #         (difference < (fx2 * GN_DIFF_ACCEPTANCE_RATIO))
        # )

        # if subtract_cancellation_condition:
        #     fmt = "%s: potential subtract cancellation (fx1-fx2)/fx < %s detected.\n"\
        #           "(fx1:%s - fx2:%s) is %s, gn %s."
        #     args = tuple([
        #         name,
        #         GN_DIFF_ACCEPTANCE_RATIO,
        #         fx1,
        #         fx2,
        #         difference,
        #         (fx1-fx2) / (2 * delta)
        #     ])
        #     Logger.warning(fmt, *args)
        #     assert ENFORCE_STRICT_ASSERT, fmt % args

        J[idx] = difference / divider
        X[idx] = tmp
        it.iternext()

    if not np.all(np.isfinite(J)):
        raise ValueError(f"{name} caused Nan or Inf")

    gradient_saturation_condition = (abs(J) < GRADIENT_SATURATION_THRESHOLD)
    if np.all(gradient_saturation_condition):
        __J = J[gradient_saturation_condition]
        msg = "%s: The gradient [%s] should be saturated."
        Logger.warning(msg, name, __J)
        assert ENFORCE_STRICT_ASSERT, msg % (name, __J)

    return J


def gn(X, t):
    """Numerical gradient for logistic log loss"""
    return [
        numerical_jacobian(lambda _x: logistic_log_loss(P=sigmoid(_x), T=t), x)
        for x in X
    ]


def compose(*args):
    """compose(f1, f2, ..., fn) == lambda x: fn(...(f2(f1(x))...)"""
    def _(x):
        result = x
        for f in args:
            result = f(result)
        return result

    return _


def prediction_grid(X, W):
    """
    https://cs231n.github.io/neural-networks-case-study/#update
    Args:
        X: Data of D features including a bias. shape (N, D). N is batch size
        W: NN layer weight of shape (M, D) where M is number of classes
    Returns:
        x1_grid: X grid of numpy meshgrid
        x2_grid: Y grid of numpy meshgrid
        Z: predictions in the index format
    """
    h = 0.02
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1_grid, x2_grid = np.meshgrid(
        np.arange(x1_min, x1_max, h),
        np.arange(x2_min, x2_max, h)
    )
    x1 = x1_grid.ravel()
    x2 = x2_grid.ravel()
    x0 = np.ones(x1.size)
    Z = np.matmul(
        np.c_[
            x0,
            x1,
            x2
        ],
        W.T
    )
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(x1_grid.shape)

    return x1_grid, x2_grid, Z


def prediction_grid_2d(x_min, x_max, y_min, y_max, prediction_function):
    """
    https://cs231n.github.io/neural-networks-case-study/#update
    1. Generate the input X from the grid (x_min, y_min, x_max, y_max)
       Add bias x0=1. X:shape(N, 3) where N is number of rows in X
    2. Calculate predictions P = prediction_function(X).
       P:shape(N, M). M is the number of categorical classes.
    3. Transform predictions (N,M) into the index format using argmax.

    Args:
        x_min:
        x_max:
        y_min:
        y_max:
        prediction_function:

    Returns:
        x1_grid: X grid of numpy meshgrid
        x2_grid: Y grid of numpy meshgrid
        Z: predictions in the index format
    """
    h = 0.02
    x1_min, x1_max = x_min - 1, x_max + 1
    x2_min, x2_max = y_min - 1, y_max + 1
    x1_grid, x2_grid = np.meshgrid(
        np.arange(x1_min, x1_max, h),
        np.arange(x2_min, x2_max, h)
    )
    x1 = x1_grid.ravel()
    x2 = x2_grid.ravel()
    P = prediction_function(
        np.c_[
            x1,
            x2
        ]
    )
    P = P.reshape(x1_grid.shape)

    return x1_grid, x2_grid, P


def shuffle(X):
    assert isinstance(X, np.ndarray) and X.ndim > 0
    indices = np.random.permutation(range(X.shape[0]))
    return X[indices]


def shuffle_X_T(X, T):
    assert \
        isinstance(X, np.ndarray) and X.ndim > 0 and \
        isinstance(T, np.ndarray) and T.ndim > 0 and \
        X.shape[0] == T.shape[0]

    indices = np.random.permutation(range(T.shape[0]))
    X = X[indices]
    T = T[indices]
    return X, T


LOSS_FUNCTIONS = {
    "softmax_cross_entropy_log_loss": softmax_cross_entropy_log_loss,
    "sigmoid_cross_entropy_log_loss": sigmoid_cross_entropy_log_loss
}
