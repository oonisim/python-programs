"""DNN functions
Those marked as "From deep-learning-from-scratch" is copied from the github.
https://github.com/oreilly-japan/deep-learning-from-scratch
"""
import logging
import copy
from typing import (
    Optional,
    Union,
    Tuple,
    Callable
)
import numpy as np
from common import (
    OFFSET_DELTA,
    OFFSET_LOG,
    OFFSET_STD,
    BOUNDARY_SIGMOID,
    GN_DIFF_ACCEPTANCE_VALUE,
    GN_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_SATURATION_THRESHOLD,
    ENFORCE_STRICT_ASSERT
)


Logger = logging.getLogger("functions")
# Logger.setLevel(logging.DEBUG)


def standardize(X: Union[np.ndarray, float], out=None, eps: float = OFFSET_STD):
    """Standardize X per-feature basis.
    Each feature is independent from other features, hence standardize per feature.
    1. Calculate the mean per each column, not entire matrix.
    2. Calculate the variance per each column
    3. Standardize mean/sqrt(variance+eps) where small positive eps prevents sd from being zero.

    Args:
        X: Input data to standardize per feature basis.
        out: Output storage for np.divide(dividend, divisor, out)
        eps: A small positive value to assure sd > 0 for deviation/sd will not be div-by-zero.
             Allow eps=0 to simulate or compare np.std().
    """
    assert (isinstance(X, np.ndarray) and X.dtype == float and eps >= 0.0)
    if isinstance(X, float):
        X = np.array(X).reshape(1, -1)
    if X.ndim <= 1:
        X = X.reshape(1, -1)

    N = X.shape[0]
    mean = np.sum(X, axis=0) / N    # mean of each feature
    deviation = X - mean
    variance = np.var(X, axis=0)
    sd = np.sqrt(variance + eps)

    standardized = np.divide(deviation, sd, out)
    assert np.all(np.isfinite(standardized))

    return standardized


def logarithm(
        X: Union[np.ndarray, float],
        offset: Optional[Union[np.ndarray, float]] = OFFSET_LOG
) -> Union[np.ndarray, float]:
    """Wrapper for np.log(x) to set the hard-limit for x
    Assure x > 0 and x is above a certain lower boundary for log(x)

    Args:
        X: > domain value for log
        offset: The lower boundary of acceptable X value.
    Returns:
        np.log(X) with X > offset
    """
    assert (isinstance(X, np.ndarray) and X.dtype == float) or isinstance(X, float)

    offset = OFFSET_LOG if not offset else offset
    assert offset > 0
    if np.all(X > offset):
        _X = X
    elif isinstance(X, np.ndarray):
        _X = copy.deepcopy(X)
        _X[X <= offset] = offset
    else:
        _X = offset

    return np.log(_X)


def sigmoid_reverse(y):
    """
    Args:
        y: y=sigmoid(x)
    Returns:
        x: x that gives y=sigmoid(x)
    """
    return np.log(y/(1-y))


def sigmoid(
    X: Union[float, np.ndarray],
    boundary: Optional[Union[np.ndarray, float]] = BOUNDARY_SIGMOID
) -> Union[int, float, np.ndarray]:
    """Sigmoid activate function
    Args:
        X: > domain value for log
        boundary: The lower boundary of acceptable X value.
    """
    assert (isinstance(X, np.ndarray) and X.dtype == float) or isinstance(X, float)
    boundary = BOUNDARY_SIGMOID if (boundary is None or boundary <= float(0)) else boundary
    assert boundary > 0

    if np.all(np.abs(X) <= boundary):
        _X = X
    elif isinstance(X, np.ndarray):
        _X = copy.deepcopy(X)
        _X[X > boundary] = boundary
        _X[X < -boundary] = -boundary
    else:
        _X = np.sign(X) * boundary

    return 1 / (1 + np.exp(-1 * _X))


def sigmoid_gradient(X: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
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
    assert (isinstance(X, np.ndarray) and X.dtype == float) or isinstance(X, float)
    Z = sigmoid(X)
    return Z * (1.0 - Z)


def relu(X: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """ReLU activation function"""
    assert (isinstance(X, np.ndarray) and X.dtype == float) or isinstance(X, float)
    return np.maximum(0, X)


def relu_gradient(X: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """ReLU gradient
    Args:
        X:
    Returns: gradient
    """
    assert (isinstance(X, np.ndarray) and X.dtype == float) or isinstance(X, float)
    grad = np.zeros_like(X)
    grad[X >= 0] = 1
    return grad


def softmax(X: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Softmax P = exp(X) / sum(exp(X))
    Args:
        X: batch input data of shape (N,M).
            N: Batch size
            M: Number of nodes
    Returns:
        P: Probability of shape (N,M)
    """
    name = "softmax"
    assert isinstance(X, float) or (isinstance(X, np.ndarray) and X.dtype == float), \
        "X must be float or ndarray(dtype=float)"

    # --------------------------------------------------------------------------------
    # exp(x-c) to prevent the infinite exp(x) for a large value x, with c = max(x).
    # keepdims=True to be able to broadcast.
    # --------------------------------------------------------------------------------
    C = np.max(X, axis=-1, keepdims=True)   # オーバーフロー対策
    exp = np.exp(X - C)
    P = exp / np.sum(exp, axis=-1, keepdims=True)
    Logger.debug("%s: X %s exp %s P %s", name, X, exp, P)

    return P


def categorical_log_loss(
        P: np.ndarray, T: np.ndarray, offset: Optional[float] = None
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
        P: np.ndarray, T: np.ndarray, offset: Optional[float] = None
) -> Union[np.ndarray, float]:
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


def logistic_log_loss_gradient(X, T, offset: float = BOUNDARY_SIGMOID):
    """Derivative of
    Z = sigmoid(X), dZ/dX = Z(1-Z)
    L = -( T*log(Z) + (1-T) * log(1-Z) ) dL/dZ = -T(1-T)/Z + (1-T)/(1-Z)
   """
    assert np.all(np.isin(T, [0, 1]))
    Z = sigmoid(X)

    # dL/dX = (Z - T)
    return -T * (1-Z) + Z * (1-T)


def transform_X_T(
        X: Union[np.ndarray, float], T: Union[np.ndarray, int]
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
    assert (isinstance(X, np.ndarray) and X.dtype == float) or isinstance(X, float), \
        "Type of P must be float"
    assert (isinstance(T, np.ndarray) and T.dtype == int) or isinstance(T, int), \
        "Type of T must be integer"

    if isinstance(X, float) or X.ndim == 0:
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
        T = np.array(T, dtype=int) if isinstance(T, int) else T

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


def cross_entropy_log_loss(
        P: Union[np.ndarray, float],
        T: Union[np.ndarray, int],
        f: Callable = categorical_log_loss,
        offset: float = OFFSET_LOG
) -> np.ndarray:
    """Cross entropy log loss [ -t(n)(m) * log(p(n)(m)) ] for multi labels.
    Assumption:
        Label is integer 0 or 1 for an OHE label and any integer for an index label.

    NOTE:
        Handle only the label whose value is True. The reason not to use non-labels to
        calculate the loss is TBD.

    Args:
        P: probabilities of shape (N,M) from soft-max layer where:
            N is Batch size
            M is Number of nodes
        T: label either in OHE format of shape (N,M) or index format of shape (N,).
           OHE: One Hot Encoding
        f: Cross entropy log loss function
        offset: small number to avoid np.inf by log(0) by log(0+offset)

    Returns:
        J: Loss value of shape (N,), a loss value per batch.
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

    assert not np.all(np.isnan(J)), "log(x) caused nan for P \n%s." % P
    Logger.debug("%s: J is [%s]", name, J)
    Logger.debug("%s: J.shape %s\n", name, J.shape)

    assert (J.ndim > 0) and (0 < N == J.shape[0]), \
        "Loss J.shape is expected to be (%s,) but %s" % (N, J.shape)
    return J


def numerical_jacobian(
        f: Callable[[np.ndarray], np.ndarray],
        X: Union[np.ndarray, float],
        delta: float = OFFSET_DELTA
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
    X = np.array(X, dtype=float) if isinstance(X, (float, int)) else X
    J = np.zeros_like(X, dtype=float)

    # --------------------------------------------------------------------------------
    # (x+h) or (x-h) may cause an invalid value area for the function f.
    # e.g log loss tries to offset x=0 by adding a small value k as log(0+k).
    # However because k=1e-7 << h=1e-5, f(x-h) causes nan due to log(x < 0)
    # as x needs to be > 0 for log.
    #
    # X and tmp must be float, or it will be int causing float calculation fail.
    # e.g. f(1-h) = log(1-h) causes log(0) instead of log(1-h).
    # --------------------------------------------------------------------------------
    assert (X.dtype == float), "X must be float type"
    assert delta > 0.0

    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp: float = X[idx]

        # --------------------------------------------------------------------------------
        # f(x+h)
        # --------------------------------------------------------------------------------
        X[idx] = tmp + delta
        fx1: Union[np.ndarray, float] = f(X)  # f(x+h)
        Logger.debug(
            "%s: idx[%s] x[%s] (x+h)[%s] fx1=[%s]",
            name, idx, tmp, tmp+delta, fx1
        )

        assert \
            ((isinstance(fx1, np.ndarray) and fx1.size == 1) or isinstance(fx1, float)), \
            "The f function needs to return scalar or shape () but %s" % fx1
        assert np.isfinite(fx1), \
            "f(x+h) caused nan for f %s for X %s" % (f, (tmp + delta))

        # --------------------------------------------------------------------------------
        # f(x-h)
        # --------------------------------------------------------------------------------
        X[idx] = tmp - delta
        fx2: Union[np.ndarray, float] = f(X)
        Logger.debug(
            "%s: idx[%s] x[%s] (x-h)[%s] fx2=[%s]",
            name, idx, tmp, tmp-delta, fx2
        )
        assert \
            ((isinstance(fx2, np.ndarray) and fx2.size == 1) or isinstance(fx2, float)), \
            "The f function needs to return scalar or shape () but %s" % fx2
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
        Logger.debug("%s: (fx1-fx2)=[%s]", name, (fx1-fx2))
        difference = np.abs(fx1 - fx2)
        subtract_cancellation_condition = (
                (difference < (fx1 * GN_DIFF_ACCEPTANCE_RATIO)) or
                (difference < (fx2 * GN_DIFF_ACCEPTANCE_RATIO))
        )
        if subtract_cancellation_condition:
            fmt = "%s: subtract cancellation ((fx1-fx2)/fx) < %s detected. gn %s."
            args = tuple([
                name,
                GN_DIFF_ACCEPTANCE_RATIO,
                difference,
                (fx1-fx2) / (2 * delta)
            ])
            Logger.warning(fmt, *args)
            assert ENFORCE_STRICT_ASSERT, fmt % args

        # --------------------------------------------------------------------------------
        # Set the gradient element scalar value or shape()
        # --------------------------------------------------------------------------------
        g: Union[np.ndarray, float] = np.subtract(fx1, fx2) / (2 * delta)
        assert np.isfinite(g)

        J[idx] = g
        Logger.debug("%s: idx[%s] j=[%s]", name, idx, g)

        X[idx] = tmp
        it.iternext()

    gradient_saturation_condition = np.all(np.abs(J) < GRADIENT_SATURATION_THRESHOLD)
    if gradient_saturation_condition:
        msg = "%s: The gradient [%s] should be saturated."
        Logger.warning(msg, name, g)
        assert ENFORCE_STRICT_ASSERT, msg % (name, g)

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
