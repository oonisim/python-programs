"""DNN functions
Those marked as "From deep-learning-from-scratch" is copied from the github.
https://github.com/oreilly-japan/deep-learning-from-scratch
"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Callable,
    NoReturn,
    Final
)
import logging
import numpy as np


OFFSET_FOR_DELTA = 1e-6
OFFSET_FOR_LOG = (OFFSET_FOR_DELTA + 1e-7)  # Avoid log(0) -> inf by log(0+offset)

Logger = logging.getLogger("functions")
Logger.setLevel(logging.DEBUG)


def sigmoid(X: Union[float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Sigmoid activate function
    Args:
        X:
    """
    assert X.dtype == float
    return 1 / (1 + np.exp(-X))


def sigmoid_gradient(X: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Sigmoid gradient
    Args:
        X:
    Returns: gradient
    """
    assert X.dtype == float
    return (1.0 - sigmoid(X)) * sigmoid(X)


def relu(X: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """ReLU activation function"""
    assert X.dtype == float
    return np.maximum(0, X)


def relu_gradient(X: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """ReLU gradient
    Args:
        X:
    Returns: gradient
    """
    assert X.dtype == float
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
        Probability P of shape (N,M)
    """
    assert X.dtype == float

    # --------------------------------------------------------------------------------
    # exp(x-c) to prevent the infinite exp(x) for a large value x, with c = max(x).
    # keepdims=True to be able to broadcast.
    # --------------------------------------------------------------------------------
    C = np.max(X, axis=-1, keepdims=True)   # オーバーフロー対策
    exp = np.exp(X - C)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_entropy_log_loss(
        P: Union[np.ndarray, float],
        T: Union[np.ndarray, int],
        offset: float = OFFSET_FOR_LOG
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
        offset: small number to avoid np.inf by log(0) by log(0+offset)
    Returns:
        J: Loss value of shape (N,), a loss value per batch.
    """
    assert (isinstance(P, np.ndarray) and P.dtype == float) or isinstance(P, float), \
        "Type of P must be float"
    assert (isinstance(T, np.ndarray) and T.dtype == int) or isinstance(T, int), \
        "Type of T must be integer"

    # --------------------------------------------------------------------------------
    # For scalar values, calculate -t * log(p) and return
    # --------------------------------------------------------------------------------
    if isinstance(P, float) or P.ndim == 0:
        return -T * np.log(P+offset)

    # --------------------------------------------------------------------------------
    # T.ndim is 1
    # --------------------------------------------------------------------------------
    if isinstance(T, int):
        T = np.array(T)
    else:
        T = T.reshape(T.size)

    # --------------------------------------------------------------------------------
    # Handle P/T as (N,M) matrix to use P[[N], [M]] to select T=1 elements.
    # --------------------------------------------------------------------------------
    if P.ndim <= 1:
        P = P.reshape(1, P.size)

    N = batch_size = P.shape[0]

    # --------------------------------------------------------------------------------
    # Convert OHE format into index label format of shape (N,).
    # T in OHE format has the shape (N,M) with P, hence same size N*M.
    # --------------------------------------------------------------------------------
    if T.size == P.size:
        T = T.argmax(axis=-1)

    # --------------------------------------------------------------------------------
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
    # --------------------------------------------------------------------------------
    rows = np.arange(N)     # 1D index for rows
    cols = T                # () or (N,) 1D index for columns
    assert rows.size == cols.size, \
        f"np tuple indices size need to be same but rows {rows.size} cols {cols.size}."

    # --------------------------------------------------------------------------------
    # Log loss per batch. Log(0+e) prevents the infinitive value log(0).
    # NOTE: numerical gradient check.
    #   Numerical gradient calculate f(x+/-h) with a small h e.g. 1e-5.
    #   However, when x=0 and h >> e, f(0-h)=log(e-h) is np.nan as because x in log(x)
    #   cannot be < 0.
    # --------------------------------------------------------------------------------
    assert np.all((P[rows, cols] + offset) > 0), \
        "x for log(x) needs to be > 0 but %s." % (P[rows, cols] + offset)

    _P = P[rows, cols]
    Logger.debug("P[rows, cols] is %s" % P[rows, cols])
    J = -np.log(_P + offset)
    assert not np.all(np.isnan(J))

    Logger.debug("P.shape %s", P.shape)
    Logger.debug("P[rows, cols].shape %s", P[rows, cols].shape)
    Logger.debug("N is [%s]", N)
    Logger.debug("J is [%s]", J)
    Logger.debug("J.shape %s\n", J.shape)

    assert 0 < N == J.shape[0], \
        "Loss J.shape is expected to be (%s,) but %s" % (N, J.shape)
    return J


def numerical_jacobian(
        f: Callable[[np.ndarray], np.ndarray],
        X: Union[np.ndarray, float],
        delta: float = OFFSET_FOR_DELTA
) -> np.ndarray:
    """Calculate Jacobian matrix J numerically with (f(X+h) - f(X-h)) / 2h
    Jacobian matrix element Jpq = df/dXpq, the impact on J by the
    small difference to Xpq where p is row index and q is col index of J.

    Args:
        f: Y=f(X) where Y is a scalar or shape() array.
        X: input of shame (N, M), or (N,) or ()
        delta: small delta value to calculate the f value for X+/-h
    Returns:
        J: Jacobian matrix that has the same shape of X.
    """
    X = np.array(X, dtype=float) if isinstance(X, (float, int)) else X
    J = np.zeros_like(X, dtype=float)

    # --------------------------------------------------------------------------------
    # (x+h) or (x-h) may cause an invalid value area for the function f.
    # e.g log loss tries to offset x=0 by adding a small value e as log(0+e).
    # However because e=1e-7 << h=1e-5, f(x-h) causes nan due to log(x < 0)
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

        X[idx] = tmp + delta
        fx1: float = f(X)  # f(x+h)
        assert not np.isnan(fx1), \
            "numerical delta f(x+h) caused nan for f %s for X %s" \
            % (f, (tmp + delta))

        X[idx] = tmp - delta
        fx2: float = f(X)  # f(x-h)
        assert not np.isnan(fx2), \
            "numerical delta f(x-h) caused nan for f %s for X %s" \
            % (f, (tmp - delta))

        # --------------------------------------------------------------------------------
        # Set the gradient element scalar value or shape()
        # --------------------------------------------------------------------------------
        g = (fx1 - fx2) / (2 * delta)
        assert g.size == 1, "The f function needs to return scalar or shape ()"
        J[idx] = g

        X[idx] = tmp
        it.iternext()

    return J


def compose(*args):
    """compose(f1, f2, ..., fn) == lambda x: fn(...(f2(f1(x))...)"""
    def _(x):
        result = x
        for f in args:
            result = f(result)
        return result

    return _
