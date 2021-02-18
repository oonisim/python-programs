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

Logger = logging.getLogger("functions")
Logger.setLevel(logging.DEBUG)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_grad(X):
    return (1.0 - sigmoid(X)) * sigmoid(X)


def relu(X):
    return np.maximum(0, X)


def relu_grad(X):
    grad = np.zeros_like(X)
    grad[X >= 0] = 1
    return grad


def softmax(X: Union[np.ndarray, float]):
    """Softmax P = exp(X) / sum(exp(X))
    Args:
        X: batch input data of shape (N,M).
            N: Batch size
            M: Number of nodes
    Returns:
        Probability P of shape (N,M)
    """
    # --------------------------------------------------------------------------------
    # exp(x-c) to prevent the infinite exp(x) for a large value x, with c = max(x).
    # keepdims=True to be able to broadcast.
    # --------------------------------------------------------------------------------
    C = np.max(X, axis=-1, keepdims=True)   # オーバーフロー対策
    exp = np.exp(X - C)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_entropy_log_loss(
        P: Union[np.ndarray, float],
        T: Union[int, np.ndarray],
        e: float = 1e-7
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
        e: small number to avoid np.inf by log(0) by log(0+e)
    Returns:
        J: Loss value of shape (N,)
    """
    # --------------------------------------------------------------------------------
    # Convert scalar and 1D array into (N,M) shape to to run run P[rows, cols]
    # --------------------------------------------------------------------------------
    P = np.array(P) if isinstance(P, float) else P
    T = np.array(T) if isinstance(T, (float, int)) else T
    if P.ndim <= 1:
        T = T.reshape(1, T.size)
        P = P.reshape(1, P.size)

    # Label is integer
    T = T.astype(int)
    assert T.shape[0] == P.shape[0], \
        f"Batch size of T {T.shape[0]} and P {P.shape[0]} should be the same."

    N = batch_size = P.shape[0]

    # --------------------------------------------------------------------------------
    # Convert OHE format into index label format of shape (N,).
    # T in OHE format has the shape (N,M) with P, hence same size N*M.
    # --------------------------------------------------------------------------------
    if T.size == P.size:
        T = T.argmax(axis=1)

    # --------------------------------------------------------------------------------
    # Numpy tuple indexing. The tuple size must be the same.
    # e.g. select P[n=0][m=2] and P[n=3][m=4]:
    # P[
    #   (0, 3),
    #   (2, 4)
    # ]
    # --------------------------------------------------------------------------------
    rows = np.arange(N)     # 1D tuple index for rows
    cols = T                # 1D tuple index for columns
    assert rows.ndim == cols.ndim == 1 and rows.size == cols.size, \
        f"np tuple indices size need to be same but rows {rows.size} cols {cols.size}."

    # Log( +e) prevents the infinitive value log(0).
    J = -np.sum(np.log(P[rows, cols] + e), axis=-1)

    Logger.debug("P.shape %s", P.shape)
    Logger.debug("P[rows, cols].shape %s", P[rows, cols].shape)
    Logger.debug("N is [%s]", N)
    Logger.debug("J is [%s]", J)
    Logger.debug("J.shape %s", J.shape)

    # --------------------------------------------------------------------------------
    # Possible P shape can be (1,1), (1, M), (N, 1), (N, M), for each of which
    # P (1,1) -> P[rows, cols] ()
    # P (1,M) -> P[rows, cols] ()
    # P (N,1) -> P[rows, cols] (N,)
    # P (N,M) -> P[rows, cols] (N,)
    # --------------------------------------------------------------------------------
    assert (1 < N == J.shape[0]) or (N == 1 and J.ndim == 0)
    return J


def numerical_jacobian(
        f: Callable[[np.ndarray], np.ndarray],
        X: Union[np.ndarray, float],
        h: float = 1e-5
) -> np.ndarray:
    """Calculate Jacobian matrix J numerically with (f(X+h) - f(X-h)) / 2h
    Jacobian matrix element Jpq = df/dXpq, the impact on J by the
    small difference to Xpq where p is row index and q is col index of J.

    Args:
        f: Y=f(X) where Y is a scalar or shape() array.
        X: input of shame (N, M), or (N,) or ()
        h: small delta value to calculate the f value for X+/-h
    Returns:
        J: Jacobian matrix that has the same shape of X.
    """
    assert h > 0.0
    X = np.array(X) if isinstance(X, (float, int)) else X
    J = np.zeros_like(X)

    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = X[idx]
        X[idx] = tmp_val + h
        fx1 = f(X)  # f(x+h)

        X[idx] = tmp_val - h
        fx2 = f(X)  # f(x-h)

        # --------------------------------------------------------------------------------
        # Set the gradient element scalar value or shape()
        # --------------------------------------------------------------------------------
        g = (fx1 - fx2) / (2 * h)
        assert g.size == 1, "The f function needs to return scalar or shape ()"
        J[idx] = g

        X[idx] = tmp_val
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
