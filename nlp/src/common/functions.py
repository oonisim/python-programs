"""DNN functions
Those marked as "From deep-learning-from-scratch" is copied from the github.
https://github.com/oreilly-japan/deep-learning-from-scratch
"""
import numpy as np


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


def softmax(X):
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


def cross_entropy_log_loss(P, T) -> float:
    """Cross entropy log loss [ -t(n)(m) * log(p(n)(m)) ] for multi labels.
    NOTE:
        Handle only the label whose value is True. The reason not to use non-labels to
        calculate the loss is TBD.

    Args:
        P: probabilities of shape (N,M) from soft-max layer where:
            N is Batch size
            M is Number of nodes
        T: label either in OHE format of shape (N,M) or index format of shape (N,).
           OHE: One Hot Encoding
    Returns:
        J: Loss value of shape (N,)
    """
    if P.ndim == 1:
        T = T.reshape(1, T.size)
        P = P.reshape(1, P.size)

    assert T.shape[0] == P.shape[0], \
        f"Batch size of T {T.shape[0]} and P {P.shape[0]} should have been the same [{N}]."

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
    rows = np.arange(N)     # tuple index for rows
    cols = T                # tuple index for columns
    assert rows.ndim == cols.ndim and len(rows) == len(cols), \
        f"numpy tuple indices need to have the same size."

    # --------------------------------------------------------------------------------
    # Log( +e) prevents the infinitive value log(0).
    # --------------------------------------------------------------------------------
    e = 1e-7
    J = -np.sum(np.log(P[rows, cols] + e), axis=-1)
    assert J.shape[0] == N

    return J