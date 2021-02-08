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
    """Softmax function from deep-learning-from-scratch
    Args:
        X: batch input data of shape (N X M).
            N: Batch size
            M: Number of nodes
    Returns:
        Prediction probability matrix P of shape (N X M)
    """
    C = np.max(X, axis=-1, keepdims=True)   # オーバーフロー対策
    exp = np.exp(X - C)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_entropy_error(P, T):
    """Cross entropy log loss for multi labels.
    NOTE:
        Handle only the label whose value is True. The reason not to use non-labels to
        calculate the loss is TBD.

    Args:
        P: probability matrix of shape (N x M) from soft-max layer where:
            N is Batch size
            M is Number of nodes
        T: label either One Hot Encoding (OHE) format or indexing format.
    Returns:
        L: Loss value normalized by the batch size N, hence a scalar value.
    """
    if P.ndim == 1:
        T = T.reshape(1, T.size)
        P = P.reshape(1, P.size)
    N = batch_size = P.shape[0]

    # --------------------------------------------------------------------------------
    # Convert into index labels format from OHE ones.
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    # --------------------------------------------------------------------------------
    if T.size == P.size:
        T = T.argmax(axis=1)

    # --------------------------------------------------------------------------------
    # Logg loss with a small value e added to prevent infinitive value by log(+e).
    # Use numpy tuple indexing to select e.g. P[n=0][m=2] and P[n=3][m=4] via:
    # P[
    #   (0, 3),
    #   (2, 4)
    # ]
    # --------------------------------------------------------------------------------
    e = 1e-7
    return -np.sum(np.log(P[np.arange(batch_size), T] + e)) / N
