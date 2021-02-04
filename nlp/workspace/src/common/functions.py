"""DNN functions
Those marked as "From deep-learning-from-scratch" is copied from the github.
https://github.com/oreilly-japan/deep-learning-from-scratch
"""
import numpy as np


def softmax(x):
    """Softmax function from deep-learning-from-scratch
    Args:
        x: batch input data of shape (N x M).
            N: Batch size
            M: Number of nodes
    Returns:
        Prediction probability matrix P of shape (N x M)
    """
    c = np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    exp = np.exp(x - c)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_entropy_error(p, t):
    """Cross entropy log loss for multi labels.
    NOTE:
        Handle only the label whose value is True. The reason not to use non-labels to
        calculate the loss is TBD.

    Args:
        p: probability matrix of shape (N x M) from soft-max layer where:
            N is Batch size
            M is Number of nodes
        t: label either One Hot Encoding (OHE) format or indexing format.
    Returns:
        L: Loss value normalized by the batch size N, hence a scalar value.
    """
    if p.ndim == 1:
        t = t.reshape(1, t.size)
        p = p.reshape(1, p.size)
    N = batch_size = p.shape[0]

    # --------------------------------------------------------------------------------
    # Convert into index labels format from OHE ones.
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    # --------------------------------------------------------------------------------
    if t.size == p.size:
        t = t.argmax(axis=1)

    # --------------------------------------------------------------------------------
    # Logg loss with a small value e added to prevent infinitive value by log(+e).
    # Use numpy tuple indexing to select e.g. P[n=0][m=2] and P[n=3][m=4] via:
    # P[
    #   (0, 3),
    #   (2, 4)
    # ]
    # --------------------------------------------------------------------------------
    e = 1e-7
    return -np.sum(np.log(p[np.arange(batch_size), t] + e)) / N
