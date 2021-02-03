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
