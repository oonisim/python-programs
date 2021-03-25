"""Node weight initializations
"""
from typing import (
    List
)
import numpy as np


def xavier(M: int, D: int) -> np.ndarray:
    """Xavier weight initialization for base-symmetric activations e.g. sigmoid/tanh
    Gaussian distribution with the standard deviation of sqrt(1/D) to initialize
    a weight W:(D,M) of shape (D, M), where D is the number of features to a node and
    M is the number of nodes in a layer.

    Args:
        M: Number of nodes in a layer
        D: Number of feature in a node to process
    Returns:
        W: weight matrix
    """
    assert all([D > 0, M > 0])
    return np.random.randn(M, D) / np.sqrt(D)


def he(M: int, D: int) -> np.ndarray:
    """He weight initialization for base-asymmetric linear activations e.g. ReLU.
    Gaussian distribution with the standard deviation of sqrt(2/D) to initialize
    a weight W:(D,M) of shape (D, M), where D is the number of features to a node and
    M is the number of nodes in a layer.

    Args:
        M: Number of nodes in a layer
        D: Number of feature in a node to process
    Returns:
        W: weight matrix
    """
    assert all([D > 0, M > 0])
    return np.random.randn(M, D) / np.sqrt(2*D)


def uniform(M: int, D: int) -> np.ndarray:
    """Uniform weight distribution to initialize a weight W:(D,M) of shape (D, M),
    where D is the number of features to a node and M is the number of nodes in a layer.

    Args:
        M: Number of nodes in a layer
        D: Number of feature in a node to process
    Returns:
        W: weight matrix
    """
    assert all([D > 0, M > 0])
    return np.random.uniform(low=0.0, high=1.0, size=(M, D))


SCHEMES = {
    "he": he,
    "xavier": xavier,
    "uniform": uniform
}
