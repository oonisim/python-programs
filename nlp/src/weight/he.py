"""He weight initialization for base-asymmetric linear activation functions e.g. ReLU.
"""
from typing import (
    List
)
import numpy as np


def he(D: int, M: int) -> np.ndarray:
    """Gaussian distribution with the standard deviation of sqrt(2/D) to initialize
    a weight W:(D,M) of shape (D, M), where D is the number of features to a node and
    M is the number of nodes in a layer.

    Args:
        D: Number of feature in a node to process
        M: Number of nodes in a layer
    Returns:
        W: weight matrix
    """
    assert all([D > 0, M > 0])
    return np.random.randn(D, M) / np.sqrt(2*D)
