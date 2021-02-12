"""Xavier weight initialization for base-symmetric functions e.g. sigmoid/tanh
which have linear activation around the base.
"""
from typing import (
    List
)
import numpy as np


def xavier(D: int, M: int) -> np.ndarray:
    """Gaussian distribution with the standard deviation of sqrt(1/D) to initialize
    a weight W:(D,M) of shape (D, M), where D is the number of features to a node and
    M is the number of nodes in a layer.

    Args:
        D: Number of feature in a node to process
        M: Number of nodes in a layer
    Returns:
        W: weight matrix
    """
    assert all([D > 0, M > 0])
    return np.random.randn(D, M) / np.sqrt(D)
