"""

"""
import numpy
import numpy as np
from typing import (
    Tuple
)

from scipy.stats import ortho_group
from util_constant import (
    TYPE_FLOAT,
    TYPE_INT,
)


def is_all_close(
        x: np.ndarray,
        y: np.ndarray,
        atol: TYPE_FLOAT = TYPE_FLOAT(1e-5)
) -> bool:
    """Check if array x and y are close enough
    """
    return np.allclose(
        x, y, atol=TYPE_FLOAT(1e-5)
    )


def get_cosine_similarity(x: numpy.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity
    Args:
        x: array in shape (N, D) for N number of D dimension vectors. Reshaped to (1, D) when it is (D,)
        y: array in shape (M, D) for M number of D dimension vectors. Reshaped to (1, D) when it is (D,)
    Returns: array in shape (N, M)
    """
    assert x.ndim > 0 and x.size > 1
    assert y.ndim > 0 and y.size > 1
    if x.ndim < 2:
        assert len(x) > 1
        x = x.reshape((-1, len(x)))
        reshaped = True
    if y.ndim < 2:
        assert len(y) > 1
        y = y.reshape((-1, len(y)))

    return np.dot(x / np.linalg.norm(x, axis=1, keepdims=True), y.T / np.linalg.norm(y.T, axis=0, keepdims=True))


def get_orthogonal_3d_vectors() -> Tuple[np.ndarray, np.ndarray]:
    """Generate two orthogonal vectors in 3D space
    Return (x, y)
    """
    dimension: int = 3
    seed: np.ndarray = np.random.randn(dimension).astype(TYPE_FLOAT)
    x: np.ndarray = np.random.randn(dimension).astype(TYPE_FLOAT)
    x -= x.dot(seed) * seed / np.linalg.norm(seed)**2
    y = np.cross(seed, x)
    return x, y


def get_orthogonal_vectors(dimension: int) -> np.ndarray:
    """Generate orthogonal vectors in shape (D, D) where each row is an orthogonal vectors
    https://math.stackexchange.com/questions/2098897/generating-random-orthogonal-matrices
    Args:
        dimension: dimension of the vectors
    Returns: array in shape (D, D) where each row is an orthogonal vector (row order matrix)
    """
    assert dimension > 1, f"dimension must be at least 2, got [{dimension}]"

    # --------------------------------------------------------------------------------
    # The eigen vectors of a square matrix are orthogonal
    # Does not work producing complex number eigen values/vectors
    # --------------------------------------------------------------------------------
    # while True:
    #     x: np.ndarray = np.random.randn(dimension, dimension).astype(TYPE_FLOAT)
    #     if np.linalg.det(x) != 0:
    #         break
    #
    # w, v = np.linalg.eig(x)
    # return v.T

    return ortho_group.rvs(dim=dimension)
