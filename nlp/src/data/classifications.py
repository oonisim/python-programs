"""Data for classifications"""
import numpy as np
from np import (
    rotate,
    is_point_inside_sector
)


def linear_separable(d: int = 3, n: int = 10000):
    """Generate a data set X to linearly separate.
    Bias x0 is added to create n+1 dimensional data

    Args:
        d: number of dimension of the data including the bias
        n: number of data to generate
    Returns:
           X dot W > 0 is True and < 0 for False.
        X: d+1 dimension data (x0, x1, ... xn) where x0=1 as bias
        T: labels. If Xi dot W > 0, then 1 else 0
        W: Vector orthogonal to the linear hyper plane that separates the data.
    """
    assert n >= 10, f"n {n} is too small"
    # Unit vector w of dimension d, dividing by its magnitude
    # Generate X:(N,D) and set bias=1 to x0
    X = np.random.randn(n, d)
    X[
        ::,
        0
    ] = 1   # bias

    while True:
        W = np.random.randn(d)
        W = W / np.linalg.norm(W)

        # Label t = 1 if X dot w > 0 else 0
        T = (np.einsum('ij,j', X, W) > 0).astype(int)

        # Each label has at least 30% of the data
        if 0.3 < np.sum(T[T == 1]) / n < 0.7:
            break

    return X, T, W


def linear_separable_sectors(n: int = 10000, d: int = 2, m: int = 3, r=0.0):
    """Generate a data set X to linearly separate into m sectors.
    Args:
        n: number of coordinates
        d: number of dimension of the data
        m: number of classes (sectors)
        r: angle to rotate X
    Returns:
        X: d dimension data (x0, x1, ... xn) of shape (n, d)
        T: labels (0, 1, ...M-1) of m classees
        B: List of the base (sin(θ), cos(θ)) of each section
    """
    assert m > 1, f"m {m} > 1 required to split."
    assert d == 2, "currently only d==2 is valid"
    assert n >= m, "At least m instance of coordinates required."

    X = np.random.uniform(-1, 1, (n, d))
    T = np.zeros(n, dtype=int)
    sector = (2 * np.pi) / float(m)  # angle of a sector

    B = np.array([[1.0, 0.0]])
    for i in range(1, m):
        base = (sector * i)
        T[is_point_inside_sector(X=X, base=base, coverage=sector)] = i
        B = np.r_[
            B,
            np.array([[np.cos(base), np.sin(base)]])
        ]

    r = r % (2 * np.pi)
    X = rotate(X, r)
    B = rotate(B, r)
    return X, T, B


def spiral(n: int, d: int=2, m: int=3):
    """Generate spiral data points
    https://cs231n.github.io/neural-networks-case-study/#data
    Args:
        n: number of points per class
        d: dimensionality
        m: number of classes
    Returns:
        X: spiral data coordinates of shape (n, d)
        y: label (0, 1, ...m-1)
    """
    N = n
    D = d
    M = m
    X = np.zeros((N * M, D))
    y = np.zeros(N * M, dtype='uint8')
    for j in range(M):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y