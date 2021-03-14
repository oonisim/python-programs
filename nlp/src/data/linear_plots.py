"""Data for classifications"""
import numpy as np
from mathematics import (
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


def linear_separable_sectors(
        n: int = 10000, d: int = 3, m: int = 3, r: float = 1.0, rotation: float = 0.0
):
    """Generate a data set X to linearly separate into m sectors.
    Args:
        n: number of coordinates
        d: number of dimension of the data
        m: number of classes (sectors)
        r: radius
        rotation = angle to rotate X
    Returns:
        X: d dimension data (x0, x1, ... xn) of shape (n, d)
        T: labels (0, 1, ...M-1) of m classees
        B: List of the base (sin(θ), cos(θ)) of each section
    """
    assert m > 1, f"m {m} > 1 required to split."
    assert d == 3, "currently only d==3 for 2D (bias, x1, x2) is valid"
    assert n >= m, "At least m instance of coordinates required."

    # Remove bias.
    d = d - 1

    Z = np.random.uniform(0, 2 * np.pi, n)
    radii = np.random.uniform(0.0, r, n)
    X = np.c_[
        radii * np.cos(Z),
        radii * np.sin(Z)
    ]
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

    rotation = rotation % (2 * np.pi)
    X = rotate(X, rotation)
    B = rotate(B, rotation)

    # Add bias
    X = np.c_[
        np.ones(n),
        X
    ]       # Add bias=1
    return X, T, B
