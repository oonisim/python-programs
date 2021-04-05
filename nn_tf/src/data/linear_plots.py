"""Data for classifications"""
import numpy as np
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL
)
from mathematics import (
    rotate,
    is_point_inside_sector
)


def linear_separable(d: int = 2, n: int = 10000):
    """Generate a data set X to linearly separate.
    Args:
        d: number of dimension of the data
        n: number of data to generate
    Returns:
           X dot W > 0 is True and < 0 for False.
        X: d dimension data (x1, ... xn)
        T: labels. If Xi dot W > 0, then 1 else 0
        W: Vector orthogonal to the linear hyper plane that separates the data.
    """
    assert n >= 10, f"n {n} is too small"
    # Unit vector w of dimension d, dividing by its magnitude
    # Generate X:(N,D) and set bias=1 to x0
    X = np.random.randn(n, d)
    _X = np.c_[
        np.ones(n),     # Bias
        X
    ]

    while True:
        W = np.random.randn(d+1)    # +1 for bias weight w0
        W = W / np.linalg.norm(W)

        # Label t = 1 if X dot w > 0 else 0
        T = (np.einsum('ij,j', _X, W) > 0).astype(int)

        # Each label has at least 30% of the data
        if 0.3 < np.sum(T[T == 1]) / n < 0.7:
            break

    return X, T, W


def linear_separable_sectors(
        n: int = 10000, d: int = 2, m: int = 3, r: float = 1.0, rotation: float = 0.0
):
    """Generate plots X in a circle to be linearly separated into m sectors.
    The sectors are rotated as per the "rotation" parameter.

    Args:
        n: number of coordinates
        d: number of dimension of the data
        m: number of classes (sectors)
        r: radius
        rotation: angle to rotate X
    Returns:
        X: d dimension data (x0, x1, ... xn) of shape (n, d)
        T: labels (0, 1, ...M-1) of m classes
        B: List of the base (sin(θ), cos(θ)) of each section
    """
    assert m > 1, f"m {m} > 1 required to split."
    assert d == 2, "currently only d==2 for 2D (bias, x1, x2) is valid"
    assert n >= m, "At least m instance of coordinates required."

    # Generate plots in the circle
    Z = np.random.uniform(0, 2 * np.pi, n)
    radii = np.random.uniform(0.0, r, n)
    X = np.c_[
        radii * np.cos(Z),
        radii * np.sin(Z)
    ]
    T = np.zeros(n, dtype=TYPE_LABEL)
    sector = (2 * np.pi) / float(m)  # angle of a sector

    # The initial vector (1, 0) forms the start angle of the 0th sector.
    B = np.array([[1.0, 0.0]])

    # The label has been already set to 0 for the 0th sector. Hence, the
    # splitting the circle into m sectors starts with the 1st sector.
    for i in range(1, m):       # Start with 1st sector.
        base = (sector * i)     # Start angle of the i-th sector.
        T[
            is_point_inside_sector(X=X, base=base, coverage=sector)
        ] = i                   # Label i for plots in the i-th sector
        B = np.r_[
            B,
            np.array([[np.cos(base), np.sin(base)]])
        ]

    # Rotate the circle and the start angles of the sectors
    rotation = rotation % (2 * np.pi)
    X = rotate(X, rotation)
    B = rotate(B, rotation)
    return X, T, B
