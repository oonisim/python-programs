"""Data for classifications"""
import numpy as np


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
    # Unit vector w of dimension d, dividing by its magnitude
    W = np.random.randn(d)
    W = W / np.linalg.norm(W)

    # Generate X:(N,D) and set bias=1 to x0
    X = np.random.randn(n, d)
    X[
        ::,
        0
    ] = 1   # bias

    # Label t = 1 if X dot w > 0 else 0
    T = (np.einsum('ij,j', X, W) > 0).astype(int)

    return X, T, W
