"""
Pearson’s coefficient of correlation
"""
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Union,
    Optional,
)
import os
import sys
import logging
import numpy as np

physics_scores: List[int] = [15, 12, 8,  8,  7,  7,  7,  6,  5,  3]
history_scores: List[int] = [10, 25, 17, 11, 13, 17, 20, 13, 9,  15]


def var(X: Union[List[float], List[int], np.ndarray], ddof: int = 1) -> np.ndarray:
    """
    Args:
        X: data points
        ddof: degree of freedom
    Returns:
    """
    variance: np.ndarray = np.inf

    # Variance = sum((X - mean)**2) / (N - ddof) where N = len(X)
    mean: np.ndarray = np.mean(X)
    n: int = len(X)

    variance = np.sum(
        (X - mean)**2
    ) / (n - ddof)

    return variance


def std(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    # std = sqrt(variance(X)
    return np.sqrt(var(X, ddof))


def cov(X: np.ndarray, Y: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Covariance between the observations X and Y
    Same with CovarianceMatrix[0][1] from numpy.corrcoef.

    Args:
        X: observation 1
        Y: observation 2
        ddof: degree of freedom
    Returns: Covariance between the observations X and Y
    """
    # cov = sum((X-X_mean) * (Y-Y-mean)) / (N - ddof)
    n: int = len(X)
    return np.sum(
        (X - np.mean(X)) * (Y - np.mean(Y))
    ) / (n - ddof)


def correlation(X: np.ndarray, Y: np.ndarray, ddof: int = 1):
    """Pearson’s coefficient of correlation
    Args:
        X: observation 1
        Y: observation 2
        ddof: degree of freedom
    Returns: Pearson’s coefficient of correlation
    """
    # correlation = cov(X, Y, ddof) / (std(X, ddof) * std(Y, ddof))
    return cov(X, Y, ddof) / std(X, ddof=ddof) / std(Y, ddof=ddof)


def main():
    X: np.ndarray = np.arange(10)
    Y: np.ndarray = np.random.randint(0, 20, size=10)

    assert np.allclose(var(X=X, ddof=1), np.var(a=X, ddof=1), atol=1e-2), \
        f"var(X) is {var(X)}, np.var(X) is {np.var(X)}"

    assert np.allclose(std(X, ddof=1), np.std(X, ddof=1)), \
        f"std(X) is {std(X)}, np.std(X) is {np.std(X)}"

    assert np.allclose(cov(X=X, Y=Y, ddof=1), np.cov(X, Y, ddof=1)[0][1])
    assert np.allclose(correlation(X=X, Y=Y, ddof=1), np.corrcoef(x=X, y=Y)[0][1], atol=1e-2), \
        f"correlation(X=X, Y=Y, ddof=1): {correlation(X=X, Y=Y, ddof=1)} " \
        f"np.corrcoef(x=X, y=Y, ddof=1)): {np.corrcoef(x=X, y=Y, ddof=1)[0][1]}"

    print(correlation(
        X=physics_scores,
        Y=history_scores,
        ddof=1
    ))


if __name__ == "__main__":
    main()
