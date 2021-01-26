from typing import List
import numpy as np


def prime_numpy_version(n: int) -> List[int]:
    """Generate prime numbers
    P = Z - D
    P: Primes
    Z: All integer numbers
    D: Divisible (non-primes)

    Args:
        n: upper bound to generate primes in-between 0 and n (inclusive)
    Returns:
        list of prime numbers
    """
    arm = range(2, np.floor(n / 2).astype(int) + 1)
    x, y = np.meshgrid(*([arm] * 2))

    Z = range(2, n + 1)
    D = x * y
    Diff = np.setdiff1d

    P = Diff(Z, D[D <= n].ravel())
    return P.tolist()