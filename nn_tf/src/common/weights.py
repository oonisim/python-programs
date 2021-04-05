"""Node weight initializations
"""
from typing import (
    List,
    Dict
)
import numpy as np
from common.constants import (
    TYPE_FLOAT
)

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
    "uniform": uniform,
    "he": he,
    "xavier": xavier,
}


class Weights:
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return Weights.specification(M=3, D=3)

    @staticmethod
    def specification(
            M: int,
            D: int
    ):
        """Generate Weights specification
        Args:
            M: Number of nodes
            D: Number of features
        Returns:
            specification
        """
        return {
            "scheme": Weights.__qualname__,
            "parameters": {
                "M": M,
                "D": D
            }
        }

    @staticmethod
    def build(parameters: Dict):
        """Build weights based on the parameter.
        """
        return Weights(**parameters)

    # ================================================================================
    # Instance
    # ================================================================================
    def __init__(
            self,
            M: int,
            D: int,
            initialization_scheme: str = "uniform"
    ):
        assert M > 0, D > 0 and initialization_scheme in SCHEMES
        self._initialization_scheme = list(SCHEMES.keys())[0]
        self._M = M
        self._D = D
        self._weights = SCHEMES[self.initialization_scheme](M, D)

    @property
    def M(self):
        return self._M

    @property
    def D(self):
        return self._M

    @property
    def initialization_scheme(self):
        return self._initialization_scheme

    @property
    def weights(self):
        return self._weights
