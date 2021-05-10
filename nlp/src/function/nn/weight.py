"""Node weight initializations
"""
from typing import (
    List,
    Dict
)
import numpy as np
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT
)


def xavier(M: TYPE_INT, D: TYPE_INT) -> np.ndarray:
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
    return (np.random.randn(M, D) / np.sqrt(D)).astype(TYPE_FLOAT)


def he(M: TYPE_INT, D: TYPE_INT) -> np.ndarray:
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
    return (np.random.randn(M, D) / np.sqrt(2*D)).astype(TYPE_FLOAT)


def uniform(M: TYPE_INT, D: TYPE_INT) -> np.ndarray:
    """Uniform weight distribution to initialize a weight W:(D,M) of shape (D, M),
    where D is the number of features to a node and M is the number of nodes in a layer.

    Args:
        M: Number of nodes in a layer
        D: Number of feature in a node to process
    Returns:
        W: weight matrix
    """
    assert all([D > 0, M > 0])
    weights = (np.random.uniform(low=-1.0, high=1.0, size=(M, D))).astype(TYPE_FLOAT)
    return weights


SCHEMES = {
    "uniform": uniform,
    "he": he,
    "xavier": xavier,
}


class Weights:
    # ================================================================================
    # Class
    # ================================================================================
    @classmethod
    def class_id(cls):
        """Identify the class
        Avoid using Python implementation specific __qualname__

        Returns:
            Class identifier
        """
        return cls.__qualname__

    @staticmethod
    def specification_template():
        return Weights.specification(M=TYPE_INT(3), D=TYPE_INT(4))

    @staticmethod
    def specification(
            M: TYPE_INT,
            D: TYPE_INT
    ):
        """Generate Weights specification
        Args:
            M: Number of nodes
            D: Number of features
        Returns:
            specification
        """
        return {
            "scheme": Weights.class_id(),
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
            M: TYPE_INT,
            D: TYPE_INT,
            initialization_scheme: str = list(SCHEMES.keys())[0]
    ):
        assert M > 0, D > 0 and initialization_scheme in SCHEMES
        self._initialization_scheme = initialization_scheme
        self._M = M
        self._D = D
        self._weights = SCHEMES[self.initialization_scheme](M, D)
        assert self._weights is not None and self._weights.shape == (self.M, self.D)

    @property
    def M(self):
        return self._M

    @property
    def D(self):
        return self._D

    @property
    def initialization_scheme(self):
        assert \
            self._initialization_scheme in SCHEMES, \
            "Invalid weight initialization scheme"
        return self._initialization_scheme

    @property
    def weights(self) -> np.ndarray:
        return self._weights
