"""Objective function layer implementations
"""
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Union,
    Final,
    Generator,
    Iterator,
    Callable
)
import logging
import numpy as np
from . base import Layer
from common.functions import (
    cross_entropy_log_loss,
    softmax,
    numerical_jacobian
)


class SoftmaxWithLogLoss(Layer):
    """Softmax cross entropy log loss class
    Combined with the log loss because calculating gradients separately is not simple.
    When combined, the dL/dX, impact on L by input delta dX, is (P - T)/N.
    """
    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(self, name: str, num_nodes: int, log_level: int = logging.ERROR):
        """Initialize a softmax layer
        Args
            name: Instance ID
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        self._N: int = -1                   # Batch size of X
        self._M: int = -1                   # Number of nodes
        self._P: np.ndarray = np.empty(())  # Probabilities of shape (N, M)
        self._J: np.ndarray = np.empty(())  # Cross entropy log loss of shape (N,).
        self._L: np.ndarray = -np.inf       # Objective value of shape ()

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def M(self) -> int:
        """Number of nodes"""
        assert self._M > 0, "M is not initialized"
        return self._M

    @property
    def P(self) -> np.ndarray:
        """Softmax probabilities of shape (N,M)"""
        assert self._P and self.P.size > 0, "P is not initialized"
        assert self._P.shape == (self.N, self.M)
        return self._P

    @property
    def J(self) -> np.ndarray:
        """Cross entropy log loss of shape(N,)"""
        assert self._J and self.J.size == self.N, "J is not initialized"
        return self._J

    @property
    def L(self) -> np.ndarray:
        """Cross entropy log loss of shape()"""
        assert self._L and self.L.size == 1, "L is not initialized"
        return self._L

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: np.ndarray) -> Union[np.ndarray, float]:
        """Layer output
        Args:
            X: Input of shape (N,M) to calculate the probabilities for the M nodes.
            T: Labels for input X. Shape(N,M) for OHE label and shape(N,) for index label.
        Returns:
            L: Objective value of shape ()
        """
        assert X and X.shape[0] == self.T.shape[0], \
            f"Batch size of X {X.shape[0]} and T {self.T.shape[0]} needs to be the same."

        # DO NOT reshape for numpy tuple indexing to work.
        # self._X = X.reshape(1, X.size) if X.ndim == 1 else X
        # self.T = T.reshape(1, T.size) if T.ndim == 1 else T
        self.X = X
        self._N, self._M = X.shape

        self._P = softmax(self.X)
        self._J = cross_entropy_log_loss(self.P, self.T)    # dJ/dP -> -T/P
        self._L = np.sum(self.J)                            # dL/dJ ->  1/N

        return self.L

    def gradient(self, dL: Union[np.ndarray, float] = 1.0) -> Union[np.ndarray, float]:
        """Calculate the gradient dL/dX, the impact on L by the input dX (NOT dL)
        dL: dF/dL is the impact on F by dL when the output L has a post layer F.
        dJ: dL/dJ is the impact on L by dJ where J = cross_entropy_log_loss(P,T) = 1/N
        dJ/dP: impact on J by the softmax output dP = -T/P.
        dP/dX: impact on P by the softmax input dX
        dL/dX = dF/dL * dL/dJ * (dJ/dP * dP/dX) where (dJ/dP * dP/dX) = P - T

        Args:
            dL: Gradient, impact by the loss dL, given from the post layer.
        Returns:
            dL/dX: (P-T)/N of shape (N, M)
        """
        # --------------------------------------------------------------------------------
        # dL/dX = dL/dJ * dJ/dX. dJ/dX = -T/P is the gradient at cross-entropy log loss.
        # If the input X is a scalar and T=0, then dJ/dA is always 0 (-T/P = 0/P).
        # --------------------------------------------------------------------------------
        if (isinstance(self.X, float) or self.X.ndim == 0) and self.T == 0:
            return 0    # -0 / P

        # Gradient dJ:(N,) = dL/dJ is 1/N for each node.
        dJ: np.ndarray = np.ones(self.N, dtype=float) / self.N

        # --------------------------------------------------------------------------------
        # Calculate the gradient at the layer
        # --------------------------------------------------------------------------------
        if self.T.size == self.P.size:  # Label is in OHE format.
            assert self.T.shape == self.P.shape, \
                f"T.shape {self.T.shape} and P.shape {self.P.shape} should be the same."

            dX = dL * dJ * (self.P - self.T)
        else:
            # --------------------------------------------------------------------------------
            # np.copy() is a shallow copy and will not copy object elements within arrays.
            # To ensure all elements within an object array are copied, use copy.deepcopy().
            # Here, reusing dP elements hence shallow copy.
            # --------------------------------------------------------------------------------
            dX = self.P.copy()

            # --------------------------------------------------------------------------------
            # Calculate dJ/dX=(P-T) using numpy tuple indexing. The tuple size must be the same.
            # e.g. select P[n=0][m=2] and P[n=3][m=4]:
            # P[
            #   (0, 3),
            #   (2, 4)
            # ]
            # --------------------------------------------------------------------------------
            rows = np.arange(self.N)    # tuple index for rows
            cols = self.T               # tuple index for columns
            # assert rows.ndim == cols.ndim and len(rows) == len(cols), \
            assert rows.shape == cols.shape, \
                f"numpy tuple indices {rows.shape} and {cols.shape} need to be the same."

            # Extract T=1 elements of P and calculate (P-T) = (P-1)
            dX[
                rows,
                cols
            ] -= 1
            # dF/dL * dL/dJ * (P-T) = dF/dL * (P-T) / N
            np.multiply((dL * dJ), dX, out=dX)

        return dX
