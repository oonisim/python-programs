"""Loss layer implementations
"""
from typing import (
    List,
    Dict,
    Optional,
    Final,
    Generator,
    Iterator,
    Callable
)
import logging
import copy
import numpy as np
from .base import Layer
from ..optimizer import (
    Optimizer,
    SGD,
)
from .. common.functions import (
    cross_entropy_log_loss,
    softmax
)


class SoftmaxWithLogLoss(Layer):
    """Softmax cross entropy log loss class
    Combined with the log loss because calculating gradients separately is not simple.
    When combined, the dL/dX, impact on L by input delta dX, is (P - T)/N.
    """
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(self, name: str):
        """Initialize a softmax layer
        Args
            name: Instance ID
        """
        super().__init__(name=name)

        self.T: np.ndarray = None   # Labels of shape (N, M) for OHE or (N,) for index labels.
        self.X: np.ndarray = None
        self.N: int = 0             # Batch size of X
        self.P: np.ndarray = None   # Probabilities of shape (N, M)
        self.J: np.ndarray = None   # Cross entropy log loss of shape (N,).
        self.L: float = -np.inf     # Loss of shape ()/scalar.

    def output(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Layer output
        Args:
            X: Input of shape (N, M) to calculate the probabilities for the M nodes.
            T: Labels for input X. Shape(N,M) for OHE label and shape(N,) for index label.
        Returns:
            L: Loss value of shape ()
        """
        assert X and T and X.shape[0] == T.shape[0], \
            f"Batch size of X {X.shape[0]} and T {T.shape[0]} needs to be the same."

        # DO NOT reshape for numpy tuple indexing to work.
        # self.X = X.reshape(1, X.size) if X.ndim == 1 else X
        # self.T = T.reshape(1, T.size) if T.ndim == 1 else T
        self.X = X
        self.T = T
        self.N = X.shape[0]

        self.P = softmax(self.X)
        self.J = cross_entropy_log_loss(self.P, self.T)         # dJ/dP -> -T/P
        self.L = np.sum(self.J)                                 # dL/dJ ->  1/N

        assert self.L.ndim == 0
        return self.L

    def gradient(self, dL=1):
        """Calculate the gradient dL/dX, the impact on L by the input dX (NOT dL)

        dL: dF/dL is the impact on F by dL when the output L has a post layer F.
        dJ: dL/dJ is the impact on L by dJ where J = cross_entropy_log_loss(P,T).
        dJ/dP: impact on J by the softmax output dP.
        dP/dX: impact on P by the softmax input dX.

        The gradient dL/dX = dF/dL * dL/dJ * (dJ/dP * dP/dX)
        (dJ/dP * dP/dX) = P - T

        Args:
            dL: Gradient, impact by the loss dL, given from the post layer.
        Returns:
            Gradient dL/dX = dL/dPof shape (N, M)
        """
        # Gradient dJ:(N,) = dL/dJ is 1/N for each node.
        dJ: float = np.ones(self.N) / self.N

        if self.T.size == self.P.size:  # Label is in OHE format.
            dX = dL * dJ * (self.P - self.T)
        else:
            # --------------------------------------------------------------------------------
            # np.copy() is a shallow copy and will not copy object elements within arrays.
            # To ensure all elements within an object array are copied, use copy.deepcopy().
            # --------------------------------------------------------------------------------
            # dX = self.P.copy()
            dX = copy.deepcopy(self.P)
            # --------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------
            # Calculate P-T using numpy tuple indexing. The tuple size must be the same.
            # e.g. select P[n=0][m=2] and P[n=3][m=4]:
            # P[
            #   (0, 3),
            #   (2, 4)
            # ]
            # --------------------------------------------------------------------------------
            rows = np.arange(self.N)    # tuple index for rows
            cols = self.T               # tuple index for columns
            assert rows.ndim == cols.ndim and len(rows) == len(cols), \
                f"numpy tuple indices need to have the same size."

            dX[
                rows,
                cols
            ] -= 1
            np.multiply((dL * dJ), dX, out=dX)

        return dX
