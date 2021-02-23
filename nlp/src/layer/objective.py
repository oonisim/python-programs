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

        self._P: np.ndarray = np.empty(())  # Probabilities of shape (N, M)
        self._J: np.ndarray = np.empty(())  # Cross entropy log loss of shape (N,).
        self._L: np.ndarray = -np.inf       # Objective value of shape ()

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def P(self) -> np.ndarray:
        """Softmax probabilities of shape (N,M)"""
        assert \
            isinstance(self._P, np.ndarray) and self.P.size > 0 and \
            self._P.shape == (self.N, self.M), "P is not initialized"
        return self._P

    @property
    def J(self) -> np.ndarray:
        """Cross entropy log loss of shape(N,)"""
        assert isinstance(self._J) and self.J.size == self.N, "J is not initialized"
        return self._J

    @property
    def L(self) -> np.ndarray:
        """Cross entropy log loss of shape()"""
        assert isinstance(self._L) and self.L.size == 1, "L is not initialized"
        return self._L

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[np.ndarray, float]) -> np.ndarray:
        """Layer output
        Args:
            X: Input of shape (N,M) to calculate the probabilities for the M nodes.
            T: Labels for input X. Shape(N,M) for OHE label and shape(N,) for index label.
        Returns:
            L: Objective value of shape ()
        """


        self.X = X
        self.logger.debug(
            "layer[%s] function(): X.shape %s T.shape %s", self.name, X.shape, self.T.shape
        )

        self._P = softmax(self.X)
        self._J = cross_entropy_log_loss(self.P, self.T)    # dJ/dP -> -T/P
        self._L = np.sum(self.J)                            # dL/dJ ->  1/N

        return self.L

    def gradient(self, dY: Union[np.ndarray, float] = 1.0) -> Union[np.ndarray, float]:
        """Calculate the gradient dL/dX, the impact on F by the input dX.
        F: Objective function of the layer.
        Y: Output of the layer
        J = cross_entropy_log_loss(P,T)
        dY: dF/dY is the the impact on the objective F by dY.
        dJ: dY/dJ = 1/N is the impact on Y by dJ where .
        dP: dJ/dP = -T/P is the impact on J by the softmax output dP.
        dP/dX: impact on P by the softmax input dX
        dF/dX = dF/dY * dY/dJ * (dJ/dP * dP/dX) where (dJ/dP * dP/dX) = (P - T)/N

        Args:
            dY: Gradient, impact by the loss dY, given from the post layer.
        Returns:
            dF/dX: (P-T)/N of shape (N, M)
        """
        assert isinstance(dY, float) or (isinstance(dY, np.ndarray) and dY.dtype == float)

        dY = np.array(dY).reshape((1, -1)) if isinstance(dY, float) or dY.ndim < 2 else dY
        assert dY.shape == (self.N, self.M), \
            "dY/dY shape needs (%s, %s) but %s" % (self.N, self.M, dY.shape)

        # --------------------------------------------------------------------------------
        # dY/dX = dY/dJ * dJ/dX. dJ/dX = -T/P is the gradient at cross-entropy log loss.
        # If the input X is a scalar and T = 0, then dJ/dA is always 0 (-T/P = 0/P).
        # --------------------------------------------------------------------------------
        if (isinstance(self.X, float) or self.X.ndim == 0) and self.T == 0:
            return np.array(0.0, dtype=float)    # -0 / P

        # Gradient dJ:(N,) = dY/dJ is 1/N.
        dJ: np.ndarray = np.ones(self.N, dtype=float) / self.N

        # --------------------------------------------------------------------------------
        # Calculate the layer gradient
        # --------------------------------------------------------------------------------
        if self.T.size == self.P.size:  # Label is in OHE format.
            assert self.T.shape == self.P.shape, \
                "T.shape %s and P.shape %s should be the same for the OHE labels." \
                % (self.T.shape, self.P.shape)
            dX = dY * dJ * (self.P - self.T)

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
            assert rows.shape == cols.shape, \
                f"numpy tuple indices {rows.shape} and {cols.shape} need to be the same."

            # Extract T=1 elements of P and calculate (P-T) = (P-1)
            dX[
                rows,
                cols
            ] -= 1.0
            # dF/dY * dY/dJ * (P-T) = dF/dY * (P-T) / N
            np.multiply((dY * dJ), dX, out=dX)

        return dX
