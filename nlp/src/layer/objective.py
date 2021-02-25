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
    transform_X_T,
    cross_entropy_log_loss,
    softmax,
    numerical_jacobian
)


# TODO
# class SigmoidWithLogLoss(Layer):


class SoftmaxWithLogLoss(Layer):
    """Softmax cross entropy log loss class
    Combined with the log loss because calculating gradients separately is not simple.
    When combined, the dL/dX, impact on L by input delta dX, is (P - T)/N.

    Note:
        Number of nodes M == number of feature D at the softmax with log loss layer.
        T has either in the OHE (One Hot Encoder) label format or index label format.
        For OHE T.shape[0] == N and T.shape[1] == M == D. For index, T.size == N.

    TODO:
        Add regularization cost 0.5 * l2 * np.sum(W ** 2) to L. Need a way to access
        W in each matmul layer.
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

        # At softmax loss layer, features in X is the same with num nodes (1:1)
        self._D = num_nodes

        self._P: np.ndarray = np.empty(())  # Probabilities of shape (N, M)
        self._J: np.ndarray = np.empty(())  # Cross entropy log loss of shape (N,).
        # Use Y for output in the consistent manner.
        # self._L: np.ndarray = -np.inf       # Objective value of shape ()

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def P(self) -> np.ndarray:
        """Softmax outputs probabilities of shape (N,M)"""
        assert \
            isinstance(self._P, np.ndarray) and self._P.size > 0 and \
            self._P.shape == (self.N, self.M), "Y is not initialized"
        return self._P

    @property
    def J(self) -> np.ndarray:
        """Cross entropy log loss of shape(N,)"""
        assert isinstance(self._J, np.ndarray) and self._J.size == self.N, \
            "J is not initialized"
        return self._J

    @property
    def L(self) -> np.ndarray:
        """Cross entropy log loss of shape()
        Alias of the layer output Y
        """
        assert isinstance(self._Y, np.ndarray) and self._Y.size == 1, \
            "Y is not initialized"
        return self._Y

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[np.ndarray, float]) -> np.ndarray:
        """Softmax wth Log Loss layer output L
        Pre-requisite:
            T has been set before calling.
        Note:
            The Softmax Log Loss output is normalized by the batch size N.
            Otherwise the NN loss is dependent on the batch size.

            For instance, with a MNIST image, the loss or NN performance should be
            per-image, so that the loss can be a universal unit to compare.
            It should NOT be dependent on what batch size was used.

            L = cross_entropy_log_loss(softmax(X), T)) / N

        Args:
            X: Input of shape (N,M) to calculate the probabilities for the M nodes.
            T: Labels for input X. Shape(N,M) for OHE label and shape(N,) for index label.
        Returns:
            L: Objective value of shape ()
        """
        # Pre-requisite: T has been set before calling.
        assert self.T.size > 0 and self.M == self.D

        # --------------------------------------------------------------------------------
        # Validate X, T and transform them to be able to use numpy tuple-like indexing.
        # P[
        #   (0,3),
        #   (1,5)
        # ]
        # --------------------------------------------------------------------------------
        self.X, self.T = transform_X_T(X, self.T)
        self.logger.debug(
            "layer[%s] function(): X.shape %s T.shape %s", self.name, self.X.shape, self.T.shape
        )

        assert \
            (
                self.X.ndim == 0 and self.T.ndim == 0 and
                self.X.size == self.T.size == self.M == 1
            ) or \
            (
                self.X.ndim == 1 and self.T.ndim in {0, 1} and
                self.X.size == self.T.size == self.M
            ) or \
            (
                self.X.ndim >= 2 and self.T.ndim in {1, 2} and
                self.X.shape[0] == self.T.shape[0] and
                self.X.shape[1] == self.M
            ), \
            "X shape %s with T.shape %s does not match with the Layer node number M[%s]" \
            % (self.X.shape, self.T.shape, self.M)

        # --------------------------------------------------------------------------------
        # Softmax probabilities P:(N, M) for each label m in each batch n.
        # --------------------------------------------------------------------------------
        self._P = softmax(self.X)

        # --------------------------------------------------------------------------------
        # Cross entropy log loss J:(N,) where j(n) for each batch n (n: 0, ..., N-1).
        # Gradient dJ/dP -> -T/P
        # --------------------------------------------------------------------------------
        self._J = cross_entropy_log_loss(self.P, self.T)

        # --------------------------------------------------------------------------------
        # Total loss L. Calculate the sum and convert scalar back to np.ndarray
        # If a for sum(a) is 0-d array or axis is None, a scalar is returned.
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # dL/dJ = 1 at this point because dsum(J)/dJ = 1
        # --------------------------------------------------------------------------------
        _L = np.array(np.sum(self.J, axis=-1))

        # --------------------------------------------------------------------------------
        # Normalize L with the batch size N to be dependent from the batch size.
        # dL/dJ = 1/N because f(X)=X/N -> df(X)/dX = 1/N
        # --------------------------------------------------------------------------------
        L = np.array(_L / self.N, dtype=float)
        self._Y = L # L is alias of Y.

        self.logger.debug("function() L = %s", self.Y)
        return self.Y

    def gradient(self, dY: Union[np.ndarray, float] = float(1)) -> Union[np.ndarray, float]:
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
        dY = np.array(dY) if isinstance(dY, float) else dY
        assert dY.shape == self.Y.shape, \
            "dY/dY shape needs %s but %s" % (self.Y.shape, dY.shape)

        # --------------------------------------------------------------------------------
        # dY/dX = dY/dJ * dJ/dX. dJ/dX = -T/P is the gradient at cross-entropy log loss.
        # If the input X is a scalar and T = 0, then dJ/dA is always 0 (-T/P = 0/P).
        # --------------------------------------------------------------------------------
        if (isinstance(self.X, float) or self.X.ndim == 0) and self.T == 0:
            return np.array(float(0), dtype=float)    # -0 / P

        # Gradient dJ:(N,) = dY/dJ is 1/N.
        dJ: np.ndarray = np.ones(self.N, dtype=float) / float(self.N)

        # --------------------------------------------------------------------------------
        # Calculate the layer gradient
        # --------------------------------------------------------------------------------
        if self.T.ndim == self.P.ndim:
            self.logger.debug("gradient(): Label is in OHE format or scalar index")
            assert (self.T.size == self.P.size) and (self.T.shape == self.P.shape), \
                "T.shape %s and P.shape %s should be the same for the OHE labels." \
                % (self.T.shape, self.P.shape)
            dX = dY * dJ * (self.P - self.T)

        else:
            self.logger.debug("gradient(): Label is index format")
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
            dF = (dJ * dY)
            dF = dF[::, np.newaxis]
            # dF/dY * dY/dJ * (P-T) = dF/dY * (P-T) / N
            self.logger.debug("dF.shape is %s dF is \n%s.\n", dF.shape, dF)
            self.logger.debug("T is %s dX.shape %s.\n", self.T, dX.shape)
            self.logger.debug(
                "gradient(): dX[rows, T] = %s,\ndX is \n%s.\n", dX[rows, cols], dX
            )
            np.multiply(dF, dX, out=dX)

        self.logger.debug("gradient(): dL/dX is \n%s.\n", dX)
        return dX
