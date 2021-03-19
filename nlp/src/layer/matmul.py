"""Matmul layer implementation
"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Final,
    Generator,
    Iterator,
    Callable
)
import logging
import copy
import numpy as np
from layer import Layer
from optimizer import (
    Optimizer,
    SGD,
)
from common import (
    TYPE_LABEL,
    TYPE_FLOAT,
    OFFSET_DELTA,
    numerical_jacobian
)


class Matmul(Layer):
    """MatMul Layer class
    Batch X: shape(N, D)
    --------------------
    Note that the original input X e.g. N x 28x28 pixel image (784 features) has
    (N, D-1) shape WITHOUT the bias feature x0. The layer adds the bias, when the
    input is given at function(X). Hence internally the shape of X is (N, D).

    X has N rows. Each row X[n] (n: 0, ... N-1) has D features of X[j][d] = x(j)(n).
    Batch is referred as 'X' in capital and its scalar element is referred as x.
    X[n] = [x(n)(0), ..., x(n)(d), ..., x(n)(D-1)]. The bias x(n)(0) = 1.

    As an illustration, an input batch X of shape (N, D-1) where (D-1)=784 pixels.
    The layer adds bias x(n)(0)=1 at each row, and the X becomes shape (N, D).
    'X' is not limited to the 0th input layer e.g. image pixels but to any layer.

    Gradient dL/dX: shape(N, D-1) to back-propagate but (N, D) internally
    --------------------
    Internally, dX or dL/dX is of shape (N, D) to match with the shape of X.
    However, dL/dX to back-propagate must match the shape of the output form the
    previous layer, which is (N, D-1) because the Matmul has added the bias.

    Weights W: shape(M, D)
    --------------------
    The layer has M nodes where each node m (m:0, 1, .. M-1) has a weight vector
    W(m) = [w(m)(0), ... , w(m)(d), w(m)(D-1)] INCLUDING the bias weight w(m)(0).
    When the layer is instantiated, W of shape (M, D) needs to be provided,
    because how to configure the weights is up to the user's decision.

    Gradient dL/dW: shape(M, D)
    --------------------
    Has the same shape of W and externaized as-is.


    Numpy slice indexing
    --------------------
    Numpy uses View with slice-indexing, hence would be able to re-use the
    existing memory area to save the memory.

    dX = self.dX[
        ::,
        1::     # Omit the bias
    ]
   """
    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(
            self,
            name: str,
            num_nodes: int,
            W: np.ndarray,
            posteriors: Optional[List[Layer]] = None,
            optimizer: Optimizer = SGD(),
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            W: Weight of shape(M=num_nodes, D). A row is a weight vector of a node.
            posteriors: Post layers to which forward the matmul layer output
            optimizer: Gradient descent implementation e.g SGD, Adam.
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        # --------------------------------------------------------------------------------
        # W: weight matrix of shape(M,D) where M=num_nodes
        # Gradient dL/dW has the same shape shape(M, D) with W because L is scalar.
        #
        # Not use WT because W keeps updated every cycle, hence need to update WT as well.
        # Hence not much performance gain and risk of introducing bugs.
        # self._WT: np.ndarray = W.T          # transpose of W
        # --------------------------------------------------------------------------------
        assert W.shape[0] == num_nodes, \
            f"W shape needs to be ({num_nodes}, D) but {W.shape}."
        self._D = W.shape[1]                # number of features in x including bias
        self._W: np.ndarray = W             # node weight vectors
        self._dW: np.ndarray = np.empty(0, dtype=TYPE_FLOAT)
        self.logger.debug(
            "Matmul[%s] W.shape is [%s], number of nodes is [%s]",
            name, W.shape, num_nodes
        )
        # --------------------------------------------------------------------------------
        # Optimizer for gradient descent
        # Z(n+1) = optimizer.update((Z(n), dL/dZ(n)+regularization)
        # --------------------------------------------------------------------------------
        assert isinstance(optimizer, Optimizer)
        self._optimizer: Optimizer = optimizer

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def W(self) -> np.ndarray:
        """Layer weight vectors W"""
        return self._W

    @property
    def dW(self) -> np.ndarray:
        """Layer weight gradients dW"""
        assert self._dW.size > 0, "dW is not initialized"
        return self._dW

    @property
    def X(self) -> np.ndarray:
        """Latest batch input to the layer"""
        return super().X

    @X.setter
    def X(self, X: np.ndarray):
        """Set X"""
        super(Matmul, type(self)).X.fset(self, X)
        assert self.X.shape[1] == self.D, \
            "X shape needs (%s, %s) but %s" % (self.N, self.D, self.X.shape)

    @property
    def optimizer(self) -> Optimizer:
        """Optimizer instance for gradient descent
        """
        return self._optimizer

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate the layer output Y = X@W.T
        Args:
            X: Batch input data from the input layer without the bias.
        Returns:
            Y: Layer value of X@W.T
        """
        assert isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT, \
            f"Only np array of type {TYPE_FLOAT} is accepted"

        name = "function"
        # --------------------------------------------------------------------------------
        # Y = (W@X + b) could be efficient
        # --------------------------------------------------------------------------------
        self.X = np.c_[
            np.ones(X.shape[0], dtype=TYPE_FLOAT),  # Add bias
            X
        ]
        self.logger.debug(
            "layer[%s].%s: X.shape %s W.shape %s",
            self.name, name, self.X.shape, self.W.shape
        )
        assert self.W.shape == (self.M, self.D), \
            f"W shape needs {(self.M, self.D)} but ({self.W.shape})"

        # --------------------------------------------------------------------------------
        # Allocate array storage for np.func(out=) for Y but not dY.
        # Y:(N,M) = [ X:(N,D) @ W.T:(D,M) ]
        # gradient() need to validate the dY shape is (N,M)
        # --------------------------------------------------------------------------------
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty((self.N, self.M), dtype=TYPE_FLOAT)
            # --------------------------------------------------------------------------------
            # DO NOT allocate memory area for the gradient that has already been calculated.
            # dL/dY is calculated at the post layer, hence it has the buffer allocated already.
            # --------------------------------------------------------------------------------
            # self._dY = np.empty((self.N, self.M), dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # TODO:
        # Because transpose(T) is run everytime matmul is invoked, using the transposed W
        # would save the calculation time. This is probably the reason why cs231n uses
        # in column order format.
        # --------------------------------------------------------------------------------
        np.matmul(self.X, self.W.T, out=self._Y)
        return self.Y

    def gradient(self, dY: Union[np.ndarray, float] = 1.0) -> Union[np.ndarray, float]:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dX: dL/dX of shape (N, D-1) without the bias
        """
        name = "gradient"
        assert isinstance(dY, float) or (isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT)

        dY = np.array(dY).reshape((1, -1)) if isinstance(dY, float) or dY.ndim < 2 else dY
        assert dY.shape == self.Y.shape, \
            "dL/dY shape needs %s but %s" % (self.Y.shape, dY.shape)

        self.logger.debug("layer[%s].%s: dY.shape %s", self.name, name, dY.shape)
        self._dY = dY

        # --------------------------------------------------------------------------------
        # dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        # --------------------------------------------------------------------------------
        np.matmul(self.dY, self.W, out=self._dX)
        assert self.dX.shape == (self.N, self.D), \
            "dL/dX shape needs (%s, %s) but %s" % (self.N, self.D, self.dX.shape)

        return self.dX[
            ::,
            1::     # Omit bias column 0
        ]

    def gradient_numerical(
            self, h: Optional[TYPE_FLOAT] = None
    ) -> List[Union[float, np.ndarray]]:
        """Calculate numerical gradients
        Args:
            h: small number for delta to calculate the numerical gradient
        Returns:
            [dX, dW]: Numerical gradients for X and W without bias
            dX is dL/dX of shape (N, D-1) without the bias to match the original input
            dW is dL/dW of shape (M, D) including the bias weight w0.
        """
        name = "gradient_numerical"
        self.logger.debug("layer[%s].%s", self.name, name)
        L = self.objective

        def objective_X(X: np.ndarray):
            return L(X @ self.W.T)

        def objective_W(W: np.ndarray):
            return L(self.X @ W.T)

        dX = numerical_jacobian(objective_X, self.X, delta=h)
        dX = dX[
            ::,
            1::     # Omit the bias
        ]
        dW = numerical_jacobian(objective_W, self.W, delta=h)
        return [dX, dW]

    def _gradient_descent(self, W, dW, out=None) -> Union[np.ndarray, float]:
        """Gradient descent
        Directly update matrices to avoid the temporary copies
        """
        return self.optimizer.update(W, dW, out=out)

    def update(self) -> List[Union[float, np.ndarray]]:
        """Calculate dL/dW = dL/dY * dY/dW and update W with gradient descent
        dL/dW.T = X.T @ dL/dY is shape (D,M) as  [ X.T:(D, N)  @ dL/dY:(N,M) ].
        Hence dL/dW of shape (M,D):  [ X.T:(D, N)  @ dL/dY:(N,M) ].T.

        Returns:
            [dL/dX, dL/dW]: List of dL/dX and dL/dW.
            dX is dL/dX of shape (N, D-1) without the bias of the layer.
            dW is dL/dW of shape (M, D) including the bias weight w0.
       """
        # --------------------------------------------------------------------------------
        # dL/dW of shape (M,D):  [ X.T:(D, N)  @ dL/dY:(N,M) ].T
        # --------------------------------------------------------------------------------
        dW = np.matmul(self.X.T, self.dY).T
        assert np.array_equal(dW.shape, (self.M, self.D)), \
            f"Gradient dL/dW shape needs {(self.M, self.D)} but ({dW.shape}))"

        if self.logger.level in (logging.DEBUG, logging.WARNING) and np.all(dW < dW / 100):
            self.logger.warning(
                "update(): Gradient descent potentially stalling with dW < 1% of W."
            )

        self._dW = dW
        self._gradient_descent(self.W, self.dW, out=self._W)
        dX = self.dX[
            ::,
            1::     # Omit the bias
        ]
        return [dX, self.dW]
