"""Matmul layer implementation
Note:
    Python passes a pointer to an object storage if it is mutable.
    Matrices can take up lage storage, hence to avoid temporary copies,
    directly update the target storage area.
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
from common.functions import (
    numerical_jacobian
)


class Matmul(Layer):
    """MatMul Layer class"""
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

        Batch X: shape(N, D)
        --------------------
        A batch X has N rows and each row x has D features where x[d=0] is the bias value 1.
        X[j] = [x(j)(0), x(j)(1), ... x(j)(D-1)]
        "input" is not limited to the 0th input layer e.g. image pixels but to any layer.

        Note that the original input data e.g. a 28x28 pixel image (784 features) does not
        have D-1 features without the bias. A bias 1 is to be prepended as x[0].

        Weights W: shape(M, D) where M=num_nodes
        --------------------
        Node k (k:0, 1, .. M-1) has a weight vector W(k):[w(k)(0), w(k)(1), ... w(k)(D-1)].
        w(k)(0) is a bias weight. Each w(k)(i) amplifies i-th feature in an input x.
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        # --------------------------------------------------------------------------------
        # Validate the expected dimensions.
        # `W` has `M` nodes (nodes)
        # --------------------------------------------------------------------------------
        self.logger.debug(
            "Matmul name [%s] W.shape is [%s], numer of nodes is [%s]",
            name, W.shape, num_nodes
        )
        assert W.shape[0] == num_nodes, \
            "W shape needs to be (%s, D) but %s." % (num_nodes, W.shape)

        # --------------------------------------------------------------------------------
        # W: weight matrix of shape(M,D) where M=num_nodes
        # Gradient dL/dW has the same shape shape(M, D) with W because L is scalar.
        # --------------------------------------------------------------------------------
        self._D = W.shape[1]                # number of features in x
        self._W: np.ndarray = W             # node weight vectors
        self._dW: np.ndarray = np.empty((num_nodes, W.shape[1]), dtype=float)

        # Layers to which forward the matmul output
        self._posteriors: List[Layer] = posteriors
        self._num_posteriors: int = len(posteriors) if posteriors else -1

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
            X: Batch input data from the input layer.
        Returns:
            Y: Layer value of X@W.T
        """
        self.X = X
        self.logger.debug(
            "layer[%s] function(): X.shape %s W.shape %s", self.name, X.shape, self.W.shape
        )
        assert self.W.shape == (self.M, self.D), \
            f"W shape needs {(self.M, self.D)} but ({self.W.shape})"

        # --------------------------------------------------------------------------------
        # Allocate array storge for np.func(out=) for Y but not dY.
        # Y:(N,M) = [ X:(N,D) @ W.T:(D,M) ]
        # gradient() need to validate the dY shape is (N,M)
        # --------------------------------------------------------------------------------
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty((self.N, self.M), dtype=float)
            # --------------------------------------------------------------------------------
            # DO NOT allocate memory area for the gradient that has already been calculated.
            # dL/dY is calculated at the post layer, hence it has the buffer allocated already.
            # --------------------------------------------------------------------------------
            # self._dY = np.empty((self.N, self.M), dtype=float)

        np.matmul(X, self.W.T, out=self._Y)
        return self.Y

    def forward(self, X: np.ndarray) -> Union[np.ndarray, float]:
        """Calculate and forward-propagate the matmul output Y to post layers if exist.
        """
        assert self._posteriors, "forward(): No post layer exists."

        def _forward(Y: np.ndarray, layer: Layer) -> None:
            """Forward the matmul output Y to a post layer
            Args:
                Y: Matmul output
                layer: Layer where to propagate Y.
            Returns:
                Z: Return value from the post layer.
            """
            layer.forward(Y)

        Y: np.ndarray = self.function(X)
        list(map(_forward, Y, self._posteriors))
        return Y

    def gradient(self, dY: Union[np.ndarray, float] = 1.0) -> Union[np.ndarray, float]:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        """
        assert isinstance(dY, float) or (isinstance(dY, np.ndarray) and dY.dtype == float)

        dY = np.array(dY).reshape((1, -1)) if isinstance(dY, float) or dY.ndim < 2 else dY
        assert dY.shape == (self.N, self.M), \
            "dL/dY shape needs (%s, %s) but %s" % (self.N, self.M, dY.shape)

        self.logger.debug("layer[%s] gradient(): dY.shape %s", self.name, dY.shape)
        self._dY = dY

        # --------------------------------------------------------------------------------
        # dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        # --------------------------------------------------------------------------------
        np.matmul(self.dY, self.W, out=self._dX)
        assert self.dX.shape == (self.N, self.D), \
            "dL/dX shape needs (%s, %s) but %s" % (self.N, self.D, self.dX.shape)

        return self.dX

    def backward(self) -> Union[np.ndarray, float]:
        """Calculate the gradient dL/dX to back-propagate
        """
        assert self._posteriors, "backward() called when no post layer exist."

        def _backward(layer: Layer) -> np.ndarray:
            """Get gradient dL/dY from a post layer
            Args:
                layer: a post layer
            Returns:
                dL/dY: the impact on L by the layer output dY
            """
            # --------------------------------------------------------------------------------
            # Back propagation from the post layer(s)
            # dL/dY has the same shape with Y:shape(N, M) as L and dL are scalar.
            # --------------------------------------------------------------------------------
            dY: np.ndarray = layer.backward()
            assert np.array_equal(dY.shape, (self.N, self.M)), \
                f"dY.shape needs {(self.N, self.M)} but ({dY.shape}))"

            return dY

        # --------------------------------------------------------------------------------
        # Gradient dL/dY, the total impact on L by dY, from post layer(s) if exist.
        # np.add.reduce() is faster than np.sum() as sum() calls it internally.
        # --------------------------------------------------------------------------------
        dY = np.add.reduce(map(_backward, self._posteriors))
        return self.gradient(dY)

    def gradient_numerical(self, h: float = 1e-5) -> List[Union[float, np.ndarray]]:
        """Calculate numerical gradients
        Args:
            h: small number for delta to calculate the numerical gradient
        Returns:
            [dX, dW]: Numerical gradients for X and W
        """
        self.logger.debug("layer[%s] gradient_numerical()", self.name)
        L = self.objective

        def objective_X(X: np.ndarray):
            return L(X @ self.W.T)

        def objective_W(W: np.ndarray):
            return L(self.X @ W.T)

        dX = numerical_jacobian(objective_X, self.X)
        dW = numerical_jacobian(objective_W, self.W)
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
            [self.dX, self.dW]: dL/dS=[dL/ds for s in S]
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
        return [self.dX, self.dW]
