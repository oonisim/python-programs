"""Affine layer implementation
Note:
    Python passes a pointer to an object storage if it is mutable.
    Matrices can take up lage storage, hence to avoid temporary copies,
    directly update the target storage area.
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
import numpy as np
from . base import Layer
from .. optimizer import (
    Optimizer,
    SGD,
)


class Affine(Layer):
    """Affine (MatMul) Layer class"""
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    @property
    def W(self) -> np.ndarray:
        """Layer weight vectors W"""
        return self._W

    @property
    def M(self) -> int:
        """Number of nodes in the affine layer"""
        return self._M

    @property
    def D(self) -> int:
        """Number of feature of a node in the affine layer"""
        return self._D

    @property
    def N(self) -> int:
        """Batch size of X"""
        return self._m

    @property
    def dW(self) -> np.ndarray:
        """Layer weight gradients dW"""
        return self._dW

    @property
    def X(self) -> np.ndarray:
        """Latest batch input to the layer"""
        return self._X

    @property
    def Y(self) -> np.ndarray:
        """Latest affine layer output"""
        return self._Y

    @property
    def dY(self) -> np.ndarray:
        """Latest gradient (normalized) from the post posteriors"""
        return self._dY

    @property
    def optimizer(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Gradient descent optimizer for the layer"""
        return self._optimizer

    @property
    def l2(self) -> float:
        """L2 regularization hyper parameter"""
        return self._l2

    def __init__(
            self,
            name: str,
            num_nodes: int,
            W: np.ndarray,
            posteriors: List[Layer],
            l2: float = 0,
            optimizer: Optimizer = SGD(),
            log_level: int = logging.ERROR
    ):
        """Initialize an affine layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            W: Weight matrix of shape(M=num_nodes, D), each row of which is a weight vector of a node.
            posteriors: Post layers to which forward the affine layer output
            l2: L2 regularization hyper parameter, e.g. 1e-3, set to 0 by default not to use it.
            optimizer: Gradient descent implementation e.g SGD, Adam.
            log_level: logging level

        Batch X: shape(N, D)
        --------------------
        A batch X has N rows of x, that has D features and x[d=0] has a bias value 1.
        X[j] = [x(j)(0), x(j)(1), ... x(j)(D)]
        "input" is not limited to the 0th input layer e.g. image pixels but to any layer.

        Weights W: shape(M, D) where M=num_nodes
        --------------------
        Node k (k:0, 1, .. M-1) has a weight vector W(k):[w(k)(0), w(k)(1), ... w(k)(D)].
        w(k)(0) is a bias weight. Each w(k)(i) amplifies i-th feature in an input x.
        """
        # --------------------------------------------------------------------------------
        # Validate the expected dimensions.
        # `W` has `M` nodes (nodes)
        # --------------------------------------------------------------------------------
        super().__init__(name=name)
        assert W.shape[0] == num_nodes, \
            f"W has {W.shape[0]} weight vectors that should have matched num_nodes {num_nodes}"

        # --------------------------------------------------------------------------------
        # W: weight matrix of shape(M,D) where M=num_nodes
        # --------------------------------------------------------------------------------
        self._W: np.ndarray = W             # node weight vectors
        self._M: int = num_nodes            # number of nodes in the layer
        self._D: int = W.shape[1]           # number of features in x

        # --------------------------------------------------------------------------------
        # Gradient dL/dW of shape(M, D)
        # Because the loss L is scalar, dL/dW has the same shape with W.
        # --------------------------------------------------------------------------------
        self._dW: np.ndarray = np.empty(num_nodes, W.shape[1])

        # --------------------------------------------------------------------------------
        # X: batch input of shape(N, D)
        # --------------------------------------------------------------------------------
        self._X: np.ndarray = np.empty((0, num_nodes), dtype=float)  # Batch input
        self._m: int = -1                   # batch size: X.shape[0]

        # --------------------------------------------------------------------------------
        # Gradient dL/dX of shape(N,D).
        # Because the loss L is scalar, dL/dX has the same shape with X.
        # --------------------------------------------------------------------------------
        self._dX: np.ndarray

        # --------------------------------------------------------------------------------
        # Affine layer output Y of shape(N, M) as per X:shape(N, D) @ W.T:shape(D, M)
        # --------------------------------------------------------------------------------
        self._Y = np.empty(0, num_nodes)

        # --------------------------------------------------------------------------------
        # Gradient dL/dY of shape(N, M)
        # Because the loss L is scalar, dL/dY has the same shape with Y.
        # Normalize gradients from the post posteriors as np.sum(dL/dY) / len(posteriors).
        # --------------------------------------------------------------------------------
        self._dY = np.empty(0, num_nodes)

        # Layers to which forward the affine output
        self._layers: List[Layer] = posteriors
        self._num_posteriors = len(posteriors)

        # Optimizer
        self._optimizer = optimizer
        self._l2 = l2

        self._log_level = log_level
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)


    def _forward(self, Y: np.ndarray, layer: Layer) -> float:
        """Forward the affine output Y to a next layer
        Args:
            Y: Affine output
            layer: Layer where to propagate Y.
        Returns:
            Layer return
        """
        loss: float = layer.forward(Y)
        return loss

    
    def forward(self, X: np.ndarray):
        """Forward propagate the affine layer output Y = X@W
        Args:
            X: Batch input data from the input layer.
        Returns:
            Normalized loss of those returned from the post posteriors.
        """
        assert X and X.shape[1] > 0 and X.shape[1] == self._D, \
            f"X is expected to have shape (N, {self._D}) but {X.shape}"
        self._X = X
        self._m = X.shape[0]

        # --------------------------------------------------------------------------------
        # Forward propagate the affine output Y = X@W.T to post posteriors.
        # X:shape(N, D) W.T:shape(D, M) -> Y:shape(N, M)
        # The backward() need to validate the dY shape is (N, M)
        # --------------------------------------------------------------------------------
        np.matmul(X, self._W.T, out=self._Y)
        loss = np.add.reduce(
            map(self._forward, self.Y, self._layers),
            axis=0
        ) / self._num_posteriors

        return loss


    def _backward(self, layer: Layer) -> np.ndarray:
        """Get a back-propagation gradient from a post layer
        Args:
            layer: a post layer
        Returns:
            The gradient from the post layer
        """
        # --------------------------------------------------------------------------------
        # Get a gradient dL/dY of  from a post layer
        # dL/dY has the same shape with Y:shape(N, M) as L and dL are scalar.
        # --------------------------------------------------------------------------------
        dY: np.ndarray = layer.backward()
        assert np.array_equal(dY.shape, (self.N, self.M)), \
            f"dY.shape is expected as ({self.N}, {self.M}) but ({dY.shape}))"

        return dY


    def _gradient_descent_W(self, dW: np.ndarray, W: np.ndarray, X: np.ndarray, dY: np.ndarray):
        """Gradient descent e.g. SGD: W = W - lr * dW (1 + l2)
        Directly updating matrices to avoid the temporary copies

        Args:
            dY: Gradient dL/dY from the post layer
        Returns:
            W
        """
        # --------------------------------------------------------------------------------
        # Gradient on W: dL/dW = dY/dW @ dL/dY = X.T @ dL/dY because dY/dW = X.T
        # dL/dW has the same shape with W:shape(M, D) because L and dL is scalar.
        # --------------------------------------------------------------------------------
        dW = np.matmul(X.T, dY, out=dW) / X.shape[0]        # Normalize with batch size
        assert np.array_equal(dW.shape, (self.M, self.D)), \
            f"dL/dW shape is expected as ({self.M}, {self.D}) but ({dW.shape}))"

        W = self._optimizer(W, dW + self.dW * self.l2)
        assert np.array_equal(W.shape, (self.M, self.D)), \
            f"Updated W shape is expected as ({self.M}, {self.D}) but ({self._dW.shape}))"

        return W

    def backward(self):
        """Back-propagate the gradient dX to the input layer.
        Returns:
            Gradient dL/dX = W.T @ dL/dY
        """
        # --------------------------------------------------------------------------------
        # Normalized gradient dL/dY from the post layer
        # dL/dY:shape(N, M) as per X:shape(N, D) @ W.T:shape(D, M)
        # --------------------------------------------------------------------------------
        self._dY = np.sum(
            map(self._backward, self._layers),
            axis=0
        ) / self._num_posteriors

        # --------------------------------------------------------------------------------
        # Gradient update on W
        # --------------------------------------------------------------------------------
        self._W = self._gradient_descent_W(self.dW, self.W, self.X, self.dY)

        # --------------------------------------------------------------------------------
        # Back-propagate dL/dX
        # --------------------------------------------------------------------------------
        self._dX = np.matmul(self.W.T, self.dY) / self.N    # Normalize by batch size
        assert np.array_equal(self._dX, (self.M, self.D)), \
            f"Gradient dL/dX shape is expected as ({self.N}, {self.D}) but ({self._dX.shape}))"