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
from .. common.decorators import generator
from . layer import Layer


logging.basicConfig(level=logging.DEBUG)


class Affine(Layer):
    """Affine (MatMul) Layer class"""
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    @property
    def name(self):
        """Layer ID"""
        return self._id

    @property
    def W(self) -> np.ndarray:
        """Layer weight vectors W"""
        return self._W

    @property
    def u(self) -> int:
        """Number of nodes in the affine layer"""
        return self._u
    num_nodes = u

    @property
    def n(self) -> int:
        """Number of feature of a node in the affine layer"""
        return self._n

    @property
    def m(self) -> int:
        """Batch size"""
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
        """Latest gradient (normalized) from the post layers"""
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
            W: np.ndarray,
            num_nodes: int,
            layers: List[Layer],
            optimizer: Callable[[np.ndarray, np.ndarray], np.ndarray],
            l2: float = 1e-3,
            log_level: int = logging.ERROR
    ):
        """Initialize an affine layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            W: Weight matrix of shape(num_nodes, n), each row of which is a weight vector of a node.
            num_nodes: Number of nodes in the layer
            layers: Layers to which forward the output
            optimizer: Gradient descent implementation e.g SGD, Adam.
            l2: L2 regularization hyper parameter
            log_level: logging level

        Batch X: shape(m, n)
        --------------------
        'x' is an individual row with n features where n=0 is a bias. A batch X has 'm' rows.
        X[j] is [x(j)(0), x(j)(1), ... x(j)(n)] where bias 'x(j)(0)' is 1 as a bias input.
        "input" is not limited to the 1st input data layer e.g. image pixels but to any layer.

        Weights W: shape(u, n) where u=num_nodes
        --------------------
        k-th node (k:0, 1, .. u-1) has a weight vector W(k):[w(k)(0), w(k)(1), ... w(k)(n)].
        w(k)(0) is its bias weight. Each w(k)(i) amplifies i-th feature in the input x.

        """
        # --------------------------------------------------------------------------------
        # Validate the expected dimensions.
        # `W` has `u` nodes (nodes)
        # --------------------------------------------------------------------------------
        super().__init__(name=name)
        self._name = name

        assert W.shape[0] == num_nodes, \
            f"W has {W.shape[0]} weight vectors that should have matched with num_nodes {num_nodes}"

        # --------------------------------------------------------------------------------
        # W: weight vectors of shape(u,n) where u=num_nodes
        # --------------------------------------------------------------------------------
        self._W: np.ndarray = W             # node weight vectors
        self._u: int = num_nodes            # number of nodes in the layer
        self._n: int = W.shape[1]           # number of features in x

        # --------------------------------------------------------------------------------
        # Gradient dL/dW of shape(u, n)
        # Because the loss L is scalar, dL/dW has the same shape with W.
        # --------------------------------------------------------------------------------
        self._dW: np.ndarray = np.empty(num_nodes, W.shape[1])

        # --------------------------------------------------------------------------------
        # X: batch input of shape(m, n)
        # --------------------------------------------------------------------------------
        self._X: np.ndarray = np.empty((0, num_nodes), dtype=float)  # Batch input
        self._m: int = -1                   # batch size: X.shape[0]

        # --------------------------------------------------------------------------------
        # Gradient dL/dX of shape(m,n).
        # Because the loss L is scalar, dL/dX has the same shape with X.
        # --------------------------------------------------------------------------------
        self._dX: np.ndarray

        # --------------------------------------------------------------------------------
        # Affine layer output Y of shape(m, u) as per X:shape(m, n) @ W.T:shape(n, u)
        # --------------------------------------------------------------------------------
        self._Y = np.empty(0, num_nodes)

        # --------------------------------------------------------------------------------
        # Gradient dL/dY of shape(m, u)
        # Because the loss L is scalar, dL/dY has the same shape with Y.
        # Normalize gradients from the post layers as np.sum(dL/dY) / len(layers).
        # --------------------------------------------------------------------------------
        self._dY = np.empty(0, num_nodes)

        # Layers to which forward the affine output
        self._layers: List[Layer] = layers
        self._num_layers = len(layers)

        # Optimizer
        self._optimizer = optimizer
        self._l2 = l2

        logging.basicConfig()
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
            Normalized loss of those returned from the post layers.
        """
        assert X and X.shape[1] > 0 and X.shape[1] == self._n, \
            f"X is expected to have shape (m, {self._n}) but {X.shape}"
        self._X = X
        self._m = X.shape[0]

        # --------------------------------------------------------------------------------
        # Forward propagate the affine output Y = X@W.T to post layers.
        # X:shape(m, n) W.T:shape(n, u) -> Y:shape(m, u)
        # The backward() need to validate the dY shape is (m, u)
        # --------------------------------------------------------------------------------
        np.matmul(X, self._W.T, out=self._Y)
        loss = np.add.reduce(
            map(self._forward, self.Y, self._layers),
            axis=0
        ) / self._num_layers

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
        # dL/dY has the same shape with Y:shape(m, u) as L and dL are scalar.
        # --------------------------------------------------------------------------------
        dY: np.ndarray = layer.backward()
        assert np.array_equal(dY.shape, (self.m, self.u)), \
            f"dY.shape is expected as ({self.m}, {self.u}) but ({dY.shape}))"

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
        # dL/dW has the same shape with W:shape(u, n) because L and dL is scalar.
        # --------------------------------------------------------------------------------
        dW = np.matmul(X.T, dY, out=dW) / X.shape[0]        # Normalize with batch size
        assert np.array_equal(dW.shape, (self.u, self.n)), \
            f"dL/dW shape is expected as ({self.u}, {self.n}) but ({dW.shape}))"

        W = self._optimizer(W, dW + self.dW * self.l2)
        assert np.array_equal(W.shape, (self.u, self.n)), \
            f"Updated W shape is expected as ({self.u}, {self.n}) but ({self._dW.shape}))"

        return W

    def backward(self):
        """Back-propagate the gradient dX to the input layer.
        Returns:
            Gradient dL/dX = W.T @ dL/dY
        """
        # --------------------------------------------------------------------------------
        # Normalized gradient dL/dY from the post layer
        # dL/dY:shape(m, u) as per X:shape(m, n) @ W.T:shape(n, u)
        # --------------------------------------------------------------------------------
        self._dY = np.sum(
            map(self._backward, self._layers),
            axis=0
        ) / self._num_layers

        # --------------------------------------------------------------------------------
        # Gradient update on W
        # --------------------------------------------------------------------------------
        self._W = self._gradient_descent_W(self.dW, self.W, self.X, self.dY)

        # --------------------------------------------------------------------------------
        # Back-propagate dL/dX
        # --------------------------------------------------------------------------------
        self._dX = np.matmul(self.W.T, self.dY) / self.m    # Normalize by batch size
        assert np.array_equal(self._dX, (m, n)), \
            f"Gradient dL/dX shape is expected as ({self.m}, {self.n}) but ({self._dX.shape}))"