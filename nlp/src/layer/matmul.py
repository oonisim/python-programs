"""Matmul layer implementation
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
import copy
import numpy as np
from . base import Layer
from .. optimizer import (
    Optimizer,
    SGD,
)


class Matmul(Layer):
    """MatMul Layer class"""
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
        """Number of nodes in the matmul layer"""
        return self._M

    @property
    def D(self) -> int:
        """Number of feature of a node in the matmul layer"""
        return self._D

    @property
    def dW(self) -> np.ndarray:
        """Layer weight gradients dW"""
        return self._dW

    @property
    def X(self) -> np.ndarray:
        """Latest batch input to the layer"""
        return self._X

    @property
    def N(self) -> int:
        """Batch size of X"""
        return self._N

    @property
    def Y(self) -> np.ndarray:
        """Latest matmul layer output"""
        return self._Y

    @property
    def dY(self) -> np.ndarray:
        """Latest gradient dL/dY (impact on L by dY) given from the post layer(s)"""
        return self._dY

    @property
    def optimizer(self) -> Optimizer:
        """Optimizer instance for gradient descent
        """
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
            posteriors: Optional[List[Layer]],
            l2: float = 0,
            optimizer: Optimizer = SGD(),
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            W: Weight matrix of shape(M=num_nodes, D), each row of which is a weight vector of a node.
            posteriors: Post layers to which forward the matmul layer output
            l2: L2 regularization hyper parameter, e.g. 1e-3, set to 0 by default not to use it.
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
        super().__init__(name=name)
        self._log_level = log_level
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        # --------------------------------------------------------------------------------
        # Validate the expected dimensions.
        # `W` has `M` nodes (nodes)
        # --------------------------------------------------------------------------------
        self._logger.debug(
            "Matmul name [%s] W.shape is [%s], numer of nodes is [%s]",
            name, W.shape, num_nodes
        )
        assert W.shape[0] == num_nodes, \
            f"W has {W.shape[0]} number of weight vectors that should have matched num_nodes {num_nodes}"

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
        self._dW: np.ndarray = np.empty(num_nodes, W.shape[1], dtype=float)

        # --------------------------------------------------------------------------------
        # X: batch input of shape(N, D)
        # Gradient dL/dX of has the same shape(N,D) with X because L is scalar.
        # --------------------------------------------------------------------------------
        self._X: np.ndarray = np.empty((0, num_nodes), dtype=float)
        self._N: int = -1                   # batch size: X.shape[0]
        self._dX: np.ndarray = np.empty((0, num_nodes), dtype=float)

        # --------------------------------------------------------------------------------
        # Matmul layer output Y of shape(N, M) as per X:shape(N, D) @ W.T:shape(D, M)
        # Gradient dL/dY has the same shape (N, M) with Y because the L is scalar.
        # dL/dY is the sum of all the impact on L by dY.
        # --------------------------------------------------------------------------------
        self._Y: np.ndarray = np.empty(0, num_nodes)
        self._dY: np.ndarray = None

        # Layers to which forward the matmul output
        self._layers: List[Layer] = posteriors
        self._num_posteriors: int = len(posteriors)

        # --------------------------------------------------------------------------------
        # Optimizer for gradient descent
        # Z(n+1) = optimizer.update((Z(n), dL/dZ(n)+regularization)
        # --------------------------------------------------------------------------------
        assert isinstance(optimizer, Optimizer)
        self._optimizer: Optimizer = optimizer
        self._l2: float = l2

    def forward(self, X: np.ndarray):
        """Calculate the layer output Y = X@W.T, and forward it to posteriors if set.
        Args:
            X: Batch input data from the input layer.
        Returns:
            Y: Layer value of X@W.T
        """
        assert X and X.shape[0] > 0 and X.shape[1] == self._D, \
            f"X is expected to have shape (N, {self._D}) but {X.shape}"

        self._X = X
        self._N = X.shape[0]
        self._dX = np.empty(self.N, self.D) if self.dX.shape[0] != self.N else self.dX

        # --------------------------------------------------------------------------------
        # Y:shape(N,M) = [ X:shape(N,D) @ W.T:shape(D,M) ]
        # backward() need to validate the dY shape is (N, M)
        # --------------------------------------------------------------------------------
        if self.Y.shape[0] != self.N:
            self._Y = np.empty(self.N, self.M)
            # --------------------------------------------------------------------------------
            # DO NOT allocate memory area for the gradient that has already been calculated.
            # dL/dY is calculated at the post layer, hence it has the buffer allocated already.
            # --------------------------------------------------------------------------------
            # self._dY = np.empty(self.N, self.M)

        np.matmul(X, self._W.T, out=self._Y)

        # --------------------------------------------------------------------------------
        # Forward propagate the matmul output Y to post posteriors if they are set.
        # --------------------------------------------------------------------------------
        def _forward(self, Y: np.ndarray, layer: Layer) -> float:
            """Forward the matmul output Y to a next layer
            Args:
                Y: Matmul output
                layer: Layer where to propagate Y.
            Returns:
                Z: Return value from the post layer.
            """
            Z: float = layer.forward(Y)
            return Z

        if self._layers:
            list(map(_forward, self.Y, self._layers))

        return self.Y

    def _backward(self, layer: Layer) -> np.ndarray:
        """Get gradient dL/dY from a post layer
        Args:
            layer: a post layer
        Returns:
            dL/dY, the impact on L by dY (delta of the layer output Y)
        """
        # --------------------------------------------------------------------------------
        # Get a gradient dL/dY from a post layer
        # dL/dY has the same shape with Y:shape(N, M) as L and dL are scalar.
        # --------------------------------------------------------------------------------
        dY: np.ndarray = layer.backward()
        assert np.array_equal(dY.shape, (self.N, self.M)), \
            f"dY.shape is expected as ({self.N}, {self.M}) but ({dY.shape}))"

        return dY

    def backward(self, dY=1) -> np.ndarray:
        """Calculate the gradients dL/dX and dL/dW.T.
        A layer does not know how L=f(Y) is calculated with its output Y.
        f'(Y) = dL/dY is given from the back (posterior) layer, hence "back" propagation.

        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dL/dX of shape (N,D):  [ dL/dY (N,M) @ W (M,D)) ]
        """
        # --------------------------------------------------------------------------------
        # Gradient dL/dY, the total impact on L by dY, from posteriors if it is set.
        # --------------------------------------------------------------------------------
        if self._layers:
            self._dY = np.sum(list(map(self._backward, self._layers)), axis=0)
        else:
            self._dY = dY

        assert np.array_equal(self.dY.shape, (self.N, self.M)), \
            f"Gradient dL/dY shape is expected as ({self.N}, {self.M}) but ({self.dY.shape}))"

        # --------------------------------------------------------------------------------
        # dL/dW of shape (M,D):  [ X.T (D, N)  @ dL/dY (N,M) ].T
        # --------------------------------------------------------------------------------
        self._dW = np.matmul(self.X.T, dY).T
        assert np.array_equal(self.dW.shape, (self.M, self.D)), \
            f"Gradient dL/dW shape is expected as ({self.M}, {self.D}) but ({self.dW.shape}))"

        # --------------------------------------------------------------------------------
        # dL/dX of shape (N,D):  [ dL/dY (N,M) @ W (M,D)) ]
        # --------------------------------------------------------------------------------
        np.matmul(self.dY, self.W, out=self._dX)
        assert np.array_equal(self.dX.shape, (self.N, self.D)), \
            f"Gradient dL/dX shape is expected as ({self.N}, {self.D}) but ({self.dX.shape}))"

        return self.dX

    def _gradient_descent(self) -> None:
        """Apply gradient descent
        Directly update matrices to avoid the temporary copies
        """
        regularization = self.dW * self.l2
        if self._logger.level == logging.DEBUG:
            # np.copy is a shallow copy and will not copy object elements within arrays.
            # To ensure all elements within an object array are copied, use copy.deepcopy.
            backup = copy.deepcopy(self.W)
            self.optimizer.update(self._W, self.dW + regularization)
            assert not np.array_equal(backup, self.W), \
                "W is not updated by the gradient descent"
        else:
            self.optimizer.update(self._W, self.dW + regularization)

        assert np.array_equal(self.W.shape, (self.M, self.D)), \
            f"Updated W shape is expected as ({self.M}, {self.D}) but ({self.W.shape}))"

    def update(self):
        self._gradient_descent()



