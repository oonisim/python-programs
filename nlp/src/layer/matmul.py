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
    numerical_gradient
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
            posteriors: Optional[List[Layer]],
            optimizer: Optimizer = SGD(),
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            W: Weight matrix of shape(M=num_nodes, D), each row of which is a weight vector of a node.
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
            f"W shape needs to be (N, {num_nodes}) but (M, {W.shape[0]})."

        # --------------------------------------------------------------------------------
        # W: weight matrix of shape(M,D) where M=num_nodes
        # Gradient dL/dW has the same shape shape(M, D) with W because L is scalar.
        # --------------------------------------------------------------------------------
        self._W: np.ndarray = W             # node weight vectors
        self._M: int = num_nodes            # number of nodes in the layer
        self._D: int = W.shape[1]           # number of features in x
        self._dW: np.ndarray = np.empty((num_nodes, W.shape[1]), dtype=float)

        # --------------------------------------------------------------------------------
        # X: batch input of shape(N, D)
        # Gradient dL/dX of has the same shape(N,D) with X because L is scalar.
        # --------------------------------------------------------------------------------
        self._X: np.ndarray = np.empty((0, num_nodes), dtype=float)
        self._dX: np.ndarray = np.empty((0, num_nodes), dtype=float)

        # --------------------------------------------------------------------------------
        # Matmul layer output Y of shape(N, M) as per X:shape(N, D) @ W.T:shape(D, M)
        # Gradient dL/dY has the same shape (N, M) with Y because the L is scalar.
        # dL/dY is the sum of all the impact on L by dY.
        # --------------------------------------------------------------------------------
        self._Y: np.ndarray = np.empty((0, num_nodes), dtype=float)
        self._dY: np.ndarray = np.empty((0, num_nodes), dtype=float)

        # Layers to which forward the matmul output
        self._posteriors: List[Layer] = posteriors
        self._num_posteriors: int = len(posteriors)

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
        assert self._dW.size, "dW is not initialized"
        return self._dW

    @property
    def X(self) -> np.ndarray:
        """Latest batch input to the layer"""
        assert self._X and self._X.size, "X is not initialized"
        return self._X

    @X.setter
    def X(self, X: np.ndarray):
        """Set X"""
        assert X and X.shape[0] > 0 and X.shape[1] == self.D, \
            f"X shape needs (N, {self.D}) but ({X.shape})"
        self._X = X
        self._N = X.shape[0]

        # Allocate the storage for np.func(out=dX).
        self._dX = np.empty(X.shape, dtype=float) \
            if self.dX.shape[0] != X.shape[0] else self.dX

    @property
    def dX(self) -> np.ndarray:
        """Gradient dL/dX"""
        assert self._dX and self._dX.size, "dX is not initialized"
        return self._dX

    @property
    def Y(self) -> np.ndarray:
        """Latest matmul layer output"""
        assert self._Y and self._Y.size, "Y is not initialized"
        return self._Y

    @property
    def dY(self) -> np.ndarray:
        """Latest gradient dL/dY (impact on L by dY) given from the post layer(s)"""
        assert self._dY and self._dY.size, "dY is not initialized"
        return self._dY

    @property
    def optimizer(self) -> Optimizer:
        """Optimizer instance for gradient descent
        """
        return self._optimizer

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: np.ndarray) -> np.ndarray:
        """Calculate the layer output Y = X@W.T
        Args:
            X: Batch input data from the input layer.
        Returns:
            Y: Layer value of X@W.T
        """
        self.X = X
        assert self.W and np.array_equal(self.W.shape, (self.M, self.D)), \
            f"W shape needs {(self.M, self.D)} but ({self.W.shape})"

        # --------------------------------------------------------------------------------
        # Allocate array for np.func(out=) for Y but not dY.
        # Y:(N,M) = [ X:(N,D) @ W.T:(D,M) ]
        # gradient() need to validate the dY shape is (N,M)
        # --------------------------------------------------------------------------------
        if self.Y.shape[0] != self.N:
            self._Y = np.empty((self.N, self.M), dtype=float)
            # --------------------------------------------------------------------------------
            # DO NOT allocate memory area for the gradient that has already been calculated.
            # dL/dY is calculated at the post layer, hence it has the buffer allocated already.
            # --------------------------------------------------------------------------------
            # self._dY = np.empty((self.N, self.M), dtype=float)

        np.matmul(X, self.W.T, out=self._Y)
        return self.Y

    def forward(self, X: np.ndarray) -> np.ndarray:
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

    def gradient(self, dY=1) -> np.ndarray:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        """
        assert dY == 1 or np.array_equal(self.dY.shape, (self.N, self.M)), \
            f"Gradient dL/dY shape needs {(self.N, self.M)} but ({self.dY.shape}))"
        self._dY = dY

        # --------------------------------------------------------------------------------
        # dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        # --------------------------------------------------------------------------------
        np.matmul(self.dY, self.W, out=self._dX)
        assert np.array_equal(self.dX.shape, (self.N, self.D)), \
            f"Gradient dL/dX shape needs {(self.N, self.D)} but ({self.dX.shape}))"

        return self.dX

    def backward(self) -> np.ndarray:
        """Calculate and back-propagate the gradient dL/dX
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

    def _gradient_descent(self) -> np.ndarray:
        """Gradient descent
        Directly update matrices to avoid the temporary copies
        """
        if self.logger.level == logging.DEBUG:
            # np.copy() is a shallow copy and will not copy object elements within arrays.
            # To ensure all elements within an object array are copied, use copy.deepcopy().
            backup = copy.deepcopy(self.W)
            W = self.optimizer.update(self._W, self.dW, out=self._W)
            assert not np.array_equal(backup, self.W), \
                "W is not updated by the gradient descent"
        else:
            W = self.optimizer.update(self._W, self.dW, out=self._W)

        assert W == self._W
        assert np.array_equal(self.W.shape, (self.M, self.D)), \
            f"Updated W shape needs {(self.M, self.D)} but ({self.W.shape}))"

        return W

    def update(self, dY) -> Union[np.ndarray, List[np.ndarray]]:
        """Calculate dL/dW = dL/dY * dY/dW
        dL/dW.T = X.T @ dL/dY is shape (D,M) as  [ X.T:(D, N)  @ dL/dY:(N,M) ].
        Hence dL/dW of shape (M,D):  [ X.T:(D, N)  @ dL/dY:(N,M) ].T.

        Args:
            dY: dL/dY, impact on L by the layer output dY.
        Returns:
            dL/dW: Impact on L by dW.
        """
        assert dY == 1 or np.array_equal(self.dY.shape, (self.N, self.M)), \
            f"Gradient dL/dY shape needs {(self.N, self.M)} but ({self.dY.shape}))"
        self._dY = dY

        # --------------------------------------------------------------------------------
        # dL/dW of shape (M,D):  [ X.T:(D, N)  @ dL/dY:(N,M) ].T
        # --------------------------------------------------------------------------------
        dW = np.matmul(self.X.T, dY).T
        assert np.array_equal(dW.shape, (self.M, self.D)), \
            f"Gradient dL/dW shape needs {(self.M, self.D)} but ({dW.shape}))"

        self._dW = dW
        self._gradient_descent()
        return self.dW

    def gradient_numerical(
            self, L: Callable[[np.ndarray], np.ndarray], h: float = 1e-05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate numerical gradients
        Args:
            L: Loss function for the layer. loss=L(f(X)), NOT L for NN.
            h: small number for delta to calculate the numerical gradient
        Returns:
            (dX, dW): Numerical gradients for X and W
        """
        def loss(W: np.ndarray): return L(self.X @ W.T)
        dW = numerical_gradient(loss, self.W)
        def loss(X: np.ndarray): return L(X @ self.W.T)
        dX = numerical_gradient(loss, self.X)
        return dX, dW
