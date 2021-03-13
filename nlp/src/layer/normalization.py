"""Normalization layer implementation
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
import inspect
import logging
import copy
import numpy as np
from layer import Layer
from optimizer import (
    Optimizer,
    SGD,
)
from common.functions import (
    standardize,
    numerical_jacobian
)


class Standardization(Layer):
    """Standardization Layer class
    Considerations:
        Need to apply the same mean and std to the non-training data set because the
        model has been trained on the specific mean/sd of the training data set.
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
            posteriors: Optional[List[Layer]] = None,
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            posteriors: Post layers to which forward the matmul layer output
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        # --------------------------------------------------------------------------------
        # Validate the expected dimensions.
        # `W` has `M` nodes (nodes)
        # --------------------------------------------------------------------------------
        self.logger.debug(
            "Standardization[%s] number of nodes is [%s]",
            name, num_nodes
        )

        # Layers to which forward the matmul output
        self._posteriors: List[Layer] = posteriors
        self._num_posteriors: int = len(posteriors) if posteriors else -1

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def X(self) -> np.ndarray:
        """Latest batch input to the layer"""
        return super().X

    @X.setter
    def X(self, X: np.ndarray):
        """Set X"""
        super(Standardization, type(self)).X.fset(self, X)
        # Cannot check. Setting _D can be done using weight shape.
        # assert self.X.shape[1] == self.D, \
        #     "X shape needs (%s, %s) but %s" % (self.N, self.D, self.X.shape)

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Standardize the input X per feature/column basis.
        Args:
            X: Batch input data from the input layer.
        Returns:
            Y: Per-feature standardized output
        """
        name = "function"
        self.X = X
        self.logger.debug("layer[%s].%s: X.shape %s", self.name, name, self.X.shape)

        # --------------------------------------------------------------------------------
        # Allocate array storage for np.func(out=) for Y but not dY.
        # Y:(N,M) = [ X:(N,D) @ W.T:(D,M) ]
        # --------------------------------------------------------------------------------
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty((self.N, self.M), dtype=float)
            # --------------------------------------------------------------------------------
            # DO NOT allocate memory area for the gradient that has already been calculated.
            # dL/dY is calculated at the post layer, hence it has the buffer allocated already.
            # --------------------------------------------------------------------------------
            # self._dY = np.empty((self.N, self.M), dtype=float)

        standardize(X, out=self._Y)
        return self.Y

    def gradient(self, dY: Union[np.ndarray, float] = 1.0) -> Union[np.ndarray, float]:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        """
        name = "gradient"
        assert isinstance(dY, float) or (isinstance(dY, np.ndarray) and dY.dtype == float)

        dY = np.array(dY).reshape((1, -1)) if isinstance(dY, float) or dY.ndim < 2 else dY
        assert dY.shape == self.Y.shape, \
            "dL/dY shape needs %s but %s" % (self.Y.shape, dY.shape)

        self.logger.debug("layer[%s].%s: dY.shape %s", self.name, name, dY.shape)
        self._dY = dY

        # --------------------------------------------------------------------------------
        # dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        # TODO: Implement the Standardization layer gradient calculation.
        # See the batch normalization reference. For now, just return dL/dY
        # --------------------------------------------------------------------------------
        # assert self.dX.shape == (self.N, self.D), \
        #     "dL/dX shape needs (%s, %s) but %s" % (self.N, self.D, self.dX.shape)
        # return self.dX
        # --------------------------------------------------------------------------------
        return self.dY
        # --------------------------------------------------------------------------------
