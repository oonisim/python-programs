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
import numexpr as ne
from layer import Layer
from optimizer import (
    Optimizer,
    SGD,
)
from common import (
    TYPE_FLOAT,
    TYPE_LABEL,
    standardize,
    numerical_jacobian
)


class Standardization(Layer):
    """Standardization Layer class
    Considerations:
        Need to apply the same mean and standard deviation (SD) to the non-training
        data set because the model has been trained on the specific mean/SD of the
        training data set.
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
            self._Y = np.empty((self.N, self.M), dtype=TYPE_FLOAT)
            # --------------------------------------------------------------------------------
            # DO NOT allocate memory area for the gradient that has already been calculated.
            # dL/dY is calculated at the post layer, hence it has the buffer allocated already.
            # --------------------------------------------------------------------------------
            # self._dY = np.empty((self.N, self.M), dtype=TYPE_FLOAT)

        _, mean, sd = standardize(X, out=self._Y)
        return self.Y

    def gradient(self, dY: Union[np.ndarray, float] = 1.0) -> Union[np.ndarray, float]:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
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
        # TODO: Implement the Standardization layer gradient calculation.
        # See the batch normalization reference. For now, just return dL/dY
        # --------------------------------------------------------------------------------
        # assert self.dX.shape == (self.N, self.D), \
        #     "dL/dX shape needs (%s, %s) but %s" % (self.N, self.D, self.dX.shape)
        # return self.dX
        # --------------------------------------------------------------------------------
        return self.dY
        # --------------------------------------------------------------------------------


class BatchNormalization(Layer):
    """BatchNormalization Layer class
    Considerations:
        Need to apply the population mean and SD to the non-training data set.
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
            momentum: float = 0.9,
            posteriors: Optional[List[Layer]] = None,
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            momentum: Decay of running mean/variance
            posteriors: Post layers to which forward the matmul layer output
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        self.logger.debug(
            "BatchNormalization[%s] number of nodes is [%s]",
            name, num_nodes
        )

        # Layers to which forward the output
        self._posteriors: List[Layer] = posteriors
        self._num_posteriors: int = len(posteriors) if posteriors else -1

        # --------------------------------------------------------------------------------
        # Standardized
        # --------------------------------------------------------------------------------
        self._Xstd = np.empty(0, dtype=TYPE_FLOAT)
        # Feature mean of shape(1,M)
        self._U: np.ndarray = np.empty((1, num_nodes), dtype=TYPE_FLOAT)
        # SD-per-feature of batch X in shape (1,M)
        self._SD: np.ndarray = np.empty((1, num_nodes), dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Statistics. allocate storage at the instance initialization.
        # --------------------------------------------------------------------------------
        # Running mean-per-feature of all the batches X* in shape(1,M)
        self._RU: np.ndarray = np.zeros((1, num_nodes), dtype=TYPE_FLOAT)
        # Running SD-per-feature of all the batches X* in shape (1,M)
        self._RSD: np.ndarray = np.zeros((1, num_nodes), dtype=TYPE_FLOAT)
        # Scale and shift parameters
        self._gamma: np.ndarray = np.ones(num_nodes)        # Scale
        self._beta: np.ndarray = np.zeros(num_nodes)        # Shift


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
        super(BatchNormalization, type(self)).X.fset(self, X)
        # Cannot check. Setting _D can be done using weight shape.
        # assert self.X.shape[1] == self.D, \
        #     "X shape needs (%s, %s) but %s" % (self.N, self.D, self.X.shape)

    @property
    def Xstd(self) -> np.ndarray:
        """Standardized X"""
        return self._Xstd

    @property
    def U(self) -> np.ndarray:
        """Mean of each feature in X"""
        assert self._U.size > 0 and self._U.shape == (self.M,), \
            "U is not initialized or invalid"
        return self._U

    @property
    def RU(self) -> np.ndarray:
        """Running mean of each feature in X"""
        assert self._RU.size > 0 and self._RU.shape == (self.M,), \
            "RU is not initialized or invalid"
        return self._RU

    @property
    def SD(self) -> np.ndarray:
        """Standard Deviation (SD) of each feature in X"""
        assert self._SD.size > 0 and self._SD.shape == (self.M,), \
            "SD is not initialized or invalid"
        return self._SD

    @property
    def RSD(self) -> np.ndarray:
        """Running SD of each feature in X"""
        assert self._RSD.size > 0 and self._RSD.shape == (self.M,), \
            "RSD is not initialized or invalid"
        return self._RSD

    @property
    def gamma(self) -> np.ndarray:
        """Feature scale parameter gamma for each feature in X"""
        assert self._gamma.size > 0 and self._gamma.shape == (self.M,), \
            "gamma is not initialized or invalid"
        return self._gamma

    @property
    def beta(self) -> np.ndarray:
        """Bias shift parameter beta for each feature in X"""
        assert self._beta.size > 0 and self._beta.shape == (self.M,), \
            "beta is not initialized or invalid"
        return self._beta

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
        assert self.X.shape[1] == self.M, \
            "Number of features %s must match number of nodes %s in the BN layer." \
            % (X.shape[1], self.M)

        # --------------------------------------------------------------------------------
        # Allocate array storage for np.func(out=) for Y but not dY.
        # Y:(N,M) = [ X:(N,D) @ W.T:(D,M) ]
        # --------------------------------------------------------------------------------
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty((self.N, self.M), dtype=TYPE_FLOAT)
            # --------------------------------------------------------------------------------
            # DO NOT allocate memory area for the gradient that has already been calculated.
            # dL/dY is calculated at the post layer, hence it has the buffer allocated already.
            # --------------------------------------------------------------------------------
            # self._dY = np.empty((self.N, self.M), dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Allocate array storages standardization.
        # --------------------------------------------------------------------------------
        if self._Xstd.size <= 0 or self._Xstd.shape[0] != self.N:
            self._Xstd = np.empty(X.shape, dtype=TYPE_FLOAT)

        _, self._U, self._SD = standardize(
            X,
            keepdims=True,
            out=self._Xstd,
            out_mean=self._U,
            out_sd=self._SD
        )
        gamma = self.gamma
        beta = self.beta
        standardized = self.Xstd
        ne.evaluate("gamma * standardized + beta", out=self._Y)

        return self.Y

    def gradient(self, dY: Union[np.ndarray, float] = 1.0) -> Union[np.ndarray, float]:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
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
        # TODO: Implement the BatchNormalization layer gradient calculation.
        # See the batch normalization reference. For now, just return dL/dY
        # --------------------------------------------------------------------------------
        # assert self.dX.shape == (self.N, self.D), \
        #     "dL/dX shape needs (%s, %s) but %s" % (self.N, self.D, self.dX.shape)
        # return self.dX
        # --------------------------------------------------------------------------------
        return self.dY
        # --------------------------------------------------------------------------------
