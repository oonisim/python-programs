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

        _, mean, sd, _ = standardize(X, out=self._Y)
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
            optimizer: Optimizer = SGD(),
            posteriors: Optional[List[Layer]] = None,
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            momentum: Decay of running mean/variance
            optimizer: Gradient descent implementation e.g SGD, Adam.
            posteriors: Post layers to which forward the matmul layer output
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        self.logger.debug(
            "BatchNormalization[%s] number of nodes is [%s] momentum is %s]",
            name, num_nodes, momentum
        )

        assert 0 < momentum < 1
        assert isinstance(optimizer, Optimizer)
        self._optimizer: Optimizer = optimizer

        # Layers to which forward the output
        self._posteriors: List[Layer] = posteriors
        self._num_posteriors: int = len(posteriors) if posteriors else -1

        # --------------------------------------------------------------------------------
        # Standardized
        # --------------------------------------------------------------------------------
        # Per-feature standardized X
        self._Xstd = np.empty(0, dtype=TYPE_FLOAT)
        self._dXstd = np.empty(0, dtype=TYPE_FLOAT)
        # Per-feature mean deviation (X-mean)
        self._Xmd = np.empty(0, dtype=TYPE_FLOAT)
        self._dXmd01 = np.empty(0, dtype=TYPE_FLOAT)
        self._dXmd02 = np.empty(0, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Statistics. allocate storage at the instance initialization.
        # --------------------------------------------------------------------------------
        # Feature mean of shape(M)
        self._U: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        self._dU: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        # SD-per-feature of batch X in shape (M,)
        self._SD: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        # Norm = 1/SD in shape (1,M)
        self._norm: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        # dL/dV. Gradient of variance V in shape (M,)
        self._dV: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)

        self._count = 0
        self._momentum: TYPE_FLOAT = momentum
        # Running mean-per-feature of all the batches X* in shape(M,)
        self._RU: np.ndarray = np.zeros(num_nodes, dtype=TYPE_FLOAT)
        # Running SD-per-feature of all the batches X* in shape (M,)
        self._RSD: np.ndarray = np.zeros(num_nodes, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Scale and shift parameters
        # --------------------------------------------------------------------------------
        self._gamma: np.ndarray = np.ones(num_nodes, dtype=TYPE_FLOAT)
        self._dGamma: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        self._beta: np.ndarray = np.zeros(num_nodes, dtype=TYPE_FLOAT)
        self._dBeta: np.ndarray = np.zeros(num_nodes, dtype=TYPE_FLOAT)

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
        assert self._Xstd.size > 0 and self._Xstd.shape == (self.N, self.M)
        return self._Xstd

    def dXstd(self) -> np.ndarray:
        """Gradient of standardized X"""
        assert self._dXstd.size > 0 and self._dXstd.shape == (self.N, self.M)
        return self._dXstd

    @property
    def Xmd(self) -> np.ndarray:
        """Mean deviation (X-mean)"""
        assert self._Xmd.size > 0 and self._Xmd.shape == (self.N, self.M)
        return self._Xmd

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
    def dV(self) -> np.ndarray:
        """dL/dV (impact on L by variance dV) of each feature"""
        assert self._dV.size > 0 and self._dV.shape == (self.M,), \
            "dV is not initialized or invalid"
        return self._dV

    @property
    def SD(self) -> np.ndarray:
        """Standard Deviation (SD) of each feature in X"""
        assert self._SD.size > 0 and self._SD.shape == (self.M,), \
            "SD is not initialized or invalid"
        return self._SD

    @property
    def norm(self) -> np.ndarray:
        """1/SD of each feature in X"""
        assert self._norm.size > 0 and self._norm.shape == (self.M,), \
            "norm is not initialized or invalid"
        return self._norm

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
    def dGamma(self) -> np.ndarray:
        """Gradient of gamma for each feature in X"""
        assert self._dGamma.size > 0 and self._dGamma.shape == (self.M,)
        return self._dGamma

    @property
    def beta(self) -> np.ndarray:
        """Bias shift parameter beta for each feature in X"""
        assert self._beta.size > 0 and self._beta.shape == (self.M,),\
            "beta is not initialized or invalid"
        return self._beta

    @property
    def dBeta(self) -> np.ndarray:
        """Gradient of beta for each feature in X"""
        assert self._dBeta.size > 0 and self._dBeta.shape == (self.M,), \
            "dBeta is not initialized or invalid"
        return self._dBeta

    @property
    def momentum(self) -> TYPE_FLOAT:
        """Decay parameter for calculating running means"""
        return self._momentum

    @property
    def count(self) -> TYPE_FLOAT:
        """Number of times mini-batch trainings have been done."""
        return self._count

    @property
    def optimizer(self) -> Optimizer:
        """Optimizer instance for gradient descent
        """
        return self._optimizer

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def update_running_means(self):
        """Update running means using decaying.
            """
        self._RU = self.momentum * self.RU + (1 - self.momentum) * self.U
        self._RSD = self.momentum * self.RSD + (1 - self.momentum) * self.SD

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
        # Allocate array storage for Y but not dY.
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
            self._dXstd = np.empty(X.shape, dtype=TYPE_FLOAT)
            self._dXmd01 = np.empty(X.shape, dtype=TYPE_FLOAT)
            self._dXmd02 = np.empty(X.shape, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Standardize X and update running means.
        # --------------------------------------------------------------------------------
        _, self._U, self._SD, self._Xmd = standardize(
            X,
            keepdims=False,
            out=self._Xstd,
            out_mean=self._U,
            out_sd=self._SD
        )
        self.update_running_means()
        self._norm = 1.0 / self.SD
        assert np.isfinite(self.norm)

        # --------------------------------------------------------------------------------
        # Calculate layer output Y
        # --------------------------------------------------------------------------------
        gamma = self.gamma
        beta = self.beta
        standardized = self.Xstd
        ne.evaluate("gamma * standardized + beta", out=self._Y)
        self._count += 1

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

        N = self.N
        ddof = 1 if N > 1 else 0
        Xstd = self.Xstd
        dXstd = self.dXstd
        Xmd = self.Xmd
        dXmd01 = self._dXmd01
        dXmd02 = self._dXmd02
        dV = self.dV
        dU = self._dU
        norm = self.norm
        gamma = self.gamma
        dGamma = self.dGamma
        dBeta = self.dBeta

        # --------------------------------------------------------------------------------
        # dL/dGamma:(M,) = sum(dL/dY:(N,M) * Xstd(N,M), axis=0)
        # --------------------------------------------------------------------------------
        ne.evaluate("sum(dY * Xstd, axis=0)", out=dGamma)

        # --------------------------------------------------------------------------------
        # dL/dBeta:(M,) = sum(dL/dY:(N,M) * 1, axis=0)
        # --------------------------------------------------------------------------------
        ne.evaluate("sum(dY, axis=0)", out=dBeta)

        # --------------------------------------------------------------------------------
        # dL/dXstd: (N,M): dL/dY:(N,M) * gamma(M,)
        # --------------------------------------------------------------------------------
        ne.evaluate("sum(dY * gamma, axis=0)", out=dXstd)

        # --------------------------------------------------------------------------------
        # dL/dV:(M,) = sum(dL/dXstd:(N,M) * Xmd:(N,M), axis=0) * -1/2 * norm**3
        # --------------------------------------------------------------------------------
        ne.evaluate("sum(dXstd * Xmd, axis=0)", out=dV)
        np.eavluate("dV * (norm ** 3) / -2", out=dV)

        # --------------------------------------------------------------------------------
        # dL/dXmd01:(N,M) = dL/dV:(M,) / (N-ddof) * 2 * Xmd:(N,M)
        # See https://github.com/pytorch/pytorch/issues/1410 for (N-ddof).
        # --------------------------------------------------------------------------------
        ne.evaluate("2 * dV * Xmd / (N - ddof)", out=dXmd02)

        # --------------------------------------------------------------------------------
        # dL/dXmd02:(N,M) = dL/dXstd:(N,M) * norm:(M,)
        # --------------------------------------------------------------------------------
        ne.evaluate("dXstd * norm", out=dXmd02)

        # --------------------------------------------------------------------------------
        # dL/dU: (M,) = sum(-dL/dXmd01 + -dL/dXmd02, axis=0)
        # --------------------------------------------------------------------------------
        ne.evaluate("sum(-1 * (dXmd01 + dXmd01), axis=0)", out=dU)

        # --------------------------------------------------------------------------------
        # dL/dX: (N,M) = dL/dXmd01 + dL/dXmd02  + dU/N
        # --------------------------------------------------------------------------------
        ne.evaluate("dXmd01 + dXmd02 + (dU / N)", out=self._dX)

        return self.dX

    def _gradient_descent(self, X, dX, out=None) -> Union[np.ndarray, float]:
        """Gradient descent
        Directly update matrices to avoid the temporary copies
        """
        return self.optimizer.update(X, dX, out=out)

    def update(self) -> List[Union[float, np.ndarray]]:
        """Run gradient descent
        Returns:
            [dL/dGamma, dL/dBeta]: List of gradients.
       """
        self._gradient_descent(self.gamma, self.dGamma, out=self._gamma)
        self._gradient_descent(self.beta, self.dBeta, out=self._beta)
        return [self.dGamma, self.dBeta]

    def predict(self, X):
        """Predict
        Args:
            X: input
        Returns:
            Prediction: Index to the category
        """
        assert \
            isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT and \
            X.ndim == 2 and X.shape[1] == self.M and X.size > 0

        assert np.all(self.RSD > 0)
        RU = self.RU
        RSD = self.RSD
        gamma = self.gamma
        beta = self.beta
        P = ne.evaluate("gamma * ((X - RU) / RSD) + beta")

        return P

