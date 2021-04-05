"""Normalization layer implementation
"""
import copy
import logging
from typing import (
    Optional,
    Union,
    List,
    Dict
)

import numexpr as ne
import numpy as np
from numba import jit

from common.constants import (
    TYPE_FLOAT,
    ENABLE_NUMEXPR,
    ENABLE_NUMBA,
)
from common.function import (
    standardize,
    numerical_jacobian
)
from layer.base import Layer
from layer.constants import (
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _PARAMETERS
)
from layer._utility_builder_non_layer import (
    build_optimizer_from_layer_parameters
)
import optimizer as optimiser


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

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[np.ndarray, TYPE_FLOAT]) -> Union[np.ndarray, TYPE_FLOAT]:
        """Calculate batch normalization.
        Args:
            X: Batch input data from the input layer.
        Returns:
            Y: gamma * Xstd + beta.
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

        standardize(X, out=self._Y)
        return self.Y

    def gradient(self, dY: Union[np.ndarray, TYPE_FLOAT] = 1.0) -> Union[np.ndarray, TYPE_FLOAT]:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        """
        name = "gradient"
        assert isinstance(dY, TYPE_FLOAT) or (isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT)

        dY = np.array(dY).reshape((1, -1)) if isinstance(dY, TYPE_FLOAT) or dY.ndim < 2 else dY
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
    @staticmethod
    def specification_template():
        return BatchNormalization.specification(name="bn001", num_nodes=3)

    @staticmethod
    def specification(
            name: str,
            num_nodes: int,
            gamma_optimizer_specification: dict = None,
            beta_optimizer_specification: dict = None,
            momentum: TYPE_FLOAT = 0.9
    ):
        """Generate Matmul specification
        """
        return {
            _SCHEME: BatchNormalization.__qualname__,
            _PARAMETERS: {
                _NAME: name,
                _NUM_NODES: num_nodes,
                "momentum": momentum,
                # Use same optimizer for gamma and beta unless it is proven not sufficient
                _OPTIMIZER: gamma_optimizer_specification
                if gamma_optimizer_specification is not None
                else optimiser.SGD.specification(),
                "eps": 0.0,
                "log_level": logging.ERROR
            }
        }

    @staticmethod
    def build(parameters: Dict):
        """Build a matmul layer based on the parameters
        """
        parameters = copy.deepcopy(parameters)
        assert (
            isinstance(parameters, dict) and
            (_NAME in parameters and len(parameters[_NAME]) > 0) and
            (_NUM_NODES in parameters and parameters[_NUM_NODES] > 0)
        )

        # Optimizer
        _optimizer = build_optimizer_from_layer_parameters(parameters)
        parameters[_OPTIMIZER] = _optimizer

        if "eps" in parameters:
            parameters["eps"] = TYPE_FLOAT(parameters["eps"])
        if "momentum" in parameters:
            parameters["momentum"] = TYPE_FLOAT(parameters["momentum"])

        instance = BatchNormalization(**parameters)

        return instance

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(
            self,
            name: str,
            num_nodes: int,
            momentum: TYPE_FLOAT = 0.9,
            optimizer: optimiser.Optimizer = optimiser.SGD(),
            eps: TYPE_FLOAT = 0.0,
            posteriors: Optional[List[Layer]] = None,
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            momentum: Decay of running mean/variance
            eps: Prevent x/sqrt(var+eps) from div-by-zero in standardize(eps=eps)
            optimizer: Gradient descent implementation e.g SGD, Adam.
            posteriors: Post layers to which forward the matmul layer output
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        self.logger.debug(
            "BatchNormalization[%s] number of nodes is [%s] momentum is %s]",
            name, num_nodes, momentum
        )
        assert TYPE_FLOAT(0) < momentum < TYPE_FLOAT(1)
        assert TYPE_FLOAT(0) <= eps < TYPE_FLOAT(1e-3)
        assert isinstance(optimizer, optimiser.Optimizer)

        # Layers to which forward the output
        self._posteriors: List[Layer] = posteriors
        self._num_posteriors: int = len(posteriors) if posteriors else -1

        # --------------------------------------------------------------------------------
        # Standardization variables
        # --------------------------------------------------------------------------------
        # Per-feature standardized X
        self._Xstd = np.empty(0, dtype=TYPE_FLOAT)
        self._dXstd = np.empty(0, dtype=TYPE_FLOAT)
        # Per-feature mean deviation (X-mean)
        self._Xmd = np.empty(0, dtype=TYPE_FLOAT)
        self._dXmd01 = np.empty(0, dtype=TYPE_FLOAT)
        self._dXmd02 = np.empty(0, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Batch statistics. allocate storage at the instance initialization.
        # --------------------------------------------------------------------------------
        self._eps = eps
        # Feature mean of shape(M)
        self._U: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        # Gradient dL/dU of shape (M,)
        self._dU: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        # SD-per-feature of batch X in shape (M,)
        self._SD: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        # Norm = 1/SD in shape (M,)
        self._norm: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        # Gradient of variance V in shape (M,)
        self._dV: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Running statistics. allocate storage at the instance initialization.
        # --------------------------------------------------------------------------------
        self._total_rows_processed = 0
        self._total_training_invocations = 0
        self._momentum: TYPE_FLOAT = momentum
        # Running mean-per-feature of all the batches X* in shape(M,)
        self._RU: np.ndarray = np.zeros(num_nodes, dtype=TYPE_FLOAT)
        # Running SD-per-feature of all the batches X* in shape (M,)
        self._RSD: np.ndarray = np.zeros(num_nodes, dtype=TYPE_FLOAT)
        self.update_running_means = self._first_update_running_means

        # --------------------------------------------------------------------------------
        # Scale and shift parameters
        # --------------------------------------------------------------------------------
        self._gamma: np.ndarray = np.ones(num_nodes, dtype=TYPE_FLOAT)
        self._dGamma: np.ndarray = np.empty(num_nodes, dtype=TYPE_FLOAT)
        self._beta: np.ndarray = np.zeros(num_nodes, dtype=TYPE_FLOAT)
        self._dBeta: np.ndarray = np.zeros(num_nodes, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # State of the layer
        # --------------------------------------------------------------------------------
        self._S = [self.gamma, self.beta]

        # --------------------------------------------------------------------------------
        # Gradient descent optimizer
        # --------------------------------------------------------------------------------
        self._optimizer: optimiser.Optimizer = optimizer

        # --------------------------------------------------------------------------------
        # Misc
        # --------------------------------------------------------------------------------
        self._args = set(locals().keys())

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    # @property
    # def X(self) -> np.ndarray:
    #     """Latest batch input to the layer"""
    #     return super().X

    # @X.setter
    # def X(self, X: np.ndarray):
    #     """Batch X in shape:(N,M)"""
    #     super(BatchNormalization, type(self)).X.fset(self, X)
    #     # Cannot check. Setting _D can be done using weight shape.
    #     # assert self.X.shape[1] == self.D, \
    #     #     "X shape needs (%s, %s) but %s" % (self.N, self.D, self.X.shape)

    @property
    def U(self) -> np.ndarray:
        """Mean of each feature in X. Shape:(M,)"""
        assert self._U.size > 0 and self._U.shape == (self.M,), \
            "U is not initialized or invalid"
        return self._U

    @property
    def dU(self) -> np.ndarray:
        """Gradient dL/dU in shape:(M,)
        dL/dU:(M,) = -sum(dL/dXmd01 + dL/dXmd02, axis=0)
        (-1 in -sum() is from d(X-U)/dU)
        """
        assert self._dU.size == self.M, \
            "dU is not initialized or invalid"
        return self._dU

    @property
    def Xmd(self) -> np.ndarray:
        """Per-feature MD(Mean Deviation) of X = (X-U) in shape:(N,M)
        """
        assert self._Xmd.size > 0 and self._Xmd.shape == (self.N, self.M)
        return self._Xmd

    @property
    def dXmd01(self) -> np.ndarray:
        """
        Gradient 01 of dL/dXmd = (dL/dV / (N-1) * 2Xmd)
        -1 is Bessel's correction at Variance calculation.
        Shape:(N,M) as per (X-mean)
        """
        assert self._dXmd01.size > 0 and self._dXmd01.shape == (self.N, self.M)
        return self._dXmd01

    @property
    def dXmd02(self) -> np.ndarray:
        """
        Gradient 02 of dL/dXmd = [(gamma * dL/dY) * norm] = (dL/dXstd * norm).
        Shape:(N,M)
        """
        assert self._dXmd02.size > 0 and self._dXmd02.shape == (self.N, self.M)
        return self._dXmd02

    @property
    def eps(self) -> TYPE_FLOAT:
        """epsilon to prevent div-by-zero at x/sqrt(var+eps)"""
        return self._eps

    @property
    def dV(self) -> np.ndarray:
        """
        Gradient dL/dV, the impact on L by the variance delta dV of X.
        dL/dV = sum((dL/dY * gamma) * Xmd, axis=0) * [-1/2 * (norm **3)]
              = sum( dL/dXstd       * Xmd, axis=0) * [-1/2 * (norm **3)]
        Shape:(M,)
        """
        assert self._dV.size == self.M, \
            "dV is not initialized or invalid"
        return self._dV

    @property
    def SD(self) -> np.ndarray:
        """Standard Deviation (SD) of each feature in X in shape:(M,)
        Bessel's correction is used when calculating the variance.
        """
        assert self._SD.size > 0 and self._SD.shape == (self.M,), \
            "SD is not initialized or invalid"
        return self._SD

    @property
    def norm(self) -> np.ndarray:
        """1/SD of each feature in X that standardizes X as (X-U) * norm
        Shape:(M,)
        """
        assert self._norm.size > 0 and self._norm.shape == (self.M,), \
            "norm is not initialized or invalid"
        return self._norm

    @property
    def Xstd(self) -> np.ndarray:
        """Per-feature standardized X in shape:(N,M)"""
        assert self._Xstd.size > 0 and self._Xstd.shape == (self.N, self.M)
        return self._Xstd

    @property
    def dXstd(self) -> np.ndarray:
        """Gradient dL/dXstd = (dL/dY * gamma) in shape:(N,M)"""
        assert self._dXstd.size > 0 and self._dXstd.shape == (self.N, self.M)
        return self._dXstd

    @property
    def gamma(self) -> np.ndarray:
        """Feature scale parameter gamma for each feature in X in shape:(M,)
        """
        assert self._gamma.size > 0 and self._gamma.shape == (self.M,), \
            "gamma is not initialized or invalid"
        return self._gamma

    @property
    def dGamma(self) -> np.ndarray:
        """Gradient dL/dGamma = sum(dL/dY * dL/dXstd, axis=0) for each feature in X.
        Shape:(M,)
        """
        assert self._dGamma.size > 0 and self._dGamma.shape == (self.M,)
        return self._dGamma

    @property
    def beta(self) -> np.ndarray:
        """Feature bias parameter beta for each feature in X.
        Shape:(M,)"""
        assert self._beta.size > 0 and self._beta.shape == (self.M,),\
            "beta is not initialized or invalid"
        return self._beta

    @property
    def dBeta(self) -> np.ndarray:
        """Gradient dL/dBeta = sum(dL/dY * 1, axis=0) for each feature in X.
        Shape:(M,)
        """
        assert self._dBeta.size > 0 and self._dBeta.shape == (self.M,), \
            "dBeta is not initialized or invalid"
        return self._dBeta

    @property
    def RU(self) -> np.ndarray:
        """Running mean of each feature in X. Shape:(M,)"""
        assert self._RU.size > 0 and self._RU.shape == (self.M,), \
            "RU is not initialized or invalid"
        return self._RU

    @property
    def RSD(self) -> np.ndarray:
        """Running SD of each feature in X. Shape:(M,)"""
        assert self._RSD.size > 0 and self._RSD.shape == (self.M,), \
            "RSD is not initialized or invalid"
        return self._RSD

    @property
    def S(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """State of the layer"""
        self._S = [self.gamma, self.beta]
        return self._S

    @property
    def momentum(self) -> TYPE_FLOAT:
        """Decay parameter for calculating running means"""
        return self._momentum

    @property
    def total_rows_processed(self) -> TYPE_FLOAT:
        """Total number of training data (row) processed"""
        return self._total_rows_processed

    @property
    def total_training_invocations(self) -> TYPE_FLOAT:
        """Total number of train invocations"""
        return self._total_training_invocations

    @property
    def optimizer(self) -> optimiser.Optimizer:
        """Optimizer instance for gradient descent
        """
        return self._optimizer

    @property
    def gamma_optimizer(self) -> optimiser.Optimizer:
        """Optimizer instance for gamma gradient descent
        For now use the same instance for gamma and beta
        """
        return self._optimizer

    @property
    def beta_optimizer(self) -> optimiser.Optimizer:
        """Optimizer instance for beta gradient descent
        For now use the same instance for gamma and beta
        """
        return self._optimizer

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def _first_update_running_means(self):
        """Set the RS and RSD for the first time only
        At the first invocation, RU and RSD are zero.
        Hence RU = (momentum * RU + (1 - momentum) * U) is 0.1 * U (momentum=0.9).
        However with 1 invocation, the correct running mean RU = U. (U/1=U).
        Hence set U to RU.
        """
        # --------------------------------------------------------------------------------
        # Bug! Beware of the memory address copy.
        # A = B is copying the memory address making A and B refer to the same memory area.
        # Need to update the memory contents of RU, not the address of the memory of RU.
        # Took 2 hours to debug ...
        # --------------------------------------------------------------------------------
        # self._RU = self.U       # <--- Changing the memory address of RU with U
        # self._RSD = self.SD     # <--- Changing the memory address of RSD with SD
        # --------------------------------------------------------------------------------
        np.copyto(self._RU, self.U)     # Copy the contents of U into the memory of RU
        np.copyto(self._RSD, self.SD)   # Copy the contents of SD into the memory of RSD
        # --------------------------------------------------------------------------------
        self.update_running_means = self._update_running_means

    def _update_running_means(self):
        """Update running means using decaying"""
        self._RU = self.momentum * self.RU + (1 - self.momentum) * self.U
        self._RSD = self.momentum * self.RSD + (1 - self.momentum) * self.SD

    @staticmethod
    def _function_numexpr(x, gamma, beta, out):
        ne.evaluate("gamma * x + beta", out=out)

    @staticmethod
    @jit(nopython=True)
    def _function_numba(x, gamma, beta):
        return gamma * x + beta

    @staticmethod
    def _function_numpy(x, gamma, beta, out=None):
        scaled = np.multiply(gamma, x, out=out)
        shifted = np.add(scaled, beta, out=out)
        return shifted

    def function(
            self, X: Union[np.ndarray, TYPE_FLOAT],
            numexpr_enabled: bool = ENABLE_NUMEXPR,
            numba_enabled: bool = ENABLE_NUMBA
    ) -> Union[np.ndarray, TYPE_FLOAT]:
        """Standardize the input X per feature/column basis.
        Args:
            X: Batch input data from the input layer.
            numexpr_enabled: flat to use numexpr
            numba_enabled: flat to use numba
        Returns:
            Y: Per-feature standardized output
        """
        name = "function"
        self.X = X
        assert self.X.shape == (self.N, self.M), \
            "Number of features %s must match number of nodes %s in the BN layer." \
            % (X.shape[1], self.M)      # Make sure to check N is updated too.

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
            self._Xmd = np.empty(X.shape, dtype=TYPE_FLOAT)
            self._dXmd01 = np.empty(X.shape, dtype=TYPE_FLOAT)
            self._dXmd02 = np.empty(X.shape, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Standardize X and update running means.
        # --------------------------------------------------------------------------------
        self._Xstd, self._U, self._SD, self._Xmd = standardize(
            X,
            eps=self.eps,
            keepdims=False,
            out=self._Xstd,
            out_mean=self._U,
            out_sd=self._SD
        )
        self.update_running_means()
        self._norm = 1.0 / self.SD
        assert np.all(np.isfinite(self.norm))

        # --------------------------------------------------------------------------------
        # Calculate layer output Y
        # --------------------------------------------------------------------------------
        gamma = self.gamma
        beta = self.beta
        x = self.Xstd
        out = self._Y
        if numexpr_enabled:
            self._function_numexpr(x=x, gamma=gamma, beta=beta, out=out)
        elif numba_enabled:
            self._Y = self._function_numba(x=x, gamma=gamma, beta=beta)
        else:
            self._function_numpy(x=x, gamma=gamma, beta=beta, out=self._Y)

        # --------------------------------------------------------------------------------
        # Total training data (rows) processed
        # --------------------------------------------------------------------------------
        self._total_training_invocations += 1
        self._total_rows_processed += self.N

        return self.Y

    def _gradient_numpy(self):
        """Calculate dL/dX using numpy"""
        ddof = 1 if self.N > 1 else 0

        # --------------------------------------------------------------------------------
        # dL/dGamma:(M,) = sum(dL/dY:(N,M) * Xstd(N,M), axis=0)
        # --------------------------------------------------------------------------------
        np.sum(self.dY * self.Xstd, axis=0, out=self._dGamma)

        # --------------------------------------------------------------------------------
        # dL/dBeta:(M,) = sum(dL/dY:(N,M) * 1, axis=0)
        # --------------------------------------------------------------------------------
        np.sum(self.dY, axis=0, out=self._dBeta)

        # --------------------------------------------------------------------------------
        # dL/dXstd: (N,M): dL/dY:(N,M) * gamma(M,)
        # --------------------------------------------------------------------------------
        np.multiply(self.dY, self.gamma, out=self._dXstd)

        # --------------------------------------------------------------------------------
        # dL/dV:(M,) = sum(dL/dXstd:(N,M) * Xmd, axis=0) * [-1/2 * (norm**3)]
        # --------------------------------------------------------------------------------
        np.sum(self.dXstd * self.Xmd, axis=0, out=self._dV)
        np.multiply(self.dV, (self.norm ** 3) / -2.0, out=self._dV)

        # --------------------------------------------------------------------------------
        # dL/dXmd01:(N,M) = dL/dV:(M,) / (N-ddof) * 2 * Xmd:(N,M)
        # See https://github.com/pytorch/pytorch/issues/1410 for (N-ddof).
        # --------------------------------------------------------------------------------
        np.multiply(self.dV, self.Xmd, out=self._dXmd01)
        np.divide(self.dXmd01, ((self.N-ddof) / 2.0), out=self._dXmd01)
        assert self.dXmd01.shape == (self.N, self.M)

        # --------------------------------------------------------------------------------
        # dL/dXmd02:(N,M) = dL/dXstd:(N,M) * norm:(M,)
        # --------------------------------------------------------------------------------
        np.multiply(self.dXstd, self.norm, out=self._dXmd02)
        assert self.dXmd02.shape == (self.N, self.M)

        combined = True
        if combined:
            # --------------------------------------------------------------------------------
            # dL/dX:(N,M) = dL/dX01 + dL/dX02
            # - dL/dX01 = (dL/dXmd01 + dL/dXmd02)
            # - dL/dX02 = [sum(-dL/dX01, axis=0) / N]
            # --------------------------------------------------------------------------------
            np.add(self.dXmd01, self.dXmd02, out=self._dX)
            np.sum(self.dX, axis=0, out=self._dU)
            self._dU *= -1
            np.add(self.dX, self.dU / self.N, out=self.dX)
        else:
            # --------------------------------------------------------------------------------
            # dL/dU:(M,) = -sum(dL/dXmd01 + dL/dXmd02, axis=0)
            # * -1 in -sum() is from d(X-U)/dU
            # --------------------------------------------------------------------------------
            np.sum((self.dXmd01+self.dXmd02), axis=0, out=self._dU)
            self._dU *= -1

            # --------------------------------------------------------------------------------
            # dL/dX:(N,M) = dL/dXmd01 + dL/dXmd02  + dU/N
            # --------------------------------------------------------------------------------
            np.add(self.dXmd01, self.dXmd02, out=self._dX)
            np.add(self.dX, self.dU / self.N, out=self._dX)

        return self.dX

    def _gradient_numexpr(self):
        """Calculate dL/dX using numexpr"""
        N = self.N
        dY = self.dY
        ddof = 1 if N > 1 else 0
        Xstd = self.Xstd
        dXstd = self.dXstd
        Xmd = self.Xmd
        dXmd01 = self.dXmd01
        dXmd02 = self.dXmd02
        dV = self.dV
        dU = self.dU
        norm = self.norm
        gamma = self.gamma
        dGamma = self.dGamma
        dBeta = self.dBeta
        dX = self.dX

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
        ne.evaluate("dY * gamma", out=dXstd)

        # --------------------------------------------------------------------------------
        # dL/dV:(M,) = sum(dL/dXstd:(N,M) * Xmd:(N,M), axis=0) * -1/2 * norm**3
        # --------------------------------------------------------------------------------
        ne.evaluate("sum(dXstd * Xmd, axis=0)", out=dV)
        ne.evaluate("dV * (norm ** 3) / -2", out=dV)

        # --------------------------------------------------------------------------------
        # dL/dXmd01:(N,M) = dL/dV:(M,) / (N-ddof) * 2 * Xmd:(N,M)
        # See https://github.com/pytorch/pytorch/issues/1410 for (N-ddof).
        # --------------------------------------------------------------------------------
        ne.evaluate("2 * dV * Xmd / (N - ddof)", out=dXmd01)

        # --------------------------------------------------------------------------------
        # dL/dXmd02:(N,M) = dL/dXstd:(N,M) * norm:(M,)
        # --------------------------------------------------------------------------------
        ne.evaluate("dXstd * norm", out=dXmd02)

        # --------------------------------------------------------------------------------
        # dL/dU: (M,) = sum(-dL/dXmd01 + -dL/dXmd02, axis=0)
        # --------------------------------------------------------------------------------
        ne.evaluate("sum(-1 * (dXmd01 + dXmd02), axis=0)", out=dU)

        # --------------------------------------------------------------------------------
        # dL/dX: (N,M) = dL/dXmd01 + dL/dXmd02  + dU/N
        # --------------------------------------------------------------------------------
        return ne.evaluate("dXmd01 + dXmd02 + (dU / N)", out=dX)

    def gradient(
            self,
            dY: Union[np.ndarray, TYPE_FLOAT] = 1.0,
            numexpr_enabled: bool = ENABLE_NUMEXPR,
            numba_enabled: bool = ENABLE_NUMBA
    ) -> Union[np.ndarray, TYPE_FLOAT]:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
            numexpr_enabled: flat to use numexpr
            numba_enabled: flat to use numexpr
        Returns:
            dX: dL/dX:(N,M) = [         dL/dXmd         + (dL/dU / N) ]
                            = [ (dL/dXmd01 + dL/dXmd02) + (dL/dU / N) ]
        """
        name = "gradient"
        assert isinstance(dY, TYPE_FLOAT) or (isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT)

        dY = np.array(dY).reshape((1, -1)) if isinstance(dY, TYPE_FLOAT) or dY.ndim < 2 else dY
        assert dY.shape == self.Y.shape, \
            "dL/dY shape needs %s but %s" % (self.Y.shape, dY.shape)

        self.logger.debug("layer[%s].%s: dY.shape %s", self.name, name, dY.shape)
        self._dY = dY

        if numexpr_enabled:
            return self._gradient_numexpr()
        else:
            return self._gradient_numpy()

    def gradient_numerical(
            self, h: float = 1e-5
    ) -> List[Union[float, np.ndarray]]:
        """Calculate numerical gradients
        Args:
            h: small number for delta to calculate the numerical gradient
        Returns:
            dX: [L(f(X+h) - L(f(X-h)] / 2h
            dGamma:
            dBeta:
        """
        # TODO:
        self.logger.debug("layer[%s]:gradient_numerical", self.name)
        L = self.objective

        def objective_gamma(gamma: np.ndarray):
            return L(gamma * self.Xstd + self.beta)

        def objective_beta(beta: np.ndarray):
            return L(self.gamma * self.Xstd + beta)

        dX = super().gradient_numerical(h=h)
        dGamma = numerical_jacobian(objective_gamma, self.gamma, delta=h)
        dBeta = numerical_jacobian(objective_beta, self.beta, delta=h)
        return [dX, dGamma, dBeta]

    def _gradient_descent(self, optimizer, X, dX, out=None) -> Union[np.ndarray, TYPE_FLOAT]:
        """Gradient descent
        Directly update matrices to avoid the temporary copies
        """
        return optimizer.update(X, dX, out=out)

    def update(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Run gradient descent
        Returns:
            [dL/dX, dL/dGamma, dL/dBeta]: List of gradients.
       """
        self._gradient_descent(self.gamma_optimizer, self.gamma, self.dGamma, out=self._gamma)
        self._gradient_descent(self.beta_optimizer, self.beta, self.dBeta, out=self._beta)
        # return [self.dX, self.dGamma, self.dBeta]
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
        scores = ne.evaluate("gamma * ((X - RU) / RSD) + beta")

        return scores
