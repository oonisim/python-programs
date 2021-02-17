"""Base for neural network implementations"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Callable,
    NoReturn,
    Final
)
import logging
import numpy as np
from layer import Layer
from common.functions import (
    compose
)


class Base:
    """Neural network base class"""

    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(
            self,
            name: str,
            layers: List[Layer],
            objective: Layer,
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: network ID
            layers: layers for inference
            objective: objective layer
            log_level: logging level
        """
        assert name
        self._name: str = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(log_level)

        # --------------------------------------------------------------------------------
        # Objective layers in the network. List although only one layer.
        # --------------------------------------------------------------------------------
        assert objective
        self._X = List[Union[float, np.ndarray]] = []   # Bath input to the network
        self._T = List[Union[float, np.ndarray]] = []   # Labels for the bach.

        self._L: np.ndarray = -np.inf                   # Network objective value L
        self._G: List[Union[float, np.ndarray]] = []    # Gradients dL/dX of layers
        self._GN: List[Union[float, np.ndarray]] = []   # Numerical gradients GN of layers
        self._U: List[Union[float, np.ndarray]] = []    # Gradients dL/dS of layers

        self._layers_objective: List[Layer] = [objective]
        self._num_layers_objective = len(self._layers_objective)

        # Layer i objective function Li: (i includes all inference and objective layers)
        # Each layer i has its objective function Li = (fn-1 o ... o fi+1) with i < n-1
        # that calculates L as Li(Yi) from its output Yi=fi(arg). L = Li(fi) i < n-1.
        # The last layer Ln-1 is the objective function that does not have Ln, hence
        # set an identity function.
        # The last inference layer has fn-1 of the objective layer as its objective function.
        def identity(x: np.ndarray): return x
        objective.objective = identity

        # --------------------------------------------------------------------------------
        # Inference layers in the network
        # --------------------------------------------------------------------------------
        assert layers
        self._layers_inference: List[Layer] = layers
        self._num_layers_inference = len(layers)
        self.__inference: Callable[[np.ndarray], np.ndarray] = \
            compose(*[layer.forward for layer in layers])

        # Each layer i has its objective function Li = (fn-1 o ... o fi+1) with i < n-1
        # The last inference layer has fn-1 of the objective layer as its Li.
        Li: Callable[[np.ndarray], np.ndarray] = objective.forward
        for layer in layers:
            layer.objective = Li
            Li = compose(*[layer.forward, Li])

        # After the first inference layer, Li is the network objective function L.
        # L = (fn-1 o ... o f0)
        self.__objective: Callable[[np.ndarray], np.ndarray] = Li

        # --------------------------------------------------------------------------------
        # Entire network layers
        # --------------------------------------------------------------------------------
        self._layers_all: List[Layer] = layers + [objective]
        self._num_layers_all = len(layers + [objective])
        self.__gradient = \
            compose(*[layer.gradient for layer in layers + [objective]])

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A unique name to identify a network"""
        return self._name

    @property
    def X(self) -> List[Union[float, np.ndarray]]:
        """Network batch input"""
        assert self._X, "Batch input not initialized or an invalid value."
        return self._X

    @property
    def T(self) -> List[Union[float, np.ndarray]]:
        """Network batch input"""
        assert self._T, "Label T not initialized or an invalid value."
        return self._T

    @property
    def L(self) -> np.ndarray:
        """Network objective value (Loss)"""
        assert self._L > 0, "Objective value L not initialized or an invalid value."
        return self._L

    @property
    def GN(self) -> List[Union[float, np.ndarray]]:
        """Numerical gradients of network layers"""
        assert self._GN, "Numerical gradients GN of the network not initialized."
        return self._GN

    @property
    def G(self) -> List[Union[float, np.ndarray]]:
        """Gradients dL/dX of network layers"""
        assert self._G, "Gradients dL/dX of the network not initialized."
        return self._G

    @property
    def U(self) -> List[Union[float, np.ndarray]]:
        """Gradients dL/dS of network layers used to update S"""
        assert self._U, "Gradients dL/dS of the network not initialized."
        return self._U

    @property
    def layers_inference(self) -> List[Layer]:
        """Inference layers"""
        assert self._layers_inference and len(self._layers_inference) > 0, \
            "Network inference layers not initialized"
        return self._layers_inference

    @property
    def layers_objective(self) -> List[Layer]:
        """objective layers"""
        assert self._layers_objective and len(self._layers_objective) > 0, \
            "Network objective layers not initialized"
        return self._layers_objective

    @property
    def layers_all(self) -> List[Layer]:
        """all layers"""
        assert self._layers_all and len(self._layers_all) > 0, \
            "Network all layers not initialized"
        return self._layers_all

    @property
    def num_layers(self) -> int:
        """Number of layers"""
        assert self._num_layers_all > 0, "Network num_layers is not initialized."
        return self._num_layers_all

    @property
    def _inference(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network inference function I=fn-2 o ... o f0"""
        assert self.__inference, "Inference function I has not been initialized."
        return self.__inference

    @property
    def _objective(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network Objective function L=fn-1 o fn-2 o ... o f0"""
        assert self.__objective, "Objective function L has not been initialized."
        return self.__objective

    @property
    def _gradient(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network gradient function G=gn-1 o gn-2 o ... o g0"""
        assert self.__gradient, "gradient function G has not been initialized."
        return self.__gradient

    @property
    def logger(self) -> logging.Logger:
        """Instance logger"""
        assert self._logger, "logger is not initialized"
        return self._logger

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def set(self, T: List[Union[float, np.ndarray]]):
        """Setup a batch
        Args:
            T: labels
        """
        self._T = T
        for layer in self.layers_objective:
            layer.T = T

    def gradient_numerical(self):
        """Numerical Gradients GN"""
        self._GN = [layer.gradient_numerical() for layer in self.layers_all]
        return self.GN

    def update(self):
        """Layer state S updates"""
        self._U = [layer.update() for layer in self.layers_all]
        return self.U
