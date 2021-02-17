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
        self._L: np.ndarray = -np.inf   # Objective value
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
        self._inference: Callable[[np.ndarray], np.ndarray] = \
            compose(*[layer.forward for layer in layers])

        # Each layer i has its objective function Li = (fn-1 o ... o fi+1) with i < n-1
        # The last inference layer has fn-1 of the objective layer as its Li.
        Li: Callable[[np.ndarray], np.ndarray] = objective.forward
        for layer in layers:
            layer.objective = Li
            Li = compose(*[layer.forward, Li])

        # After the first inference layer, Li is the network objective function L.
        # L = (fn-1 o ... o f0)
        self._objective: Callable[[np.ndarray], np.ndarray] = Li

        # --------------------------------------------------------------------------------
        # Entire network layers
        # --------------------------------------------------------------------------------
        self._layers_all: List[Layer] = layers + [objective]
        self._num_layers_all = len(layers + [objective])
        self._gradient = \
            compose(*[layer.gradient for layer in (layers + [objective])])
        self._gradient_numerical = \
            compose(*[layer.gradient for layer in (layers + [objective])])
        self._update = \
            compose(*[layer.update for layer in (layers + [objective])])

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A unique name to identify a network"""
        return self._name

    @property
    def L(self) -> np.ndarray:
        """Network objective value (Loss)"""
        assert self._L > 0, "Objective value not initialized or an invalid value."
        return self._L

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
    def num_layers(self) -> int:
        """Number of layers"""
        assert self._num_layers_all > 0, "Network num_layers is not initialized."
        return self._num_layers_all

    @property
    def inference(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network inference function I=fn-2 o ... o f0"""
        assert self._inference, "Inference function I has not been initialized."
        return self._inference

    @property
    def objective(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network Objective function L=fn-1 o fn-2 o ... o f0"""
        assert self._objective, "Objective function L has not been initialized."
        return self._objective

    @property
    def gradient(self) -> Callable[[np.ndarray], np.ndarray]:
        """Network gradient function G=gn-1 o gn-2 o ... o g0"""
        assert self._gradient, "gradient function G has not been initialized."
        return self._gradient

    @property
    def logger(self) -> logging.Logger:
        """Instance logger"""
        assert self._logger, "logger is not initialized"
        return self._logger

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def set(self, T: np.ndarray):
        """Setup a batch
        Args:
            T: labels
        """
        for layer in self.layers_objective:
            layer.T = T

    def function(self, X: np.ndarray) -> np.ndarray:
        """Calculate the network inference F(arg)
        Args:
            X: input to the layer
        Returns:
            Y: Layer output
        """
        return self.inference(X)

    def gradient(self, dY: np.ndarray) -> np.ndarray:
        """Calculate the gradient dL/dX=g(dL/dY), the impact on L by the input dX
        to back propagate to the previous layer.

        Args:
            dY: dL/dY, impact on L by the layer output Y
        Returns:
            dL/dX: impact on L by the layer input X
        """
        assert False, "I am a template method. You need to override me."

    def backward(self) -> np.ndarray:
        """Calculate and back-propagate the gradient dL/dX"""
        assert False, "I am a template method. You need to override me."

    def update(self, dY: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """Calculate the gradient dL/dS and update S
        Args:
            dY: dL/dY
        Returns:
            dL/dS: Gradient(s) on state S. There may be multiple dL/dS.
        """
        assert False, "I am a template method. You need to override me."
