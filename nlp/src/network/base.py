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
            objectives: List[Layer],
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: network ID
            layers: layers for inference
            objectives: objective layer
            log_level: logging level
        """
        assert name
        self._name: str = name

        # Inference layers in the network
        assert layers
        self._layers_inference: List[Layer] = layers
        self._num_layers_inference = len(layers)
        self._inference: Callable[[np.ndarray], np.ndarray] = \
            compose(*[l.forward for l in layers])

        # Objective layers in the network
        assert objectives
        self._layers_objective: List[Layer] = objectives
        self._num_layers_objective = len(objectives)

        self._layers_all: List[Layer] = layers + objectives
        self._num_layers_all = len(self._layers_all)
        self._objective: Callable[[np.ndarray], np.ndarray] = \
            compose(*[layer.forward for layer in self._layers_all])

        # Objective value
        self._L: np.ndarray = -np.inf

        self._logger = logging.getLogger(name)
        self._logger.setLevel(log_level)

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A unique name to identify a network"""
        return self._name

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
        """Network inference function"""
        return self._inference

    @property
    def objective(self) -> Callable[[np.ndarray], np.ndarray]:
        """Objective function L=fn-1 o fn-2 o ... o fi"""
        assert self._objective, "Objective function L has not been initialized."
        return self._objective

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

    def gradient_numerical(
            self, L: Callable[[np.ndarray], np.ndarray], h: float = 1e-05
    ) -> np.ndarray:
        """Calculate numerical gradients
        Args:
            L: Objective function for the layer. objective=L(f(X)), NOT L for NN.
            h: small number for delta to calculate the numerical gradient
        Returns:
            dX: [L(f(X+h) - L(f(X-h)] / 2h
        """
        assert False, "I am a template method. You need to override me."

    def update(self, dY: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """Calculate the gradient dL/dS and update S
        Args:
            dY: dL/dY
        Returns:
            dL/dS: Gradient(s) on state S. There may be multiple dL/dS.
        """
        assert False, "I am a template method. You need to override me."
