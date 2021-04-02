"""A layer of sequence of layers
A sequence of layers is yet another layer which has the same signature of a layer.
The difference is each I/F is a composite of the layers in the sequence.
Hereafter, this sequential layer is called 'container' or 'container layer'.

layer function F:
    F=(fn-1 o ... fi o ... o f0) where i is the index of each layer in the container.

Layer gradient G
    G=g0(g1(...(gn-1(X))...) = (g0 o ... o gi o ... gn-1)
    Back-propagation passes the gradient dL/dX backwards from the last layer in the network.

Objective function Li for an internal layer i:
    Li = (Li+1.objective o fi) or Li+1(fi(X)) for i < n-1.
    The last layer (n-i) in the container does not know its objective function Ln-1
    until it is set to the container layer.
"""
import logging
from typing import (
    Tuple,
    Optional,
    Union,
    List,
    Dict,
    Callable,
    NoReturn
)

import numpy as np

from common.constants import (
    TYPE_LABEL
)
from layer.base import Layer
from layer.constants import (
    _NAME,
    _NUM_NODES,
    _LOG_LEVEL,
    _COMPOSITE_LAYER_SPEC
)
from layer.utility_builder_layer import (
    compose_sequential_layer_interface,
    compose_sequential_layer_objective
)
from layer.composite import (
    Composite
)


class Sequential(Composite):
    """Container layer to sequence layers."""

    # ================================================================================
    # Class initialization
    # ================================================================================
    @staticmethod
    def build(parameters: Dict):
        layers = super(Sequential, Sequential)._build_layers(parameters)
        return Sequential(
            name=parameters[_NAME],
            num_nodes=parameters[_NUM_NODES],
            layers=layers,
            log_level=parameters[_LOG_LEVEL] if _LOG_LEVEL in parameters else logging.ERROR
        )

    # ================================================================================
    # Instance initialization
    # ================================================================================
    @staticmethod
    def _wire_layer_interfaces(layers: List[Layer]) -> Tuple[Callable, Callable, Callable]:
        # --------------------------------------------------------------------------------
        # Layer function F=(fn-1 o ... o f0)
        # Layer prediction function F=(fn-1 o ... o f0)
        # Gradient function G=(g0 o g1 o ... o gn-1)
        # --------------------------------------------------------------------------------
        function, predict, gradient = \
            compose_sequential_layer_interface(layers)

        return function, predict, gradient

    def __init__(
            self,
            name: str,
            num_nodes: int,
            layers: List[Layer],
            posteriors: Optional[List[Layer]] = None,
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes M of the first the layer
            layers: Layers to sequence
                flag to omit the last activation at compositing prediction.
            posteriors: Post layers to which forward the matmul layer output
            log_level: logging level
        """
        super().__init__(
            name=name,
            num_nodes=num_nodes,
            layers=layers,
            posteriors=None,
            log_level=log_level
        )

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def objective(self) -> Callable[[np.ndarray], np.ndarray]:
        return super().objective

    @objective.setter
    def objective(self, objective: Callable[[np.ndarray], np.ndarray]) -> NoReturn:
        """
        Responsibility:
            Set objective function Li=(fn-1 o fn-2 o ... o fi) to each layer i
            in the sequence.

            Layer i has its objective function Li = (fn-1 o ... o fi+1) for i < n-1
            that calculates L=Li(Yi) from its output Yi=fi(Xi). L = Li(fi) i < n-1.
        """
        super(Sequential, type(self)).objective.fset(self, objective)
        compose_sequential_layer_objective(self.layers, objective)

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
