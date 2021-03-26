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
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable,
    Iterable,
    NoReturn,
    Final
)
import logging
import numpy as np
from layer.base import Layer
from common.constants import (
    TYPE_FLOAT
)
from common.functions import (
    compose
)


class Sequential(Layer):
    """Container layer to sequence layers."""

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
            layers: List[Layer],
            posteriors: Optional[List[Layer]] = None,
            log_level: int = logging.ERROR
    ):
        """Initialize a matmul layer that has 'num_nodes' nodes
        Args:
            name: Layer identity name
            num_nodes: Number of nodes M of the first the layer
            posteriors: Post layers to which forward the matmul layer output
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert \
            isinstance(layers, List) and len(layers) > 0 and \
            all([isinstance(__layer, Layer) for __layer in layers])

        assert num_nodes == layers[0].num_nodes, \
            "The num_nodes %s must match with that of the first layer %s" \
            % (num_nodes, layers[0].num_nodes)

        self._layers: List[Layer] = layers

        # Layer function F=(fn-1 o ... o f0)
        self.function: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = \
            compose(*[__layer.function for __layer in layers])

        self.predict: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = \
            compose(*[__layer.predict for __layer in layers])

        # Gradient function G=(g0 o g1 o ... o gn-1)
        self.gradient: [[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = \
            compose(*[__layer.gradient for __layer in layers[::-1]])

        # List of dSi which is List[Gradients] from each layer
        self._dS = []

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def layers(self) -> List[Layer]:
        """Inference layers"""
        return self._layers

    @property
    def num_layers(self) -> int:
        """Number of layers"""
        return len(self.layers)

    @property
    def objective(self) -> Callable[[np.ndarray], np.ndarray]:
        """Layer objective function
        Layer i has its objective function Li = (fn-1 o ... o fi+1) for i < n-1
        that calculates L=Li(Yi) from its output Yi=fi(Xi). L = Li(fi) i < n-1.

        Li   = (fn-1 o ... o fi+1)
        Li-1 = (fn-1 o ... o fi+1 o fi) = Li(fi) = Li o fi
        L = Li-1(fi-1(Xi-1)) = Li(fi(Yi-1)) where Yi-1=fi-1(Xi-1)
        #
        Set Li to each inference layer i in the reverse order.
        Ln-2 for the last inference layer n-2 is fn-1 of the objective layer.
        Ln-2 = fn-1 by the definition Li = (fn-1 o ... o fi+1).
        """
        assert callable(self._objective), "Objective function L not initialized."
        return self._objective

    @objective.setter
    def objective(self, objective: Callable[[np.ndarray], np.ndarray]) -> NoReturn:
        assert callable(objective)
        self._objective = objective
        Li = objective

        for layer in self.layers[::-1]:
            layer.objective = Li
            # Next Li is Li-1 = Li(fi)
            Li = compose(*[layer.function, Li])

    @property
    def dS(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Gradients dL/dS that have been used to update S in each layer"""
        assert self._dS, "Gradients dL/dS of the network not initialized."
        return self._dS

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def update(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Invoke the update() method of each layer in the container.
        Returns:
            [*dL/dS]: List of dL/dS form each layer update()
        """
        self._dS = [layer.update() for layer in self.layers]
        return self.dS
