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
from layer.utilities import (
    compose_sequential_layer_interface,
    compose_sequential_layer_objective
)
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL
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

        # --------------------------------------------------------------------------------
        # State of the layer
        # --------------------------------------------------------------------------------
        self._S = [__layer.S for __layer in self.layers]

        # --------------------------------------------------------------------------------
        # Layer function F=(fn-1 o ... o f0)
        # Gradient function G=(g0 o g1 o ... o gn-1)
        # --------------------------------------------------------------------------------
        self.function, self.predict, self.gradient = \
            compose_sequential_layer_interface(layers)

        # --------------------------------------------------------------------------------
        # Layer objective to be initialized with its setter
        # --------------------------------------------------------------------------------
        self._objective = None

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def T(self) -> np.ndarray:
        """Label in OHE or index format"""
        return super().T

    @T.setter
    def T(self, T: Union[np.ndarray, TYPE_LABEL]):
        super(Sequential, type(self)).T.fset(self, T)
        self._set_label(self.T)

    @property
    def S(self) -> List[
            List[Union[float, np.ndarray]]
    ]:
        """List of the states of from each layer [ S0, S1, ..., Sn-1]
        where each Si is a list of states in the layer Si.
        """
        self._S = [__layer.S for __layer in self.layers]
        return self._S

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
        compose_sequential_layer_objective(self.layers, objective)

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def _set_label(self, T: Union[np.ndarray, TYPE_LABEL]):
        """
        Responsibility:
            Set the label T to the layers in the sequence.
            Sequential is used for objective layer(s) as well.
        """
        for __layer in self.layers:
            __layer.T = T

    def update(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """Invoke the update() method of each layer in the container.
        Returns:
            [*dL/dS]: List of dL/dS form each layer update()
        """
        self._dS = [__layer.update() for __layer in self.layers]
        return self.dS
