"""A layer of composite of layers
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
    build_layers_from_composite_layer_specification
)
from layer.composite_layer_specification_template import (
    _composite_layer_specification
)


class Composite(Layer):
    """Container layer"""

    # ================================================================================
    # Class initialization
    # ================================================================================
    @staticmethod
    def specification_template():
        return _composite_layer_specification

    @staticmethod
    def _build_layers(parameters: Dict) -> List[Layer]:
        assert (
            _NAME in parameters and
            _NUM_NODES in parameters and
            _COMPOSITE_LAYER_SPEC in parameters
        )
        log_level = parameters[_LOG_LEVEL] if _LOG_LEVEL in parameters else logging.ERROR
        layers = build_layers_from_composite_layer_specification(
            specification=parameters[_COMPOSITE_LAYER_SPEC],
            log_level=log_level
        )
        return layers

    @staticmethod
    def build(parameters: Dict):
        raise NotImplementedError("Must override")

    # ================================================================================
    # Instance initialization
    # ================================================================================
    @staticmethod
    def _wire_layer_interfaces(
            layers: List[Layer]
    ) -> Tuple[Callable, Callable, Callable]:
        raise NotImplementedError("Must override")

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
            layers: Layers in the composite
                flag to omit the last activation at compositing prediction.
            posteriors: Post layers to which forward the matmul layer output
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert \
            isinstance(layers, List) and len(layers) > 0 and \
            all([isinstance(__layer, Layer) for __layer in layers])

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])

        # --------------------------------------------------------------------------------
        # num_nodes is to specify the number of outputs from the layer.
        # For objective layer(s), num_nodes are the same at input and output.
        # Hence the input number into the objective layer will match the
        # output number from the inference layer by checking num_nodes with
        # the number of the last layer output.
        # --------------------------------------------------------------------------------
        assert num_nodes == layers[-1].num_nodes, \
            "The num_nodes %s must match with that of the last layer %s" \
            % (num_nodes, layers[-1].num_nodes)

        self._layers: List[Layer] = layers
        self.function, self.predict, self.gradient = \
            self._wire_layer_interfaces(self.layers)

        # --------------------------------------------------------------------------------
        # Layer objective to be initialized with its setter
        # --------------------------------------------------------------------------------
        self._objective = None

        # --------------------------------------------------------------------------------
        # State of the layer
        # --------------------------------------------------------------------------------
        self._S: List[
            List[Union[float, np.ndarray]]
        ] = [__layer.S for __layer in self.layers]
        self._dS: List[
            List[Union[float, np.ndarray]]
        ] = []

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def T(self) -> np.ndarray:
        """Label in OHE or index format"""
        return super().T

    @T.setter
    def T(self, T: Union[np.ndarray, TYPE_LABEL]):
        super(Composite, type(self)).T.fset(self, T)
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
    def dS(self) -> List[
            List[Union[float, np.ndarray]]
    ]:
        """List of the state gradients from each layer [ dS0, dS1, ..., dSn-1]
        Layers may not have state e.g. ReLU, hence cannot check if initialized
        """
        return self._dS

    @property
    def layers(self) -> List[Layer]:
        """Layers in the composite"""
        return self._layers

    @property
    def num_layers(self) -> int:
        """Number of layers in the composite"""
        return len(self.layers)

    @property
    def layer_names(self) -> List[str]:
        """Inference layers"""
        return [_layer.name for _layer in self.layers]

    @property
    def objective(self) -> Callable[[np.ndarray], np.ndarray]:
        return super().objective

    @objective.setter
    def objective(self, objective: Callable[[np.ndarray], np.ndarray]) -> NoReturn:
        """
        Responsibility:
            Set objective function
        """
        assert callable(objective)
        self._objective = objective

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def _set_label(self, T: Union[np.ndarray, TYPE_LABEL]):
        """
        Responsibility:
            Set the label T to the layers in the composite.
        """
        for __layer in self.layers:
            __layer.T = T

    def update(self) -> List[
            List[Union[float, np.ndarray]]
    ]:
        """Invoke the update() method of each layer in the container.
        Returns:
            [*dL/dS]: List of dL/dS form each layer update()
        """
        dS = [__layer.update() for __layer in self.layers]
        self._dS = dS
        return dS
