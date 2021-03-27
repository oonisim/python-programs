"""
Neural network implementation
"""
from typing import (
    Dict,
    List,
    Tuple,
    Union,
    Callable,
    NoReturn
)
import copy
import logging
import numpy as np
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL,
    LAYER_MUX_NUM_NODES
)
from common.functions import (
    compose
)
import layer
import optimizer as optimiser
from network.base import Network


# From Python 3.6 onwards, the standard dict type maintains
# insertion order by default.
specification_template = {
    "matmul01": {
        "scheme": layer.Matmul.__qualname__,
        "arguments": {
            "num_nodes": 8,
            "num_features": 3,
            "weights": {
                "scheme": "he"
            },
            "optimizer": {
                "scheme": optimiser.SGD.__qualname__,
                "lr": 0.1,
                "l2": 0.1
            }
        },
    },
    "activation01": {
        "scheme": layer.ReLU.__qualname__,
        "arguments": {
            "num_nodes": 8
        }
    },
    "matmul02": {
        "scheme": layer.Matmul.__qualname__,
        "parameters": {
            "num_nodes": 3,
            "num_features": 3,
            "weights": {
                "scheme": "he"
            }
        }
    },
    "activation02": {
        "scheme": layer.ReLU.__qualname__,
        "parameters": {
            "num_nodes": 3
        }
    },
    "objective": {
        "scheme": layer.CrossEntropyLogLoss.__qualname__,
        "parameters": {
            "num_nodes": 3,
            "loss_function": "softmax_cross_entropy_log_loss"
        }
    }
}


SCHEME = "scheme"
NAME = "name"
NUM_NODES = "num_nodes"
ARGUMENTS = "arguments"
LOG_LEVEL = "log_level"
LAYER_CLASS = "layer_class"


class SequentialNetwork(Network):
    """Neural network implementation class
    Objective:
        A neural network instance that represents a specific experiment:
        - a set of hyper parameters
        - a specific number of training epochs to run
        - a specific performance metrics achieved (recall, precision, etc).
    """

    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def _parse_network_specs(
            network_spec: Dict,
            num_nodes: int,
            log_level: int
    ) -> Tuple[Dict, Dict]:
        """
        Responsibility
            1. Validate the network specification
            2. Generate inference layer specifications
            3. Generate objective layer specifications

        Args:
            network_spec: Network specification
            num_nodes: Number of nodes (outputs) of the network.
            log_level: logging level

        Returns:
            (inference_layer_specs, objective_layer_specs)

            inference_layer_specs defines the layers to produce an inference.
            objective_layer_specs defines the layer to produce the network
            objective e.g. the loss. Objective layer can be single but to
            allow cascading, it is a list.
        """
        assert isinstance(network_spec, dict) and len(network_spec) > 0
        assert 0 < num_nodes < LAYER_MUX_NUM_NODES, \
            "0 < num_nodes %s required" % LAYER_MUX_NUM_NODES

        inference_layer_specs = {}
        objective_layer_specs = {}
        for layer_name, layer_spec in network_spec.items():
            assert isinstance(layer_name, str) and isinstance(layer_spec, dict)
            if SCHEME in layer_spec:
                layer_name = layer_name.lower()
                layer_scheme = layer_spec[SCHEME].lower()     # Layer class, e.g. Matmul

                # --------------------------------------------------------------------------------
                # Compose the arguments to pass to the layer __init__()
                # --------------------------------------------------------------------------------
                args = layer_spec[ARGUMENTS] \
                    if ARGUMENTS in layer_spec else {}

                # Mandatory "name" arg. Use the layer_spec key when not specified
                args[NAME] = layer_name \
                    if NAME not in args else args[NAME].lower()

                # Mandatory "num_nodes" arg.
                assert NUM_NODES in args, "%s is mandatory" % NUM_NODES

                # Logging default level
                args[LOG_LEVEL] = log_level \
                    if LOG_LEVEL not in args else args[LOG_LEVEL]

                # --------------------------------------------------------------------------------
                # Compose a layer spec
                # --------------------------------------------------------------------------------
                if layer_scheme in layer.FUNCTION_LAYER_SCHEMES:
                    target_layer_spec = inference_layer_specs
                elif layer_scheme in layer.OBJECTIVE_LAYER_SCHEMES:
                    target_layer_spec = objective_layer_specs
                else:
                    assert False, \
                        "Invalid layer %s. Must be one of %s." \
                        % (layer_scheme, layer.SCHEMES.keys())

                assert not (
                    layer_name in target_layer_spec or layer_name == LAYER_CLASS
                ), "Duplicated layer name [%s] or do not use [%s] as a dictionary key" \
                   % (layer_name, LAYER_CLASS)

                target_layer_spec[layer_name][LAYER_CLASS] = \
                    layer.FUNCTION_LAYER_SCHEMES[layer_scheme]
                target_layer_spec[layer_name][ARGUMENTS] = \
                    copy.deepcopy(args)

        assert \
            len(inference_layer_specs) > 0 and \
            len(objective_layer_specs) > 0, \
            "There must be at least 1 inference and 1 objective layer. " \
            "Inference count %s objective count is %s" \
            % (len(inference_layer_specs), len(objective_layer_specs))

        assert \
            inference_layer_specs[-1][ARGUMENTS][NUM_NODES] == \
            objective_layer_specs[0][ARGUMENTS][NUM_NODES] == \
            num_nodes, \
            "The number of nodes in the last inference layer [%s] "\
            "must match that of the first objective layer [%s]." \
            % (
                inference_layer_specs[-1][ARGUMENTS][NUM_NODES],
                objective_layer_specs[0][ARGUMENTS][NUM_NODES]
            )

        assert set(inference_layer_specs.keys()) == {LAYER_CLASS, ARGUMENTS}
        assert set(objective_layer_specs.keys()) == {LAYER_CLASS, ARGUMENTS}

        return inference_layer_specs, objective_layer_specs

    @staticmethod
    def _build_network_layers(
            layer_specs: Dict
    ) -> List[layer.Layer]:
        return [
            spec[LAYER_CLASS].build(spec[ARGUMENTS])
            for spec in layer_specs.values()
        ]

    @staticmethod
    def _wire_network_layers(
            inference_layers: List[layer.Layer],
            objective_layers: List[layer.Layer],
            name: str,
            num_nodes: int,
            log_level: int,
    ):
        """
        Responsibility:
            Wire the layers to function as a network.

        Args:
            inference_layers: layer instances to produce an inference.
            objective_layers: layer to produce the objective e.g. the loss.

        Returns:
            layer_inference, layer_objective

        """
        def identity(x: np.ndarray):
            assert x.ndim == 0, "The output of the log loss should be of shape ()"
            return x

        layer_objective = layer.Sequential(
            name=name + "_objective",
            num_nodes=num_nodes,
            layers=objective_layers,
            log_level=log_level
        )
        # Sequential objective is initialized upon its setter call.
        # Hence required even when the objective layer has its setup.
        layer_objective.objective = identity

        layer_inference = layer.Sequential(
            name=name + "_inference",
            num_nodes=num_nodes,
            layers=inference_layers,
            log_level=log_level
        )
        layer_inference.objective = layer_objective.function

        return layer_inference, layer_objective

    @staticmethod
    def build(
            name: str,
            num_nodes: int,
            network_spec: Dict,
            log_level: int
    ):
        """Build a neural network instance from the specification
        """
        inference_layer_specs, objective_layer_specs = \
            SequentialNetwork._parse_network_specs(
                network_spec=network_spec,
                num_nodes=num_nodes,
                log_level=log_level
            )

        inference_layers = SequentialNetwork._build_network_layers(
            inference_layer_specs
        )
        objective_layers = SequentialNetwork._build_network_layers(
            objective_layer_specs
        )

        *layers, = SequentialNetwork._wire_network_layers(
            inference_layers=inference_layers,
            objective_layers=objective_layers,
            name=name,
            num_nodes=num_nodes,
            log_level=log_level
        )
        layer_inference, layer_objective = layers

        return layer_inference, layer_objective

    # ================================================================================
    # Instance
    # ================================================================================

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            num_nodes: int,
            specification: Dict,
            log_level: int = logging.ERROR
    ):
        """Initialize the network instance
        Args:
            name: Network ID
            num_nodes: Number of nodes
            specification: Network specification in JSON.
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        spec = copy.deepcopy(specification)

        # --------------------------------------------------------------------------------
        # Build network
        # --------------------------------------------------------------------------------
        self._specification = spec
        self._layer_inference, self._layer_objective = SequentialNetwork.build(
            name=name,
            num_nodes=num_nodes,
            network_spec=spec,
            log_level=log_level
        )
        self._layers_all = [self.layer_inference, self.layer_objective]

        # --------------------------------------------------------------------------------
        # Wire network functions
        # --------------------------------------------------------------------------------
        self._function: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = \
            self.layer_inference.function

        self._predict: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = \
            self.layer_inference.predict

        self._gradient: Callable[[Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]] = \
            self.layer_inference.gradient
