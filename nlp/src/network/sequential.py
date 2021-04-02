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
    LAYER_MAX_NUM_NODES
)
from common.function import (
    compose
)
import layer
from layer.utility_builder_layer import (
    build_layers_from_composite_layer_specification
)
from network.base import Network
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS,
    _LAYER_CLASS,
    _LOG_LEVEL,
    _COMPOSITE_LAYER_SPEC
)


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
    def _validate_network_specification(network_spec: dict):
        assert \
            isinstance(network_spec, dict) and \
            len(network_spec) > 0

        assert \
            _NAME in network_spec and \
            isinstance(network_spec[_NAME], str) and \
            len(network_spec[_NAME]) > 0
        name = network_spec[_NAME].lower()

        assert \
            _NUM_NODES in network_spec and \
            isinstance(network_spec[_NUM_NODES], int) and \
            0 < network_spec[_NUM_NODES] < LAYER_MAX_NUM_NODES, \
            "0 < num_nodes %s required" % LAYER_MAX_NUM_NODES
        num_nodes = network_spec[_NUM_NODES]

        assert \
            _COMPOSITE_LAYER_SPEC in network_spec and \
            isinstance(network_spec[_COMPOSITE_LAYER_SPEC], dict) and \
            len(network_spec[_COMPOSITE_LAYER_SPEC]) > 0, \
            "Invalid %s\n%s\n" % (_COMPOSITE_LAYER_SPEC, network_spec)

        log_level = network_spec[_LOG_LEVEL] \
            if _LOG_LEVEL in network_spec else logging.ERROR

        return name, num_nodes, log_level

    @staticmethod
    def _build_network_layers(
            network_spec: dict,
    ) -> Tuple[List[layer.Layer], List[layer.Layer]]:
        """
        Responsibility
            1. Instantiate network layers
            2. Classify them into inference and objective
        Args:

        Returns:
            (inference_layers, objective_layers)
        """
        num_nodes = network_spec[_NUM_NODES]
        log_level = network_spec[_LOG_LEVEL]
        inference_layers: List[layer.Layer] = []
        objective_layers: List[layer.Layer] = []

        layers = build_layers_from_composite_layer_specification(
            specification=network_spec[_COMPOSITE_LAYER_SPEC],
            log_level=log_level
        )
        assert len(layers) > 0

        layer_names: set = set()
        for _layer in layers:
            assert _layer.name not in layer_names, \
                "Duplicated layer name %s" % _layer.name
            layer_names.add(_layer.name)

            if isinstance(_layer, layer.FUNCTION_LAYERS):
                inference_layers.append(_layer)
            elif isinstance(_layer, layer.OBJECTIVE_LAYERS):
                objective_layers.append(_layer)
            else:
                assert False, \
                    "Invalid layer scheme %s. Must be one of %s." \
                    % (type(_layer), layer.SCHEMES.keys())

        assert \
            len(inference_layers) > 0 and \
            len(objective_layers) > 0, \
            "There must be at least 1 inference and 1 objective layer. " \
            "Inference count %s objective count is %s" \
            % (len(inference_layers), len(objective_layers))

        last_inference_num_nodes = inference_layers[-1].num_nodes
        first_objective_num_nodes = objective_layers[0].num_nodes
        assert \
            last_inference_num_nodes == first_objective_num_nodes == num_nodes, \
            "The number of nodes in the last inference layer [%s] and "\
            "that of the first objective layer [%s]. "\
            "must match num_nodes %s in the network_spec " \
            % (last_inference_num_nodes, first_objective_num_nodes, num_nodes)

        return inference_layers, objective_layers

    @staticmethod
    def _wire_network_layers(
            network_spec: dict,
            inference_layers: List[layer.Layer],
            objective_layers: List[layer.Layer],
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
        name: str = network_spec[_NAME]
        num_nodes: int = network_spec[_NUM_NODES]
        log_level: int = network_spec[_LOG_LEVEL]

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
        # Explicitly set "compose(objective.function o objective.objective)"
        # although it is the same with setting (objective.function)
        layer_inference.objective = \
            compose(layer_objective.function, layer_objective.objective)

        return layer_inference, layer_objective

    @staticmethod
    def build(
            network_spec: Dict,
    ):
        """Build a neural network instance from the specification
        """
        # --------------------------------------------------------------------------------
        # Build layers in the network
        # --------------------------------------------------------------------------------
        inference_layers, objective_layers = \
            SequentialNetwork._build_network_layers(
                network_spec=network_spec
            )

        # --------------------------------------------------------------------------------
        # Wire the layers to function as a network
        # --------------------------------------------------------------------------------
        *layers, = SequentialNetwork._wire_network_layers(
            network_spec=network_spec,
            inference_layers=inference_layers,
            objective_layers=objective_layers,
        )
        layer_inference, layer_objective = layers

        return layer_inference, layer_objective

    # ================================================================================
    # Instance
    # ================================================================================
    @property
    def specification(self) -> Dict:
        """Network specification"""
        return self._specification

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            specification: Dict
    ):
        """Initialize the network instance
        Args:
            specification: Network specification in JSON.
        """
        name, num_nodes, log_level = \
            SequentialNetwork._validate_network_specification(specification)

        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        network_spec = copy.deepcopy(specification)
        self._specification = network_spec

        # --------------------------------------------------------------------------------
        # Build network
        # --------------------------------------------------------------------------------
        self._layer_inference, self._layer_objective = SequentialNetwork.build(
            network_spec=self.specification
        )
        self._layers_all = [self.layer_inference, self.layer_objective]

        # --------------------------------------------------------------------------------
        # Expose the network training functions.
        # --------------------------------------------------------------------------------
        # train() invokes L = objective(function(X))
        self._function: Callable[
            [Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]
        ] = self.layer_inference.function

        self._objective: Callable[
            [Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]
        ] = self.layer_objective.function

        # train() invokes gradient(dL/dL=1.0). Beware gradient is for training.
        # Needs composing gradients of objective and inference layers to let
        # the back propagation flow from the objective to the input.
        self._gradient: Callable[
            [Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]
        ] = compose(self.layer_objective.gradient, self.layer_inference.gradient)

        # --------------------------------------------------------------------------------
        # Expose the network prediction function.
        # --------------------------------------------------------------------------------
        self._predict: Callable[
            [Union[np.ndarray, TYPE_FLOAT]], Union[np.ndarray, TYPE_FLOAT]
        ] = compose(self.layer_inference.predict, self.layer_objective.predict)
