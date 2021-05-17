"""
Neural network implementation
"""
import logging
from typing import (
    Dict,
    List,
    Tuple,
    Union,
    Callable
)

import numpy as np

import layer
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
    LAYER_MAX_NUM_NODES
)
from common.function import (
    compose
)
from layer.constants import (
    _NAME,
    _NUM_NODES,
    _LOG_LEVEL,
    _COMPOSITE_LAYER_SPEC
)
from layer.utility_builder_layer import (
    build_layers_from_composite_layer_specification
)
from network.base import Network
from network.utility import (
    multilayer_network_specification
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
    # --------------------------------------------------------------------------------
    # Specification
    # --------------------------------------------------------------------------------
    @staticmethod
    def specification_template():
        """
        Specification template
        """
        M = TYPE_INT(3)
        D = TYPE_INT(2)
        M01 = M
        return SequentialNetwork.specification(
            num_features=[D, M01, M],
            activation_scheme=layer.ReLU.class_id()
        )

    @staticmethod
    def specification(
            num_features: List[TYPE_INT],
            use_input_standardization=True,
            activation_scheme: str = layer.ReLU.class_id(),
            weights_initialization_scheme: str = "uniform",
            weights_optimizer_specification: dict = None
    ):
        """Generate network specification
        TODO:
            Use weights_initialization_scheme
            Use weights_optimizer_specification

        Args:
            num_features: number of features. e.g. [D, M01, M02, M]
            use_input_standardization: flag to use the input standardization
            activation_scheme: Activation function to use e.g. ReLU
            weights_initialization_scheme: weight initialization scheme e.g. he
            weights_optimizer_specification: optimizer specification.
                                             Default to SGD if omitted
        """
        specification = multilayer_network_specification(
            num_features=num_features,
            use_input_standardization=use_input_standardization,
            activation=activation_scheme
        )
        return specification

    # --------------------------------------------------------------------------------
    # Factory
    # --------------------------------------------------------------------------------
    @staticmethod
    def _validate_network_specification(specification: dict):
        """Validate if the neural network specification has mandatory parameters
        Args:
            specification: Neural network specification
        """
        assert \
            isinstance(specification, dict) and \
            len(specification) > 0

        assert \
            _NAME in specification and \
            isinstance(specification[_NAME], str) and \
            len(specification[_NAME]) > 0
        name = specification[_NAME].lower()

        assert \
            _NUM_NODES in specification and \
            isinstance(specification[_NUM_NODES], int) and \
            0 < specification[_NUM_NODES] < LAYER_MAX_NUM_NODES, \
            "0 < num_nodes %s required" % LAYER_MAX_NUM_NODES
        num_nodes = specification[_NUM_NODES]

        assert \
            _COMPOSITE_LAYER_SPEC in specification and \
            isinstance(specification[_COMPOSITE_LAYER_SPEC], dict) and \
            len(specification[_COMPOSITE_LAYER_SPEC]) > 0, \
            "Invalid %s\n%s\n" % (_COMPOSITE_LAYER_SPEC, specification)

        log_level = specification[_LOG_LEVEL] \
            if _LOG_LEVEL in specification else logging.ERROR

        return name, num_nodes, log_level

    @staticmethod
    def _build_network_layers(
            specification: dict,
    ) -> Tuple[List[layer.Layer], List[layer.Layer]]:
        """
        Responsibility
            1. Instantiate network layers
            2. Classify them into inference and objective
        Args:

        Returns:
            (inference_layers, objective_layers)
        """
        *parameters, = SequentialNetwork._validate_network_specification(specification)
        name, num_nodes, log_level = parameters
        inference_layers: List[layer.Layer] = []
        objective_layers: List[layer.Layer] = []

        layers = build_layers_from_composite_layer_specification(
            specification=specification[_COMPOSITE_LAYER_SPEC],
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
            "must match num_nodes %s in the specification " \
            % (last_inference_num_nodes, first_objective_num_nodes, num_nodes)

        return inference_layers, objective_layers

    @staticmethod
    def _wire_network_layers(
            name: str,
            num_nodes: int,
            inference_layers: List[layer.Layer],
            objective_layers: List[layer.Layer],
            log_level: int = logging.ERROR
    ) -> Tuple[layer.Sequential, layer.Sequential]:
        """
        Responsibility:
            Wire the layers to function as a network.

        Args:
            inference_layers: layer instances to produce an inference.
            objective_layers: layer to produce the objective e.g. the loss.

        Returns:
            (layer_inference, layer_objective) where layer_inference and
            layer_objective are layer.Sequential class instances.
        """
        def identity(x: np.ndarray):
            assert x.ndim == 0, "The output of the log loss should be of shape ()"
            return x

        layer_objective = layer.Sequential(
            name=name + "_objective",
            num_nodes=num_nodes,
            layers=list(filter(lambda x: x is not None, objective_layers)),
            log_level=log_level
        )
        # The objective function of the layer.Sequential class is initialized
        # upon its setter call. Hence it is required to set the objective
        # function although the objective layer itself has its own objective
        # function setup.
        layer_objective.objective = identity
        layer_inference = layer.Sequential(
            name=name + "_inference",
            num_nodes=num_nodes,
            layers=list(filter(lambda x: x is not None, inference_layers)),
            log_level=log_level
        )
        # Explicitly set "compose(objective.function o objective.objective)"
        # although it is actually the same with setting (objective.function)
        # because:
        # compose(layer_objective.function, layer_objective.objective) ->
        # compose (layer_objective.function, identity) ->
        # layer.objective.function
        layer_inference.objective = \
            compose(layer_objective.function, layer_objective.objective)

        return layer_inference, layer_objective

    @staticmethod
    def build(
            specification: Dict,
    ):
        """Build a neural network instance from the specification
        Args:
            specification: Network specification in JSON.
        """
        *parameters, = SequentialNetwork._validate_network_specification(specification)
        name, num_nodes, log_level = parameters

        # --------------------------------------------------------------------------------
        # Build layers in the network
        # --------------------------------------------------------------------------------
        inference_layers, objective_layers = SequentialNetwork._build_network_layers(
            specification=specification
        )

        return SequentialNetwork(
            name=name,
            num_nodes=num_nodes,
            inference_layers=inference_layers,
            objective_layers=objective_layers,
            log_level=log_level
        )

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
            inference_layers: List[layer.Layer],
            objective_layers: List[layer.Layer],
            log_level: int = logging.ERROR
    ):
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        # --------------------------------------------------------------------------------
        # Wire the layers to function as a network
        # --------------------------------------------------------------------------------
        *layers, = SequentialNetwork._wire_network_layers(
            name=name,
            num_nodes=num_nodes,
            inference_layers=inference_layers,
            objective_layers=objective_layers,
            log_level=log_level
        )
        self._layer_inference, self._layer_objective = layers
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
