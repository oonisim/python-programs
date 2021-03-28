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
from common.functions import (
    compose
)
import layer
import optimizer as optimiser
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
    _LOG_LEVEL
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
    def place_layer_spec(
            container: dict,
            layer_name: str,
            layer_scheme: str,
            layer_spec: dict,
            schemes: dict,
            log_level: int
    ):
        """Generate a specification to instantiate a Layer e.g. Matmul
        The spec format is {
            _LAYER_CLASS: "<class> e.g Matmul",
            _PARAMETERS: {
                "name": "<layer name>",
                "num_nodes": <number of nodes in the layer>,
                "log_level": <log level>,
                <other mandatory arguments for the layer>
            }
        }
        """
        # --------------------------------------------------------------------------------
        # Detect duplicated layer names (unique in inference and unique in objective)
        # _LAYER_CLASS cannot be used as the layer name.
        # --------------------------------------------------------------------------------
        assert not (
                layer_name in container or
                layer_name == _LAYER_CLASS
        ), "Cannot have duplicate name [%s] or [%s] as a dictionary key.\nspec is %s." \
           % (layer_name, _LAYER_CLASS, layer_spec)

        container[layer_name] = {}

        # --------------------------------------------------------------------------------
        # Layer class to instantiate
        # --------------------------------------------------------------------------------
        container[layer_name][_LAYER_CLASS] = schemes[layer_scheme]

        # --------------------------------------------------------------------------------
        # Arguments to pass to the layer class __init__() method
        # --------------------------------------------------------------------------------
        assert _PARAMETERS in layer_spec, \
            "%s is mandatory in a layer specification.\nspec is %s" \
            % (_PARAMETERS, layer_spec)

        args = copy.deepcopy(layer_spec[_PARAMETERS]) \

        # Mandatory "name" arg. Use the layer_spec key when not specified
        args[_NAME] = layer_name \
            if _NAME not in args else args[_NAME].lower()

        # Mandatory "num_nodes" arg.
        assert _NUM_NODES in args, \
            "%s is mandatory in a layer specification.\nspec is %s" \
            % (_NUM_NODES, layer_spec)

        # Logging default level
        args[_LOG_LEVEL] = log_level \
            if _LOG_LEVEL not in args else args[_LOG_LEVEL]

        container[layer_name][_PARAMETERS] = args

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
        inference_layer_specs = {}
        objective_layer_specs = {}
        for layer_name, layer_spec in network_spec.items():
            assert isinstance(layer_name, str) and isinstance(layer_spec, dict)
            if _SCHEME in layer_spec:
                layer_name = layer_name.lower()
                layer_scheme = layer_spec[_SCHEME].lower()     # Layer class, e.g. Matmul

                # --------------------------------------------------------------------------------
                # Compose a layer spec
                # --------------------------------------------------------------------------------
                if layer_scheme in layer.FUNCTION_LAYER_SCHEMES:
                    target_layer_specs = inference_layer_specs
                    schemes = layer.FUNCTION_LAYER_SCHEMES
                elif layer_scheme in layer.OBJECTIVE_LAYER_SCHEMES:
                    target_layer_specs = objective_layer_specs
                    schemes = layer.OBJECTIVE_LAYER_SCHEMES
                else:
                    assert False, \
                        "Invalid layer %s. Must be one of %s." \
                        % (layer_scheme, layer.SCHEMES.keys())

                SequentialNetwork.place_layer_spec(
                    container=target_layer_specs,
                    layer_name=layer_name,
                    layer_scheme=layer_scheme,
                    layer_spec=layer_spec,
                    schemes=schemes,
                    log_level=log_level
                )
                assert set(target_layer_specs[layer_name].keys()) == \
                       {_LAYER_CLASS, _PARAMETERS}

        assert \
            len(inference_layer_specs) > 0 and \
            len(objective_layer_specs) > 0, \
            "There must be at least 1 inference and 1 objective layer. " \
            "Inference count %s objective count is %s" \
            % (len(inference_layer_specs), len(objective_layer_specs))

        assert \
            list(inference_layer_specs.values())[-1][_PARAMETERS][_NUM_NODES] == \
            list(objective_layer_specs.values())[0][_PARAMETERS][_NUM_NODES] == \
            num_nodes, \
            "The number of nodes in the last inference layer [%s] "\
            "must match that of the first objective layer [%s]." \
            % (
                inference_layer_specs[-1][_PARAMETERS][_NUM_NODES],
                objective_layer_specs[0][_PARAMETERS][_NUM_NODES]
            )

        return inference_layer_specs, objective_layer_specs

    @staticmethod
    def _build_network_layers(
            layer_specs: Dict
    ) -> List[layer.Layer]:
        return [
            spec[_LAYER_CLASS].build(spec[_PARAMETERS])
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
            omit_last_activation_for_prediction=True,
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
            name: str,
            num_nodes: int,
            network_spec: Dict,
            log_level: int
    ):
        """Build a neural network instance from the specification
        """
        # --------------------------------------------------------------------------------
        # Validate the specification and generate specifications respectively for
        # - inference
        # - objective
        # --------------------------------------------------------------------------------
        inference_layer_specs, objective_layer_specs = \
            SequentialNetwork._parse_network_specs(
                network_spec=network_spec,
                num_nodes=num_nodes,
                log_level=log_level
            )

        # --------------------------------------------------------------------------------
        # Build a sequential inference layer and a objective layer
        # --------------------------------------------------------------------------------
        inference_layers: List[layer.Layer] = SequentialNetwork._build_network_layers(
            inference_layer_specs
        )
        objective_layers: List[layer.Layer] = SequentialNetwork._build_network_layers(
            objective_layer_specs
        )

        # --------------------------------------------------------------------------------
        # Wire the layers to function as a network
        # --------------------------------------------------------------------------------
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
        assert isinstance(name, str) and len(name) > 0
        assert isinstance(specification, dict) and len(specification) > 0
        assert 0 < num_nodes < LAYER_MAX_NUM_NODES, \
            "0 < num_nodes %s required" % LAYER_MAX_NUM_NODES

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
        ] = self.layer_inference.predict
