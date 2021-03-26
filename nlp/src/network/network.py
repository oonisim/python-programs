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
from common.functions import (
    compose
)
import layer
import optimizer as optimiser


# From Python 3.6 onwards, the standard dict type maintains
# insertion order by default.
specification_template = {
    "name": "network",
    "log_level": logging.ERROR,
    "matmul01": {
        "scheme": layer.Matmul.__qualname__,
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
    "activation01": {
        "scheme": layer.ReLU.__qualname__,
        "num_nodes": 8
    },
    "matmul02": {
        "scheme": layer.Matmul.__qualname__,
        "num_nodes": 3,
        "num_features": 3,
        "weights": {
            "scheme": "he"
        }
    },
    "activation02": {
        "scheme": layer.ReLU.__qualname__,
        "num_nodes": 3
    },
    "objective": {
        "scheme": layer.CrossEntropyLogLoss.__qualname__,
        "num_nodes": 3,
        "loss_function": "softmax_cross_entropy_log_loss"
    }
}


class Network:
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
    def build(
            specification: Dict
    ) -> Tuple[List[layer.Layer], List[layer.Layer]]:
        """Build a neural network instance from the specification
        """
        layers = []
        objectives = []
        network_spec = copy.deepcopy(specification)

        # Default log level
        if "log_level" not in network_spec:
            network_spec["log_level"] = logging.ERROR

        # Network layers
        for name, layer_spec in network_spec.items():
            # Network layers
            if "layer_scheme" in layer_spec:
                layer_scheme = layer_spec["layer_scheme"].lower()
                layer_class = layer.SCHEMES[layer_scheme]

                # Key is the name
                if "name" not in layer_spec:
                    layer_spec["name"] = name

                # Cascade down the network log level if layer has no level.
                if "log_level" not in layer_spec:
                    layer_spec["log_level"] = network_spec["log_level"]

                __layer = layer_class.build(layer_spec)
                if isinstance(__layer, layer.OBJECTIVE_LAYERS):
                    objectives.append(__layer)
                elif isinstance(__layer, layer.FUNCTION_LAYERS):
                    layers.append(__layer)
                else:
                    assert False, "Invalid network component"

        assert layers, "Expected layers but none"
        assert objectives, "Expected objectives but none"
        return layers, objectives

    # ================================================================================
    # Instance
    # ================================================================================
    @property
    def name(self):
        return self._name

    @property
    def layers(self):
        return self._layers

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(self, specification):
        """Initialize the network instance
        Args:
            specification: Network specificaiton.
            See network.specification_template
        """
        def __assertion(spec: Dict):
            assert isinstance(spec, dict) and len(spec) > 0
            assert "name" in spec

        spec = specification
        __assertion(spec)

        self._specification = spec
        self._name = spec['name']
        self._layers, _objectives = Network.build(spec)

        # --------------------------------------------------------------------------------
        # Network inference layer
        # --------------------------------------------------------------------------------
        _layer_inference = layer.Sequential(
            name=spec["name"],
            num_nodes=self.layers[0].num_nodes,
            layers=self._layers,
            log_level=spec["log_level"] if "log_level" in spec else logging.ERROR
        )
        _layer_inference.objective = compose(*[o.function for o in _objectives])
        self._layer_inference = _layer_inference

        # --------------------------------------------------------------------------------
        # Network function
        # --------------------------------------------------------------------------------
        self.function: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]] = \
            _layer_inference.function

        # --------------------------------------------------------------------------------
        # Network prediction function
        # --------------------------------------------------------------------------------
        self.predict: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]] = \
            _layer_inference.predict

        # --------------------------------------------------------------------------------
        # Network objective function
        # --------------------------------------------------------------------------------
        self.objective: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]] = \
            compose(_layer_inference.function, _layer_inference.objective)
