from typing import (
    Dict
)
import copy
import logging
import numpy as np
import common
from common import (
    functions,
    weights
)
import layer
import optimizer as optimizer


network_spec = {
    "matmul01": {
        "type": layer.Matmul.__qualname__,
        "num_nodes": 8,
        "num_features": 3,
        "weights": {
            "scheme": "he"
        }
    },
    "activation01": {
        "type": layer.ReLU.__qualname__,
        "num_nodes": 8
    },
    "matmul02": {
        "type": layer.Matmul.__qualname__,
        "num_nodes": 3,
        "num_features": 3,
        "weights": {
            "scheme": "he"
        }
    },
    "activation02": {
        "type": layer.ReLU.__qualname__,
        "num_nodes": 3
    },
    "objective": {
        "type": layer.CrossEntropyLogLoss.__qualname__,
        "num_nodes": 3,
        "loss_function": "softmax_cross_entropy_log_loss"
    }
}


def build_log_loss_function():
    log_loss_function = common.softmax_cross_entropy_log_loss
    return log_loss_function


def build_log_loss(specification: Dict):
    spec = specification
    assert (
        "num_nodes" in spec and
        "loss_function" in spec and
        spec["loss_function"] in functions.LOSS_FUNCTIONS
    )

    return layer.CrossEntropyLogLoss(
        name=spec["name"],
        num_nodes=spec["num_nodes"],
        log_loss_function=functions.LOSS_FUNCTIONS[spec["loss_function"]],
        log_level=spec["log_level"] if "log_level" in spec else logging.ERROR
    )


def build_relu(specification: Dict):
    assert "num_nodes" in specification
    spec = copy.deepcopy(specification)

    return layer.ReLU(
        name="relu01",
        num_nodes=spec["num_nodes"],
        log_level=spec["log_level"] if "log_level" in spec else logging.ERROR
    )


def build_weight(specification: Dict) -> np:
    """Build layer.Matmul weights
    Args:
        specification: weight specification
    Return: initialized weight

    specification = {
        "scheme": "weight initialization scheme [he|xavier]"
        "num_nodes": "number of nodes M",
        "num_feature": "number of features in a batch X including bias"
    }
    """
    spec = specification
    assert (
        "num_nodes" in spec and
        "num_features" in spec["num_features"]
    )

    M = spec["num_nodes"]
    D = spec["num_features"]
    scheme = spec["scheme"] if "scheme" in spec else "uniform"
    assert scheme in weights.INITIALIZATION_SCHEMES
    return weights.INITIALIZATION_SCHEMES[scheme](M, D)


def build_optimizer(specification: Dict) -> optimizer.Optimizer:
    spec = specification
    if "scheme" in spec["optimizer"]:
        scheme = spec["optimizer"]["scheme"]
        assert scheme in optimizer.SCHEMES
        __optimizer = optimizer.SCHEMES[scheme]()
    else:
        __optimizer = optimizer.SGD()

    return __optimizer


def build_matmul(specification: Dict):
    spec = copy.deepcopy(specification)
    assert (
        "num_nodes" in spec and
        "num_features" in spec["num_features"] and
        "weights" in spec
    )

    # Geometry
    num_nodes = spec["num_nodes"]
    num_features = spec["num_features"]

    # Weights
    weight_spec = spec["weights"]
    weight_spec["num_nodes"] = num_nodes
    weight_spec["num_features"] = num_features
    W = build_weight(weight_spec)

    # Optimizer
    if "optimizer" in spec:
        __optimizer = build_optimizer(spec["optimizer"])
    else:
        spec["optimizer"] = {}
        __optimizer = build_optimizer(spec["optimizer"])

    matmul = layer.Matmul(
        name=spec["name"],
        num_nodes=num_nodes,
        W=W,
        optimizer=__optimizer,
        log_level=spec["log_level"] if "log_level" in spec else logging.ERROR
    )

    return matmul


NETWORK_COMPONENT_BUILDERS = {
    "matmul": build_matmul,
    "relu": build_relu,
    "crossentropylogloss": build_log_loss
}


def build_network(specification: Dict):
    layers = []
    objectives = []

    spec = copy.deepcopy(specification)
    for key, item in spec.items():
        # Network layer
        if "type" in item:
            assert item["type"].lower() in NETWORK_COMPONENT_BUILDERS
            if "name" not in item:
                item["name"] = str(key)

            component = NETWORK_COMPONENT_BUILDERS[item["type"].lower()]
            if isinstance(component, layer.CrossEntropyLogLoss):
                objectives.append(component)
            elif isinstance(component, layer.Layer):
                layers.append(component)
            else:
                assert False, "Invalid network component"

    assert layers, "Expected layers but none"
    assert objectives, "Expected objectives but none"
    return [layers, objectives]
