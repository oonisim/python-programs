import logging

from layer import (
    BatchNormalization,
    Matmul,
    ReLU,
    CrossEntropyLogLoss
)
from layer.constants import (
    _NAME,
    _NUM_NODES,
    _COMPOSITE_LAYER_SPEC,
    _LOG_LEVEL
)
import optimizer


def multilayer_network_specification(num_nodes):
    """
    Responsibility:
        Generate multi-layer network specification
    """
    # At least 3: 0:(m1,m0)->(m) where m is the number of outputs at the last layer
    # 0:(matmul:(m1,m0)-bn(m1)-relu(m1))-(matmul(m,m1)-bn(m)-loss(m)
    assert len(num_nodes) >= 3

    # Number of output from the network
    M = num_nodes[-1]

    def inference(index, m, d):
        return {
            f"matmul{index:03d}": Matmul.specification(
                name=f"matmul{index:03d}",
                num_nodes=m,
                num_features=d,
                weights_initialization_scheme="he",
                weights_optimizer_specification=optimizer.SGD.specification(
                    lr=0.05,
                    l2=1e-3
                )
            ),
            f"bn{index:03d}": BatchNormalization.specification(
                name=f"bn{index:03d}",
                num_nodes=m,
                gamma_optimizer_specification=optimizer.SGD.specification(
                    lr=0.05,
                    l2=1e-3
                ),
                beta_optimizer_specification=optimizer.SGD.specification(
                    lr=0.05,
                    l2=1e-3,
                ),
                momentum=0.9
            ),
            f"relu{index:03d}": ReLU.specification(
                name=f"relu{index:03d}",
                num_nodes=m,
            ),

        }

    def output(m, d):
        return {
            "matmul": Matmul.specification(
                name="matmul",
                num_nodes=m,
                num_features=d,
                weights_initialization_scheme="he",
                weights_optimizer_specification=optimizer.SGD.specification(
                    lr=0.05,
                    l2=1e-3
                )
            ),
            "bn": BatchNormalization.specification(
                name="bn",
                num_nodes=m,
                gamma_optimizer_specification=optimizer.SGD.specification(
                    lr=0.05,
                    l2=1e-3
                ),
                beta_optimizer_specification=optimizer.SGD.specification(
                    lr=0.05,
                    l2=1e-3,
                ),
                momentum=0.9
            ),
            "loss": CrossEntropyLogLoss.specification(
                name="loss", num_nodes=m
            )

        }

    sequential_layer_specification = {}
    D_M = [
        (num_nodes[index], num_nodes[index + 1])
        for index in range(len(num_nodes)-2)
    ]
    for index, (d,m) in enumerate(D_M):
        for k, v in inference(index=index, m=m, d=d).items():
            sequential_layer_specification[k] = v

    # Number of output from the inference lat layer
    D = m
    assert D == num_nodes[-2]
    for k, v in output(m=M, d=D).items():
        sequential_layer_specification[k] = v

    #import json
    #print(json.dumps(sequential_layer_specification, indent=4))

    return {
        _NAME: "multilayer_network",
        _NUM_NODES: num_nodes[-1],
        _LOG_LEVEL: logging.ERROR,
        _COMPOSITE_LAYER_SPEC: sequential_layer_specification
    }


def test():
    num_nodes = [8, 32, 32, 32, 3]
    multilayer_network_specification(num_nodes)