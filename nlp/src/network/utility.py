from typing import (
    List,
    Dict
)
import logging

from layer import (
    Standardization,
    BatchNormalization,
    Matmul,
    ReLU,
    Sigmoid,
    CrossEntropyLogLoss
)
from layer.constants import (
    _NAME,
    _NUM_NODES,
    _COMPOSITE_LAYER_SPEC,
    _LOG_LEVEL
)
import optimizer


def multilayer_network_specification(
        num_features: List[int],
        use_input_standardization: bool = True,
        activation: str = ReLU.class_id()
):
    """
    Responsibility:
        Generate multi-layer network specification whose structure is:
        [standardization] -> [matmul-bn-activation]+ -> [matmul-loss].
        where [...] is a stack. 

    Args:
        num_features:
            number of features. e.g. [D, M01, M02, M]

            D: the number of input features into the network.
            M01: the number of outputs from the first [matmul-bn-activation],
                 which are the number of input features into the next layer.
            M02: the number of outputs from the second [matmul-bn-activation],
                 which are the number of input features into the next layer.
            M: the number of outputs, or classes to predict by the network.

            X:(N,D) -> [matmul-bn-activation] -> A01:(N,M01)
            A01:(N,M01) -> [matmul-bn-activation] -> A02:(N, M02)
            A02:(N, M02) -> [matmul -> Y:(N,M) -> loss -> ()].

        use_input_standardization: flag to use the input standardization
        activation: Activation class to use.
    """
    # At least 3 stack: m0:(m1,m0)->(m) where m is the number of outputs at the last layer
    # m0->[matmul:(m1,m0)-bn(m1)-relu(m1)]->[matmul(m,m1)-bn(m)-loss(m)]
    # m0 is the 1st input features such as (28*28) MNIST image features.
    assert len(num_features) >= 3

    # Number of inputs to the network
    D = num_features[0]

    # Number of outputs from the network
    M = num_features[-1]

    def input(index: int, d: int) -> Dict[str, dict]:
        """Build input normalization/standardization layer specification
        Args:
            index: stack position in the network
            d: number of features in the input
        """
        return {
            f"std{index:03d}": Standardization.specification(
                name=f"std{index:03d}",
                num_nodes=d,
                momentum=0.9
            ),
        }

    def inference(index: int, m: int, d: int) -> Dict[str, dict]:
        """Build matmul-bn-activation specifications
        Args:
            index: stack position in the network
            m: number of outputs (== number of nodes)
            d: number of features in the input
        """
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
            f"activation{index:03d}": ReLU.specification(
                name=f"relu{index:03d}",
                num_nodes=m,
            )
            if activation == ReLU.class_id()
            else Sigmoid.specification(
                name=f"sigmoid{index:03d}",
                num_nodes=m,
            )

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
            "loss": CrossEntropyLogLoss.specification(
                name="loss", num_nodes=m
            )

        }

    sequential_layer_specification = {}

    # --------------------------------------------------------------------------------
    # Standardization as the first stack
    # --------------------------------------------------------------------------------
    if use_input_standardization:
        for k, v in input(index=0, d=D).items():
            sequential_layer_specification[k] = v

    # --------------------------------------------------------------------------------
    # Inference stacks
    # --------------------------------------------------------------------------------
    # Matmul W.shape == (D:dimension, M:number of nodes in a layer)
    # Matmul layer i outputs Yi:(Ni,Mi), which is the input Xi+1:(Ni+1, Di+1) to the
    # next layer i+1. Hence Mi == Di+1.
    # --------------------------------------------------------------------------------
    M_D = [     # Create W:(M, D) at the matmul layer in a stack
        (num_features[index + 1], num_features[index])
        for index in range(len(num_features)-2)
    ]
    num_features_into_last_stack = 0
    for index, (m, d) in enumerate(M_D, start=1):
        num_features_into_last_stack = m
        for k, v in inference(index=index, m=m, d=d).items():
            sequential_layer_specification[k] = v

    # --------------------------------------------------------------------------------
    # Output stack
    # --------------------------------------------------------------------------------
    # Number of output from the inference into the last layer
    assert num_features_into_last_stack == num_features[-2]
    for k, v in output(m=M, d=num_features_into_last_stack).items():
        sequential_layer_specification[k] = v

    # import json
    # print(json.dumps(sequential_layer_specification, indent=4))

    return {
        _NAME: "multilayer_network",
        _NUM_NODES: M,
        _LOG_LEVEL: logging.ERROR,
        _COMPOSITE_LAYER_SPEC: sequential_layer_specification
    }


def test():
    num_nodes = [8, 32, 32, 32, 3]
    multilayer_network_specification(num_nodes)