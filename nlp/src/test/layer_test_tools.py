"""Network layer test tools
"""
from typing import (
    List
)
import numpy as np
from common.constants import (
    TYPE_FLOAT
)
from layer.matmul import Matmul


def expected_gradient_from_log_loss(
        P: np.ndarray,
        T: np.ndarray,
        N: int
) -> np.ndarray:
    """Calculate expected back-propagation from a log loss layer.
    L:() = log_loss(P)
    P:(N,M): = activation(Y) where activation is softmax or sigmoid.

    Args:
        P: Probabilities in the log loss layer
        T: Labels in the index format, NOT in OHE format
        N: Batch size
    Returns:

    """
    assert T.shape == (N,)
    # --------------------------------------------------------------------------------
    # EDY: Expected back-propagation E[dL/dY] from the log-loss layer
    # EDY.shape(N,M) = (P-T)/N
    # EDY should match the actual back-propagation dL/dY from log-loss layer.
    # --------------------------------------------------------------------------------
    # (P-T)/N, NOT P/N - T
    EDY = np.copy(P)
    EDY[
        np.arange(N),
        T
    ] -= TYPE_FLOAT(1)
    EDY /= TYPE_FLOAT(N)

    return EDY


def expected_gradients_from_relu_neuron(
        EDA,
        Y,
        matmul
):
    """Calculate expected back-propagations at a Matmul-ReLU neuron
    A = ReLU(Y) of shape (M,D+1)
    Y = Matmul(X) of shape (M,D+1)
    X is input to matmul of shape (N,D+1) with 1 bias column added at Matmul.

    X     -> Matmul -> Y     -> ReLU -> A
    dL/dX <- Matmul <- dY/dL <- ReLU <- dL/dA

    Args:
        EDA: Expected back-propagation E[dL/dA] from the post layer.
        Y: Output of the matmul layer
        matmul: The matmul layer instance of the neuron
    Returns:
        EDY: Expected back-propagation E[dL/dY] from the ReLU to the Matmul layer.
        EDW: Expected gradient E[dL/dW] in the Matmul layer for its weight W.
        EDX: Expected back-propagation E[dL/dX] from the Matmul to the previous layer
    """
    # --------------------------------------------------------------------------------
    # EDY: dL/dY, expected back-propagation from the ReLU to the Matmul layer.
    # EDY = EDA when Y > 0 else 0.
    # This should match the actual back-propagation dL/dY from the ReLU layer.
    # --------------------------------------------------------------------------------
    EDY = np.copy(EDA)
    EDY[Y < 0] = TYPE_FLOAT(0)

    # --------------------------------------------------------------------------------
    # EDW: dL/dW, expected gradient in the Matmul layer for its weight W.
    # EDW:(M,D+1) = matmul.X.T @ EDY.
    # EDW should match the actual dL/dW in [dL/dX, dL/dW] from the matmul.update().
    # --------------------------------------------------------------------------------
    EDW = np.matmul(matmul.X.T, EDY)  # dL/dW.T: [(D+1,N) @ (N,M)].T -> (D+1,M)
    EDW = EDW.T

    # --------------------------------------------------------------------------------
    # EDX: dL/dX, expected back-propagation from the Matmul to the previous layer
    # EDX:(N,M) = EDY @ matmul.W
    # EDA should match the actual back-propagation dL/dX from the Matmul layer.
    # --------------------------------------------------------------------------------
    EDX = np.matmul(EDY, matmul.W)  # dL/dA01: (N,M) @ (M, D+1) -> (N, D+1)

    # --------------------------------------------------------------------------------
    # Remove the bias column added to the matmul input X.
    # EDX.shape(N,M1) without bias to match the Matmul input shape :(N,D)
    # --------------------------------------------------------------------------------
    EDX = EDX[
        ::,
        1::
    ]

    return EDY, EDW, EDX


def expected_gradients_from_relu_neurons(
        EDA,
        sequentials,
        matmul_outputs,
) -> List:
    """Calculate expected back-propagations at sequence of ReLU neurons
    Sequential is (Matmul -> ReLU)
    A = ReLU(Y) of shape (M,D+1)
    Y = Matmul(X) of shape (M,D+1)
    X is input to matmul of shape (N,D+1) with 1 bias column added at Matmul.

    sequentials = [
        (matmul(n-1)-relu(n-1)),
        ...,
        (matmul(i)-relu(i))
        ...,
        (matmul(0)-relu(0))
    ]

    X     -> Matmul -> Y     -> ReLU -> A
    dL/dX <- Matmul <- dY/dL <- ReLU <- dL/dA

    Args:
        EDA: Expected back propagation E[dL/dA] from the post layer
        sequentials: List of sequential layer (Matmul -> ReLU)
        matmul_outputs: List of Matmul output [Y(n-1), ... Y(0)]

    Returns:
        gradients_from_neurons: List of gradients [EDY, EDW, EDX] from each
        (Matmul->ReLU) sequential layer
    """
    assert len(sequentials > 0)
    gradients_from_neurons = []

    eda = EDA
    for sequential, Y in zip(sequentials, matmul_outputs):
        matmul = sequential.layers[0]
        assert isinstance(matmul, Matmul)

        edy, edw, eda = expected_gradients_from_relu_neuron(eda, Y, matmul)
        gradients_from_neurons.append([edy, edw, eda])
    assert (len(gradients_from_neurons) == len(sequentials))
    return gradients_from_neurons
