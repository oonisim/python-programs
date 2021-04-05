"""Network layer test tools
"""
from typing import (
    List,
    Union,
    Callable
)
import copy
import logging
import numpy as np
from common.constants import (
    TYPE_FLOAT
)
from common.function import (
    softmax,
    relu,
    compose
)
from layer import (
    Matmul,
    ReLU,
    CrossEntropyLogLoss
)
from layer.utility import (
    forward_outputs,
    backward_outputs,
)
from test.config import (
    GRADIENT_DIFF_CHECK_TRIGGER,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE,
    ENFORCE_STRICT_ASSERT,
)


Logger = logging.getLogger(__name__)


def validate_against_numerical_gradient(dS: List[np.ndarray], gn: List[np.ndarray], logger: logging.Logger):
    assert len(dS) == len(gn)
    for ds, gn in zip(dS, gn):
        if not (
                np.all(gn <= GRADIENT_DIFF_CHECK_TRIGGER) or
                np.allclose(
                    a=ds[gn != 0],  # dL/dX: (N,M1)
                    b=gn[gn != 0],  # dL/dX: (N,M1)
                    atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
                    rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
                )
        ):
            logger.error(
                "Need similar analytical and numerical ds. \n"
                "Analytical=\n%s\nNumerical\n%s\ndifference=\n%s\n",
                ds, gn, (ds - gn)
            )
            assert ENFORCE_STRICT_ASSERT


def validate_against_expected_gradient(
        expected: np.ndarray,
        actual: np.ndarray
) -> bool:
    return \
        np.all(np.abs(expected) <= GRADIENT_DIFF_CHECK_TRIGGER) or \
        np.allclose(
            a=actual,
            b=expected,
            atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
            rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
        )


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


def validate_relu_neuron_round_trip(
        matmul: Matmul,
        activation: ReLU,
        X: np.ndarray,
        dA: np.ndarray
):
    """Validate the expected (loss, gradients) against the actual
    DO NOT run gradient descent here.
    Forward path:
        Y = matmul.function(X)
        A = ReLU.function(Y)

    Expected gradients:
        dL/dA = Back propagation from the post layer
        dL/dY = dL/dA if Y > 0 else 0
        dL/dX = (P-T) @ W if Y > 0 else 0
        dL/dW.T = X.T @ (P-T)

    Backward path:
        dL/dY:
        dL/dX:
    """
    N = X.shape[0]
    layers = [matmul, activation]

    # --------------------------------------------------------------------------------
    # Layer forward path
    # --------------------------------------------------------------------------------
    Y, A = forward_outputs(layers, X)

    # --------------------------------------------------------------------------------
    # Expected gradients
    # --------------------------------------------------------------------------------
    EDY, EDW, EDX = expected_gradients_from_relu_neuron(dA, Y, matmul)

    # ================================================================================
    # Layer backward path
    # 1. Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dL/dY.
    # 2. Gradient descent to update Wn+1 = Wn - lr * dL/dX.
    # ================================================================================
    dY, dX = backward_outputs(layers, dA)
    dW = matmul.dW

    # ********************************************************************************
    # Constraint. Analytical gradients from layer close to expected gradients EDX/EDW.
    # ********************************************************************************
    if not validate_against_expected_gradient(dY, EDY):
        Logger.error("Expected dL/dY \n%s\nDiff\n%s", EDY, (EDY-dY))
    if not validate_against_expected_gradient(dX, EDX):
        Logger.error("Expected dL/dX \n%s\nDiff\n%s", EDX, (EDX-dX))
    if not validate_against_expected_gradient(dW, EDW):
        Logger.error("Expected dL/dW \n%s\nDiff\n%s", EDW, (EDW-dW))

    return Y, A, EDY, EDW, EDX, dY, dX, dW


def validate_relu_neuron_training(
        matmul: Matmul,
        activation: ReLU,
        loss: CrossEntropyLogLoss,
        X: np.ndarray,
        T: np.ndarray,
        num_epochs: int = 100,
        test_numerical_gradient: bool = False,
        callback: Callable = None
):
    activation.objective = loss.function
    matmul.objective = compose(activation.function, loss.function)
    objective = compose(matmul.function, matmul.objective)

    num_no_progress: int = 0  # how many time when loss L not decreased.
    history: List[np.ndarray] = []

    loss.T = T
    for i in range(num_epochs):
        L = objective(X)
        N = X.shape[0]
        P = softmax(relu(np.matmul(matmul.X, matmul.W.T)))
        EDA = expected_gradient_from_log_loss(P=P, T=T, N=N)

        # ********************************************************************************
        # Constraint: Expected gradients must match actual
        # ********************************************************************************
        validate_relu_neuron_round_trip(
            matmul=matmul,
            activation=activation,
            X=X,
            dA=EDA
        )

        # --------------------------------------------------------------------------------
        # gradient descent and get the analytical dL/dX, dL/dW
        # --------------------------------------------------------------------------------
        previous_W = copy.deepcopy(matmul.W)
        matmul.update()  # dL/dX, dL/dW

        # ********************************************************************************
        #  Constraint. W in the matmul has been updated by the gradient descent.
        # ********************************************************************************
        Logger.debug("W after is \n%s", matmul.W)
        if np.array_equal(previous_W, matmul.W):
            Logger.warning("W has not been updated")

        # ********************************************************************************
        # Constraint: Objective/Loss L(Yn+1) after gradient descent < L(Yn)
        # ********************************************************************************
        if i > 0 and L >= history[-1]:
            Logger.warning(
                "Iteration [%i]: Loss[%s] has not improved from the previous [%s] for %s times.",
                i, L, history[-1], num_no_progress + 1
            )
            # --------------------------------------------------------------------------------
            # Reduce the learning rate can make the situation worse.
            # When reduced the lr every time L >= history, the (L >= history) became successive
            # and eventually exceeded 50 successive non-improvement ending in failure.
            # Keep the learning rate make the L>=history more frequent but still up to 3
            # successive events, and the training still kept progressing.
            # --------------------------------------------------------------------------------
            num_no_progress += 1
            if num_no_progress > 5:
                matmul.lr = matmul.lr * 0.95

            if num_no_progress > 50:
                Logger.error(
                    "The training has no progress more than %s times.", num_no_progress
                )
                break
        else:
            num_no_progress = 0

        history.append(L)

        if callback:
            callback(W=matmul.W)

    return history
