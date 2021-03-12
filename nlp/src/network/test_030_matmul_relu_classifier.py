"""Matmul layer test cases"""
from typing import (
    List,
    Callable
)
import copy
import logging
import cProfile
import numpy as np
from common import (
    weights,
    softmax,
    relu,
    transform_X_T,
    softmax_cross_entropy_log_loss,
    compose
)
from common.test_config import (
    GRADIENT_DIFF_CHECK_TRIGGER,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE
)
from data import (
    linear_separable_sectors
)
from layer import (
    Matmul,
    Relu,
    CrossEntropyLogLoss
)
from optimizer import (
    Optimizer,
    SGD
)

Logger = logging.getLogger(__name__)


def train_matmul_relu_classifier(
        N: int,
        D: int,
        M: int,
        X: np.ndarray,
        T: np.ndarray,
        W: np.ndarray,
        log_loss_function: Callable,
        optimizer: Optimizer,
        num_epochs: int = 100,
        test_numerical_gradient: bool = True,
        callback: Callable = None
):
    """Test case for binary classification with matmul + log loss.
    Args:
        N: Batch size
        D: Number of features
        M: Number of nodes. 1 for sigmoid and 2 for softmax
        X: train data
        T: labels
        W: weight
        log_loss_function: cross entropy logg loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to run
        test_numerical_gradient: Flag if test the analytical gradient with the numerical one.
        callback: callback function to invoke at the each epoch end.
    """
    name = __name__
    assert isinstance(T, np.ndarray) and np.issubdtype(T.dtype, np.integer) and T.ndim == 1 and T.shape[0] == N
    assert isinstance(X, np.ndarray) and X.dtype == float and X.ndim == 2 and X.shape[0] == N and X.shape[1] == D
    assert isinstance(W, np.ndarray) and W.dtype == float and W.ndim == 2 and W.shape[0] == M and W.shape[1] == D
    assert num_epochs > 0 and N > 0 and D > 0

    assert (
        log_loss_function == softmax_cross_entropy_log_loss and M >= 2
    )

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    loss = CrossEntropyLogLoss(
        name="loss",
        num_nodes=M,
        log_loss_function=log_loss_function,
        log_level=logging.WARNING
    )

    # --------------------------------------------------------------------------------
    # Instantiate a ReLu layer
    # --------------------------------------------------------------------------------
    activation = Relu(
        name="relu",
        num_nodes=M,
        log_level=logging.WARNING
    )
    activation.objective = loss.function

    # --------------------------------------------------------------------------------
    # Instantiate a Matmul layer
    # --------------------------------------------------------------------------------
    matmul = Matmul(
        name="matmul",
        num_nodes=M,
        W=W,
        optimizer=optimizer,
        log_level=logging.WARNING
    )
    matmul.objective = compose(activation.function, loss.function)

    # Network objective function f: L=f(X)
    objective = compose(matmul.function, matmul.objective)

    num_no_progress: int = 0     # how many time when loss L not decreased.
    loss.T = T
    history: List[np.ndarray] = [matmul.objective(matmul.function(X))]

    for i in range(num_epochs):
        # --------------------------------------------------------------------------------
        # Layer forward path
        # 1. Calculate the matmul output Y=matmul.f(X)
        # 2. Calculate the ReLU output A=activation.f(Y)
        # 3. Calculate the loss L = loss(A)
        # Test the numerical gradient dL/dX=matmul.gradient_numerical().
        # --------------------------------------------------------------------------------
        Y = matmul.function(X)
        A = activation.function(Y)
        L = loss.function(A)

        # ********************************************************************************
        # Constraint: Network objective L must match layer-by-layer output
        # ********************************************************************************
        assert L == objective(X) and L.shape == (), \
            f"Network objective L(X) %s must match layer-by-layer output %s." \
            % (objective(X), L)

        print(L)
        Logger.info("%s: iteration[%s]. Loss is [%s]", name, i, L)

        # ********************************************************************************
        # Constraint: Objective/Loss L(Yn+1) after gradient descent < L(Yn)
        # ********************************************************************************
        if L >= history[-1]:
            Logger.warning(
                "Iteration [%i]: Loss[%s] has not improved from the previous [%s].",
                i, L, history[-1]
            )
            if (num_no_progress := num_no_progress+1) > 20:
                Logger.error(
                    "The training has no progress more than %s times.", num_no_progress
                )
                # break
        else:
            num_no_progress = 0

        history.append(L)

        # ================================================================================
        # Layer backward path
        # 1. Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dL/dY.
        # 2. Gradient descent to update Wn+1 = Wn - lr * dL/dX.
        # ================================================================================
        before = copy.deepcopy(matmul.W)
        dA = loss.gradient(float(1))        # dL/dA
        dY = activation.gradient(dA)              # dL/dY
        dX = matmul.gradient(dY)            # dL/dX

        # gradient descent and get the analytical dL/dX, dL/dW
        dS = matmul.update()                # dL/dX, dL/dW

        # ********************************************************************************
        #  Constraint. W in the matmul has been updated by the gradient descent.
        # ********************************************************************************
        Logger.debug("W after is \n%s", matmul.W)
        assert not np.array_equal(before, matmul.W), "W has not been updated."

        # --------------------------------------------------------------------------------
        # Expected dL/dW.T = X.T @ dL/dY = X.T @ (P-T) / N for y > 0 because of ReLU.
        # Expected dL/dX = dL/dY @ W = (P-T) @ W / N for y > 0 or 0 for y <= 0.
        # P = softmax(A)
        # --------------------------------------------------------------------------------
        P = softmax(relu(np.matmul(matmul.X, matmul.W.T)))
        assert P.shape == Y.shape
        # gradient dL/dA = (P-T) from softmax-log-loss
        P[
            np.arange(N),
            T
        ] -= 1
        # dA/dY gradient at ReLU. 1 when y > 0 or 0 otherwise.
        P[(Y <= 0)] = 0                    # Expected dL/dY
        EDX = np.matmul(P/N, matmul.W)        # (N,M) @ (M, D) -> (N, D)
        assert np.array_equal(X, matmul.X)
        EDW = np.matmul(matmul.X.T, P/N).T    # dL/dW.T shape(D,M) -> dL/dW shape(M, D)

        # ********************************************************************************
        # Constraint. Analytical gradients from layer close to expected gradients EDX/EDW.
        # ********************************************************************************
        delta_dX = np.abs(EDX-dS[0])
        delta_dW = np.abs(EDW-dS[1])
        if not (
            np.all(np.abs(EDX) <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.all(delta_dX <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
            np.all(delta_dX <= np.abs(dS[0] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
        ):
            Logger.error("Expected dL/dX \n%s\nDiff\n%s", EDX, EDX-dS[0])
        if not (
            np.all(np.abs(EDW) <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.all(delta_dW <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
            np.all(delta_dW <= np.abs(dS[1] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
        ):
            Logger.error("Expected dL/dW \n%s\nDiff\n%s", EDW, EDW-dS[1])

        if test_numerical_gradient:
            # --------------------------------------------------------------------------------
            # Numerical gradient
            # --------------------------------------------------------------------------------
            gn = matmul.gradient_numerical()

            # ********************************************************************************
            #  Constraint. Numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
            # ********************************************************************************
            delta_GX = np.abs(dS[0] - gn[0])
            if not (
                np.all(delta_GX <= GRADIENT_DIFF_CHECK_TRIGGER) or
                np.all(delta_GX <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
                np.all(delta_GX <= np.abs(gn[0] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
            ):
                Logger.error(
                    "dL/dX analytical gradient \n%s \nneed to close to numerical gradient \n%s\ndifference=\n%s\n",
                    dS[0], gn[0], delta_GX
                )

            delta_GW = np.abs(dS[1] - gn[1])
            if not (
                np.all(delta_GW <= GRADIENT_DIFF_CHECK_TRIGGER) or
                np.all(delta_GW <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
                np.all(delta_GW <= np.abs(gn[1] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
            ):
                Logger.error(
                    "dL/dW analytical gradient \n%s \nneed to close to numerical gradient \n%s\ndifference=\n%s\n",
                    dS[1], gn[1], delta_GW
                )

        if callback:
            # if W.shape[1] == 1 else callback(W=np.average(matmul.W, axis=0))
            callback(W=matmul.W)

    return matmul.W


def test_matmul_relu_classifier(
        M: int = 3
):
    """Test case for layer matmul class
    """
    N = 50
    D = 3
    W = weights.he(M, D)
    optimizer = SGD(lr=0.1)
    X, T, V = linear_separable_sectors(n=N, d=D, m=M)
    assert X.shape == (N, D)
    X, T = transform_X_T(X, T)

    def callback(W):
        """Dummy callback"""
        W

    profiler = cProfile.Profile()
    profiler.enable()

    train_matmul_relu_classifier(
        N=N,
        D=D,
        M=M,
        X=X,
        T=T,
        W=W,
        log_loss_function=softmax_cross_entropy_log_loss,
        optimizer=optimizer,
        callback=callback
    )

    profiler.disable()
    profiler.print_stats(sort="cumtime")
