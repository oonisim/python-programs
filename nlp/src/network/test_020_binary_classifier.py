"""Matmul layer test cases"""
import cProfile
import copy
import logging
from typing import (
    Union,
    List,
    Callable
)

import numpy as np
import pytest_check as check    # https://pypi.org/project/pytest-check/
from common import (
    weights,
    softmax,
    transform_X_T,
    sigmoid,
    sigmoid_cross_entropy_log_loss,
    softmax_cross_entropy_log_loss
)
from data.classifications import (
    linear_separable
)
from layer import (
    Matmul,
    CrossEntropyLogLoss
)
from optimizer import (
    Optimizer,
    SGD
)
from common.test_config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    MAX_ACTIVATION_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE
)


Logger = logging.getLogger(__name__)
Logger.setLevel(logging.DEBUG)


def train_binary_classifier(
        N: int,
        D: int,
        M: int,
        X: np.ndarray,
        T: np.ndarray,
        W: np.ndarray,
        log_loss_function: Callable,
        optimizer: Optimizer,
        num_epochs: int = 100,
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
        callback: callback function to invoke at the each epoch end.
    """
    name = __name__
    assert isinstance(T, np.ndarray) and T.dtype == int and T.ndim == 1 and T.shape[0] == N
    assert isinstance(X, np.ndarray) and X.dtype == float and X.ndim == 2 and X.shape[0] == N and X.shape[1] == D
    assert isinstance(W, np.ndarray) and W.dtype == float and W.ndim == 2 and W.shape[0] == M and W.shape[1] == D
    assert num_epochs > 0 and N > 0 and D > 0

    assert (
        (log_loss_function == sigmoid_cross_entropy_log_loss and M == 1) or
        (log_loss_function == softmax_cross_entropy_log_loss and M == 2)
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
    # Instantiate a Matmul layer
    # --------------------------------------------------------------------------------
    matmul = Matmul(
        name="matmul",
        num_nodes=M,
        W=W,
        optimizer=optimizer,
        log_level=logging.WARNING
    )
    matmul.objective = loss.function

    num_no_progress:int = 0     # how many time when loss L not decreased.
    loss.T = T
    history: List[np.ndarray] = [loss.function(matmul.function(X))]

    for i in range(num_epochs):
        # --------------------------------------------------------------------------------
        # Layer forward path
        # Calculate the matmul output Y=f(X), and get the loss L = objective(Y)
        # Test the numerical gradient dL/dX=matmul.gradient_numerical().
        # --------------------------------------------------------------------------------
        Y = matmul.function(X)
        L = loss.function(Y)

        print(L)
        Logger.info("%s: iteration[%s]. Loss is [%s]", name, i, L)

        # --------------------------------------------------------------------------------
        # Constraint: 1. Objective/Loss L(Yn+1) after gradient descent < L(Yn)
        # --------------------------------------------------------------------------------
        if L >= history[-1]:
            Logger.warning(
                "Iteration [%i]: Loss[%s] has not improved from the previous [%s]."
                % (i, L, history[-1])
            )
            if (num_no_progress:= num_no_progress+1) > 20:
                Logger.error(
                    "The training has no progress more than %s times." % num_no_progress
                )
                # break
        else:
            num_no_progress = 0

        history.append(L)

        # --------------------------------------------------------------------------------
        # Layer backward path
        # 1. Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dL/dY.
        # 2. Gradient descent to update Wn+1 = Wn - lr * dL/dX.
        # --------------------------------------------------------------------------------
        before = copy.deepcopy(matmul.W)
        dY = loss.gradient(float(1))
        dX = matmul.gradient(dY)

        # gradient descent and get the analytical dL/dX, dL/dW
        dS = matmul.update()

        # --------------------------------------------------------------------------------
        #  Constraint 1. W in the matmul has been updated by the gradient descent.
        # --------------------------------------------------------------------------------
        Logger.info("W after is \n%s", matmul.W)
        assert not np.array_equal(before, matmul.W), \
            "W has not been updated. \n%s\n"

        # --------------------------------------------------------------------------------
        # Expected dL/dW.T = X.T @ dL/dY = X.T @ (P-T) / N, and dL/dX = dL/dY @ W
        # P = sigmoid(X) or softmax(X)
        # --------------------------------------------------------------------------------
        P = None
        if log_loss_function == sigmoid_cross_entropy_log_loss:
            P = sigmoid(np.matmul(X, W.T))
            P = P - T.reshape(-1, 1)    # T(N,) -> T(N,1) to align with P(N,1)
            assert P.shape == (N,1), "P.shape is %s T.shape is %s" % (P.shape, T.shape)
        elif log_loss_function == softmax_cross_entropy_log_loss:
            P = softmax(np.matmul(X, W.T))
            P[
                np.arange(N),
                T
            ] -= 1

        EDW = np.dot(X.T, P/N).T    # dL/dW.T shape(D,M) -> dL/dW shape(M, D)
        EDX = np.dot(P/N, W)        # (N,M) @ (M, D) -> (N, D)

        np.all(np.abs(EDX - dS[0]) < GRADIENT_DIFF_ACCEPTANCE_VALUE)
        np.all(np.abs(EDW - dS[1]) < GRADIENT_DIFF_ACCEPTANCE_VALUE)

        TEST_GRADIENT_NUMERICAL = False
        if TEST_GRADIENT_NUMERICAL:
            # --------------------------------------------------------------------------------
            # Numerical gradient
            # --------------------------------------------------------------------------------
            gn = matmul.gradient_numerical()

            # --------------------------------------------------------------------------------
            #  Constraint 2. Numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
            # --------------------------------------------------------------------------------
            assert np.all(np.abs(dS[0] - gn[0]) < GRADIENT_DIFF_ACCEPTANCE_VALUE), \
                "dL/dX analytical gradient \n%s \nneed to close to numerical gradient \n%s\n" \
                % (dS[0], gn[0])
            assert np.all(np.abs(dS[1] - gn[1]) < GRADIENT_DIFF_ACCEPTANCE_VALUE), \
                "dL/dW analytical gradient \n%s \nneed to close to numerical gradient \n%s\n" \
                % (dS[1], gn[1])

        if callback:
            callback(W=matmul.W[0]) if W.shape[1] == 1 else callback(W=np.average(matmul.W, axis=0))


def _test_binary_classifier(M: int = 2, log_loss_function: Callable = softmax_cross_entropy_log_loss):
    """Test case for layer matmul class
    """
    N = 50
    D = 3
    W = weights.he(M, D)
    optimizer = SGD(lr=0.1)
    X, T, V = linear_separable(d=D, n=N)
    # X, T = transform_X_T(X, T)

    def callback(W):
        W

    train_binary_classifier(
        N=N,
        D=D,
        M=M,
        X=X,
        T=T,
        W=W,
        log_loss_function=log_loss_function,
        optimizer=optimizer,
        num_epochs=1,
        callback=callback
    )


def test_softmax_classifier(caplog):
    profiler = cProfile.Profile()
    profiler.enable()

    _test_binary_classifier(M=2, log_loss_function=softmax_cross_entropy_log_loss)

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def test_sigmoid_classifier(caplog):
    profiler = cProfile.Profile()
    profiler.enable()

    _test_binary_classifier(M=1, log_loss_function=sigmoid_cross_entropy_log_loss)

    profiler = cProfile.Profile()
    profiler.enable()

