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
    sigmoid_cross_entropy_log_loss
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
        X: np.ndarray,
        T: np.ndarray,
        W: np.ndarray,
        optimizer: Optimizer,
        num_epochs: int = 100,
        callback: Callable = None
):
    """Test case for binary classification with matmul + logistic log loss.
    Args:
        N: Batch size
        D: Number of features
        X: train data
        T: labels
        W: weight
        optimizer: Optimizer
        num_epochs: Number of epochs to run
        callback: callback function to invoke at the each epoch end.
    """
    name = __name__
    M = 1
    assert isinstance(T, np.ndarray) and T.dtype == int and T.ndim == 1 and T.shape[0] == N
    assert isinstance(X, np.ndarray) and X.dtype == float and X.ndim == 2 and X.shape[0] == N and X.shape[1] == D
    assert isinstance(W, np.ndarray) and W.dtype == float and W.ndim == 2 and W.shape[0] == M and W.shape[1] == D
    assert num_epochs > 0 and N > 0 and D > 0

    def objective_logloss(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective_logloss function to calculate the loss L"""
        assert X.ndim == 0, "The output of the log loss should be of shape ()"
        return X

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    loss = CrossEntropyLogLoss(
        name="loss",
        num_nodes=M,
        log_loss_function=sigmoid_cross_entropy_log_loss,
        log_level=logging.WARNING
    )
    loss.objective = objective_logloss

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

    history: List[float] = [np.finfo(float, max(float))]
    for i in range(num_epochs):
        # --------------------------------------------------------------------------------
        # Layer forward path
        # Calculate the matmul output Y=f(X), and get the loss L = objective(Y)
        # Test the numerical gradient dL/dX=matmul.gradient_numerical().
        # --------------------------------------------------------------------------------
        Y = matmul.function(X)
        loss.T = T
        L = loss.function(Y)
        history.append(L)
        print(L)

        Logger.info("%s: iteration[%s]. Loss is [%s]", name, i, L)

        # --------------------------------------------------------------------------------
        # Constraint: 1. Objective/Loss L(Yn+1) after gradient descent < L(Yn)
        # --------------------------------------------------------------------------------
        check.less(
            L, history[-2],
            "Iteration [%i]: Loss[%s] should decrease but increased from previous [%s]"
            % (i, L, history[-1])
        )

        # --------------------------------------------------------------------------------
        # Numerical gradient
        # --------------------------------------------------------------------------------
        gn = matmul.gradient_numerical()

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
        assert not np.array_equal(before, matmul.W), \
            "W has not been updated. \n%s\n"

        # --------------------------------------------------------------------------------
        #  Constraint 2. Numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
        # --------------------------------------------------------------------------------
        assert np.all(np.abs(dS[0] - gn[0]) < GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            "dL/dX analytical gradient \n%s \nneed to close to numerical gradient \n%s\n" \
            % (dS[0], gn[0])
        assert np.all(np.abs(dS[1] - gn[1]) < GRADIENT_DIFF_ACCEPTANCE_VALUE), \
            "dL/dW analytical gradient \n%s \nneed to close to numerical gradient \n%s\n" \
            % (dS[1], gn[1])

        Logger.info("W after is \n%s", matmul.W)

        if callback: callback(matmul.W[0])


def test_binary_classification(caplog, graph=False):
    """Test case for layer matmul class
    """
    caplog.set_level(logging.DEBUG)

    N = 50
    D = 3
    M = 1
    W = weights.he(M, D)
    optimizer = SGD(lr=0.1)
    X, T, V = linear_separable(d=D, n=N)
    callback = lambda _: _

    train_binary_classifier(
        N=N,
        D=D,
        X=X,
        T=T,
        W=W,
        optimizer=optimizer,
        num_epochs=100,
        callback=callback
    )
    return
    profiler = cProfile.Profile()
    profiler.enable()


    profiler.disable()
    profiler.print_stats(sort="cumtime")
