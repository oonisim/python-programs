"""Matmul layer test cases"""
from typing import (
    List,
    Callable
)
import sys
import copy
import logging
import cProfile
import numpy as np
from common import (
    TYPE_FLOAT,
    TYPE_LABEL,
    weights,
    softmax,
    relu,
    transform_X_T,
    softmax_cross_entropy_log_loss,
    compose,
    ENFORCE_STRICT_ASSERT
)
from common.test_config import (
    GRADIENT_DIFF_CHECK_TRIGGER,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE
)
from data import (
    spiral
)
from layer import (
    Standardization,
    Matmul,
    Relu,
    CrossEntropyLogLoss
)
from optimizer import (
    Optimizer,
    SGD
)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1024)
Logger = logging.getLogger(__name__)


def train_two_layer_classifier(
        N: int,
        D: int,
        X: np.ndarray,
        T: np.ndarray,
        M1: int,
        W1: np.ndarray,
        M2: int,
        W2: np.ndarray,
        log_loss_function: Callable,
        optimizer: Optimizer,
        num_epochs: int = 100,
        test_numerical_gradient: bool = False,
        log_level: int = logging.ERROR,
        callback: Callable = None
):
    """Test case for binary classification with matmul + log loss.
    Args:
        N: Batch size of X
        D: Number of features in X
        X: train data
        T: labels
        M1: Number of nodes in layer 1.
        W1: weight for layer 1
        M2: Number of nodes in layer 2.
        W2: weight for layer 2
        log_loss_function: cross entropy logg loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to run
        test_numerical_gradient: Flag if test the analytical gradient with the numerical one.
        log_level: logging level
        callback: callback function to invoke at the each epoch end.
    """
    name = __name__
    assert isinstance(T, np.ndarray) and np.issubdtype(T.dtype, np.integer) and T.ndim == 1 and T.shape[0] == N
    assert isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT and X.ndim == 2 and X.shape[0] == N and X.shape[1] == D
    assert isinstance(W1, np.ndarray) and W1.dtype == TYPE_FLOAT and W1.ndim == 2 and W1.shape[0] == M1 and W1.shape[1] == D
    assert isinstance(W2, np.ndarray) and W2.dtype == TYPE_FLOAT and W2.ndim == 2 and W2.shape[0] == M2 and W2.shape[1] == M1
    assert num_epochs > 0 and N > 0 and D > 0 and M1 > 1
    assert (
        log_loss_function == softmax_cross_entropy_log_loss and M2 >= 2
    )

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    loss = CrossEntropyLogLoss(
        name="loss",
        num_nodes=M2,
        log_loss_function=log_loss_function,
        log_level=log_level
    )

    # --------------------------------------------------------------------------------
    # Instantiate the 2nd ReLu layer
    # --------------------------------------------------------------------------------
    activation02 = Relu(
        name="relu02",
        num_nodes=M2,
        log_level=log_level
    )
    activation02.objective = loss.function

    # --------------------------------------------------------------------------------
    # Instantiate the 2nd Matmul layer
    # --------------------------------------------------------------------------------
    matmul02 = Matmul(
        name="matmul02",
        num_nodes=M2,
        W=W2,
        optimizer=optimizer,
        log_level=log_level
    )
    matmul02.objective = compose(activation02.function, activation02.objective)

    # --------------------------------------------------------------------------------
    # Instantiate the 1st ReLu layer
    # --------------------------------------------------------------------------------
    activation01 = Relu(
        name="relu01",
        num_nodes=M1,
        log_level=log_level
    )
    activation01.objective = compose(matmul02.function, matmul02.objective)

    # --------------------------------------------------------------------------------
    # Instantiate the 2nd Matmul layer
    # --------------------------------------------------------------------------------
    matmul01 = Matmul(
        name="matmul01",
        num_nodes=M1,
        W=W1,
        optimizer=optimizer,
        log_level=log_level
    )
    matmul01.objective = compose(activation01.function, activation01.objective)

    # --------------------------------------------------------------------------------
    # Instantiate a Normalization layer
    # Need to apply the same mean and std to the non-training data set.
    # --------------------------------------------------------------------------------
    # # norm = Standardization(
    #     name="standardization",
    #     num_nodes=D
    # )
    # X = np.copy(X)
    # X = norm.function(X)

    # --------------------------------------------------------------------------------
    # Network objective function f: L=f(X)
    # --------------------------------------------------------------------------------
    objective = compose(matmul01.function, matmul01.objective)
    prediction = compose(
        matmul01.function,
        activation01.function,
        matmul02.function
    )

    # ================================================================================
    # Train the classifier
    # ================================================================================
    num_no_progress: int = 0     # how many time when loss L not decreased.
    loss.T = T
    history: List[np.ndarray] = [objective(X)]

    for i in range(num_epochs):
        # ================================================================================
        # Layer forward path
        # 1. Calculate the matmul output Y=matmul.f(X)
        # 2. Calculate the ReLU output A=activation.f(Y)
        # 3. Calculate the loss L = loss(A)
        # Test the numerical gradient dL/dX=matmul.gradient_numerical().
        # ================================================================================
        Y01 = matmul01.function(X)
        A01 = activation01.function(Y01)
        Y02 = matmul02.function(A01)
        A02 = activation02.function(Y02)
        L = loss.function(A02)

        # ********************************************************************************
        # Constraint: Network objective L must match layer-by-layer output
        # ********************************************************************************
        assert L == objective(X) and L.shape == (), \
            "Network objective L(X) %s must match layer-by-layer output %s." \
            % (objective(X), L)

        if not (i % 100): print(f"iteration {i} Loss {L}")
        Logger.info("%s: iteration[%s]. Loss is [%s]", name, i, L)

        # ********************************************************************************
        # Constraint: Objective/Loss L(Yn+1) after gradient descent < L(Yn)
        # ********************************************************************************
        if L >= history[-1] and (i % 20) == 1:
            Logger.warning(
                "Iteration [%i]: Loss[%s] has not improved from the previous [%s] for %s times.",
                i, L, history[-1], num_no_progress+1
            )
            if (num_no_progress := num_no_progress+1) > 50:
                Logger.error(
                    "The training has no progress more than %s times.", num_no_progress
                )
                break
        else:
            num_no_progress = 0

        history.append(L)

        # ================================================================================
        # Layer 02 backward path
        # 1. Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dL/dY.
        # 2. Gradient descent to update Wn+1 = Wn - lr * dL/dX.
        # ================================================================================
        before02 = copy.deepcopy(matmul02.W)
        before01 = copy.deepcopy(matmul01.W)

        dA02 = loss.gradient(float(1))      # dL/dA02
        dY02 = activation02.gradient(dA02)  # dL/dY02
        dA01 = matmul02.gradient(dY02)      # dL/dA01

        dY01 = activation01.gradient(dA01)  # dL/dY01
        dX = matmul01.gradient(dY01)      # dL/dX

        # gradient descent and get the analytical dL/dX, dL/dW
        dS02 = matmul02.update()            # dL/dA01, dL/dW02

        # ********************************************************************************
        #  Constraint. W in the matmul has been updated by the gradient descent.
        # ********************************************************************************
        Logger.debug("W02 after is \n%s", matmul02.W)
        assert not np.array_equal(before02, matmul02.W), "W02 has not been updated."

        # --------------------------------------------------------------------------------
        # Expected dL/dW02.T = X02.T @ dL/dY02 = X02.T @ (P-T) / N for y02 > 0 because of ReLU.
        # Expected dL/dX02 = dL/dY02 @ W02 = (P-T) @ W02 / N for y02 > 0 or 0 for y02 <= 0.
        # X02 = A01
        # P = softmax(A02)
        # --------------------------------------------------------------------------------
        P = softmax(relu(np.matmul(matmul02.X, matmul02.W.T)))
        assert P.shape == Y02.shape
        # gradient dL/dA02 = (P-T)/N from softmax-log-loss
        P[
            np.arange(N),
            T
        ] -= 1
        P /= N

        # ********************************************************************************
        # Constraint. Analytical gradients dY02 or dL/dY02 from ReLU 02 layer is close to
        # Expected dL/dY.
        # ********************************************************************************
        # dL/dY02 gradient at ReLU. 1 when y > 0 or 0 otherwise.
        P[(Y02 <= 0)] = 0                    # Expected dL/dY
        if not np.all(np.abs(dY02 - P) < GRADIENT_DIFF_ACCEPTANCE_VALUE):
            Logger.error(
                "dL/dY02=\n%s\nExpected=\n%s\nDiff=\n%s", dY02, P, (dY02 - P)
            )
            assert ENFORCE_STRICT_ASSERT

        # ********************************************************************************
        # Constraint. Analytical gradients from layer close to expected gradients EDA01/EDW02.
        # ********************************************************************************
        EDA01 = np.matmul(P, matmul02.W)        # dL/dA01 (N,M) @ (M, D) -> (N, D)
        EDW02 = np.matmul(matmul02.X.T, P).T    # dL/dW.T shape(D,M) -> dL/dW shape(M, D)

        if not(
                np.all(np.abs(dA01 - EDA01) < GRADIENT_DIFF_ACCEPTANCE_VALUE)
        ):
            Logger.error(
                "dL/dA01=\n%s\nExpected=\n%s\nDiff=\n%s", dA01, EDA01, (dA01-EDA01)
            )
            assert ENFORCE_STRICT_ASSERT

        delta_dX = np.abs(EDA01-dS02[0])
        delta_dW01 = np.abs(EDW02-dS02[1])
        if not (
            np.all(np.abs(EDA01) <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.all(delta_dX <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
            np.all(delta_dX <= np.abs(dS02[0] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
        ):
            Logger.error("Expected dL/dX \n%s\nDiff\n%s", EDA01, EDA01-dS02[0])
        if not (
            np.all(np.abs(EDW02) <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.all(delta_dW01 <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
            np.all(delta_dW01 <= np.abs(dS02[1] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
        ):
            Logger.error("Expected dL/dW \n%s\nDiff\n%s", EDW02, EDW02-dS02[1])

        if test_numerical_gradient:
            # --------------------------------------------------------------------------------
            # Numerical gradient
            # --------------------------------------------------------------------------------
            gn02 = matmul02.gradient_numerical()

            # ********************************************************************************
            #  Constraint. Numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
            # ********************************************************************************
            delta_GX02 = np.abs(dS02[0] - gn02[0])
            if not (
                np.all(delta_GX02 <= GRADIENT_DIFF_CHECK_TRIGGER) or
                np.all(delta_GX02 <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
                np.all(delta_GX02 <= np.abs(gn02[0] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
            ):
                Logger.error(
                    "dL/dX analytical gradient \n%s \nneed to close to numerical gradient \n%s\ndifference=\n%s\n",
                    dS02[0], gn02[0], delta_GX02
                )

            delta_GW02 = np.abs(dS02[1] - gn02[1])
            if not (
                np.all(delta_GW02 <= GRADIENT_DIFF_CHECK_TRIGGER) or
                np.all(delta_GW02 <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
                np.all(delta_GW02 <= np.abs(gn02[1] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
            ):
                Logger.error(
                    "dL/dW analytical gradient \n%s \nneed to close to numerical gradient \n%s\ndifference=\n%s\n",
                    dS02[1], gn02[1], delta_GW02
                )

        # ================================================================================
        # Layer 01 backward path
        # 1. Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dL/dY.
        # 2. Gradient descent to update Wn+1 = Wn - lr * dL/dX.
        # ================================================================================
        dS01 = matmul01.update()            # dL/dX, dL/dW01
        Logger.debug("W01 after is \n%s", matmul01.W)
        assert not np.array_equal(before01, matmul01.W), "W01 has not been updated."

        # --------------------------------------------------------------------------------
        # Expected dL/dW01.T = X.T @ dL/dY01 = X.T @ for y01 > 0 because of ReLU.
        # Expected dL/dX = dL/dY01 @ W01 = y01 > 0 or 0 for y01 <= 0.
        # --------------------------------------------------------------------------------
        EDY01 = np.copy(EDA01)
        # dY01 or dL/dY01 = dL/dA01 * dA01/dY01
        # dA01/dY01 at ReLU 01 = 1 when y1 > 0 otherwise 0.
        EDY01[Y01 <= 0] = 0

        # ********************************************************************************
        # dL/dY01 from the ReLU 01 layer close to the expected
        # ********************************************************************************
        delta_dY01 = np.abs(EDY01 - dY01)
        if not (
            np.all(dY01 <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.all(delta_dY01 <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
            np.all(delta_dY01 <= np.abs(dY01 * GRADIENT_DIFF_ACCEPTANCE_RATIO))
        ):
            Logger.error(
                "dL/dY01=\n%s\nExpected=\n%s\nDiff=\n%s",
                dY01, EDY01, (EDY01 - dY01)
            )
            assert ENFORCE_STRICT_ASSERT

        EDX = np.matmul(EDY01, matmul01.W)     # dL/dX (N,M) @ (M, D) -> (N, D)
        EDW01 = np.matmul(matmul01.X.T, EDY01).T   # dL/dW01.T shape(D,M) -> dL/dW shape(M, D)

        delta_EDX = np.abs(dX - EDX)
        if not(
            np.all(dX <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.all(delta_EDX < GRADIENT_DIFF_ACCEPTANCE_VALUE) or
            np.all(delta_EDX <= np.abs(dX * GRADIENT_DIFF_ACCEPTANCE_RATIO))
        ):
            Logger.error(
                f"dL/dX=\n%s\nExpected=\n%s\nDiff=\n%s", dX, EDX, (dX-EDX)
            )
            assert ENFORCE_STRICT_ASSERT

        delta_dX = np.abs(EDX-dS01[0])
        delta_dW01 = np.abs(EDW01-dS01[1])
        if not (
            np.all(np.abs(EDX) <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.all(delta_dX <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
            np.all(delta_dX <= np.abs(dS01[0] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
        ):
            Logger.error("Expected dL/dX \n%s\nDiff\n%s", EDX, EDX-dS01[0])
            assert ENFORCE_STRICT_ASSERT

        if not (
            np.all(np.abs(EDW01) <= GRADIENT_DIFF_CHECK_TRIGGER) or
            np.all(delta_dW01 <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
            np.all(delta_dW01 <= np.abs(dS01[1] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
        ):
            Logger.error("Expected dL/dW \n%s\nDiff\n%s", EDW01, EDW01-dS01[1])
            assert ENFORCE_STRICT_ASSERT

        if test_numerical_gradient:
            # --------------------------------------------------------------------------------
            # Numerical gradient
            # --------------------------------------------------------------------------------
            gn01 = matmul01.gradient_numerical()

            # ********************************************************************************
            #  Constraint. Numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
            # ********************************************************************************
            delta_GX01 = np.abs(dS01[0] - gn01[0])
            if not (
                np.all(delta_GX01 <= GRADIENT_DIFF_CHECK_TRIGGER) or
                np.all(delta_GX01 <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
                np.all(delta_GX01 <= np.abs(gn01[0] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
            ):
                Logger.error(
                    "dL/dX analytical gradient \n%s \nneed to close to numerical gradient \n%s\ndifference=\n%s\n",
                    dS01[0], gn01[0], delta_GX01
                )
                assert ENFORCE_STRICT_ASSERT

            delta_GW01 = np.abs(dS01[1] - gn01[1])
            if not (
                np.all(delta_GW01 <= GRADIENT_DIFF_CHECK_TRIGGER) or
                np.all(delta_GW01 <= GRADIENT_DIFF_ACCEPTANCE_VALUE) or
                np.all(delta_GW01 <= np.abs(gn01[1] * GRADIENT_DIFF_ACCEPTANCE_RATIO))
            ):
                Logger.error(
                    "dL/dW analytical gradient \n%s \nneed to close to numerical gradient \n%s\ndifference=\n%s\n",
                    dS01[1], gn01[1], delta_GW01
                )
                assert ENFORCE_STRICT_ASSERT

        if callback:
            callback(matmul01.W, matmul02.W)

    return matmul01.W, matmul02.W, objective, prediction


def test_two_layer_classifier(caplog):
    """Test case for layer matmul class
    """
    caplog.set_level(logging.DEBUG, logger=Logger.name)

    D = 3
    M1 = 4
    W1 = weights.he(M1, D)
    M2: int = 3                 # Number of categories to classify
    W2 = weights.he(M2, M1)
    optimizer = SGD(lr=0.2)
    # X, T, V = linear_separable_sectors(n=N, d=D, m=M)

    # X[::,0] is bias
    K = 3                      # Number of data points per class
    N = M2 * K                  # Number of entire data points
    X, T = spiral(K, D, M2)

    assert X.shape == (N, D)
    X, T = transform_X_T(X, T)

    def callback(W1, W2):
        """Dummy callback"""
        pass

    profiler = cProfile.Profile()
    profiler.enable()

    train_two_layer_classifier(
        N=N,
        D=D,
        X=X,
        T=T,
        M1=M1,
        W1=W1,
        M2=M2,
        W2=W2,
        log_loss_function=softmax_cross_entropy_log_loss,
        optimizer=optimizer,
        test_numerical_gradient=True,
        log_level=logging.DEBUG,
        callback=callback
    )

    profiler.disable()
    profiler.print_stats(sort="cumtime")
