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
    check_with_numerical_gradient,
    ENFORCE_STRICT_ASSERT
)
from test import (
    GRADIENT_DIFF_CHECK_TRIGGER,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE
)
from data import (
    spiral,
    venn_of_circle_a_not_b
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


def build(
        M1: int,
        W1: np.ndarray,
        M2: int,
        W2: np.ndarray,
        log_loss_function: Callable,
        optimizer,
        log_level
):
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
        W=W2,                   # (M2, M1+1)
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
        W=W1,                   # (M1, D+1)
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

    return objective, prediction, loss, activation02, matmul02, activation01, matmul01


def forward(
        loss,
        activation02,
        matmul02,
        activation01,
        matmul01,
        objective,
        X
):
    # ================================================================================
    # Layer forward path
    # 1. Calculate the matmul output Y=matmul.f(X)
    # 2. Calculate the ReLU output A=activation.f(Y)
    # 3. Calculate the loss L = loss(A)
    # Test the numerical gradient dL/dX=matmul.gradient_numerical().
    # ================================================================================
    Y01 = matmul01.function(X)  # (N, D+1)  @ (M1, D+1).T  -> (N, M1)
    A01 = activation01.function(Y01)  # (N, M1)
    Y02 = matmul02.function(A01)  # (N, M1+1) @ (M2, M1+1).T -> (N, M2)
    A02 = activation02.function(Y02)  # (N, M2)
    L = loss.function(A02)

    # ********************************************************************************
    # Constraint: Network objective L must match layer-by-layer output
    # ********************************************************************************
    assert L == objective(X) and L.shape == (), \
        "Network objective L(X) %s must match layer-by-layer output %s." \
        % (objective(X), L)

    # ================================================================================
    # Expected gradients at 2nd layers
    # P = softmax(A02): (N,M2)
    # ================================================================================
    P = softmax(relu(np.matmul(matmul02.X, matmul02.W.T)))  # (N,M2)
    assert np.allclose(a=loss.P, b=P, atol=1e-12, rtol=0), \
        "Loss layer P\n%s\nExpected P\n%s\n" % (loss.P, P)

    assert P.shape == Y02.shape

    return L, P, A02, Y02, A01, Y01


def __expected_gradients_relu(
        EDA,
        Y,
        matmul,
):
    """Calculate expected gradient for ReLU"""
    # --------------------------------------------------------------------------------
    # Expected gradient EDY = dL/dY at the Matmul02 layer.
    # EDY = EDA (P-T)/N if Y > 0 else 0.
    # EDY.shape(N,M2)
    # This should match the back-propagation dL/dY from the ReLu02 layer.
    # --------------------------------------------------------------------------------
    EDY = np.copy(EDA)
    EDY[Y < 0] = TYPE_FLOAT(0)

    # --------------------------------------------------------------------------------
    # Expected gradient EDW = dL/dW02 in the Matmul02 layer.
    # EDW = matmul.X.T @ EDY.
    # EDW.shape(M2,M1+1):
    #   (M1+1,N) @ (N, M2) -> (M1+1,M2)
    #   (M1+1,M2).T        -> (M2,M1+1)
    # EDW should match dL/dW02 in [dL/dA02, dL/dW02] from the matmul.update().
    # --------------------------------------------------------------------------------
    EDW = np.matmul(matmul.X.T, EDY).T  # dL/dW.T: [(M1+1,N) @ (N,M2)].T -> (M2,M1+1)

    # ================================================================================
    # Expected gradients at 1st layers
    # Expected dL/dX = dL/dY01 @ W01 = y01 > 0 or 0 for y01 <= 0.
    # Expected dL/dW01.T = matmal01.X.T @ dL/dY01 = matmul.X.T @ W01 for y01 > 0.
    # Expected dL/dW01.T = matmal01.X.T:(D+1,N) @ dL/dY01:(N,M1) -> (D+1,M1)
    #
    # Expected dL/dX   = EDX  : dL/dY01:(N,M1) @ W01:(M1,D+1) -> (N,D+1)
    # Expected dL/dW01 = EDW01: (D+1,M1).T -> (M1,D+1)
    # ================================================================================

    # --------------------------------------------------------------------------------
    # Expected gradient EDX = dL/dA01 at the ReLu01 layer
    # EDX = EDY:(N,M2) @ W02:(M2,M1+1) -> (N,M1+1).
    # Shape of EDX must match A01:(N,M1).
    # EDA should match the back-propagation dL/dA01 from the Matmul02 layer.
    # --------------------------------------------------------------------------------
    EDX = np.matmul(EDY, matmul.W)  # dL/dA01: (N,M2) @ (M2, M1+1) -> (N, M1+1)
    EDX = EDX[
        ::,
        1::
    ]  # EDX.shape(N,M1) without bias to match A01:(N,M1)

    return EDY, EDW, EDX


def expected_gradients_relu(
        N,
        T,
        P,
        A02,
        Y02,
        A01,
        Y01,
        activation02,
        matmul02,
        activation01,
        matmul01
):
    # --------------------------------------------------------------------------------
    # Expected gradient EDA02 = dL/dA02 = (P-T)/N at the ReLU02 layer.
    # EDA02.shape(N,M2)
    # EDA02 should match the back-propagation dL/dA02 from softmax-log-loss layer.
    # --------------------------------------------------------------------------------
    # (P-T)/N, NOT P/N - T
    EDA02 = np.copy(P)
    EDA02[
        np.arange(N),
        T
    ] -= TYPE_FLOAT(1)
    EDA02 /= TYPE_FLOAT(N)

    EDY02, EDW02, EDA01 = __expected_gradients_relu(EDA02, Y02, matmul02)
    EDY01, EDW01, EDX = __expected_gradients_relu(EDA01, Y01, matmul01)

    return EDA02, EDY02, EDW02, EDA01, EDY01, EDW01, EDX


def __backward(
        back_propagation,
        activation,
        matmul,
        EDA,
        EDY,
        EDW,
        EDX,
        test_numerical_gradient

):
    # ================================================================================
    # Layer  backward path
    # 1. Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dL/dY.
    # 2. Gradient descent to update Wn+1 = Wn - lr * dL/dX.
    # ================================================================================
    before = copy.deepcopy(matmul.W)

    # ********************************************************************************
    # Constraint:
    # EDA should match the gradient dL/dA back-propagated from the log-loss layer.
    # ********************************************************************************
    dA = back_propagation
    assert np.allclose(
        a=dA,
        b=EDA,
        atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
        rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
    ), \
        "dA should match EDA. dA=\n%s\nEDA=\n%s\nDiff=\n%s\n" \
        % (dA, EDA, (dA - EDA))

    # ********************************************************************************
    # Constraint:
    # EDY should match the gradient dL/dY back-propagated from the ReLu layer.
    # ********************************************************************************
    dY = activation.gradient(dA)  # dL/dY: (N, M2)
    assert \
        np.allclose(
            a=dY,
            b=EDY,
            atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
            rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
        ), \
        "dY should match EDY. dY=\n%s\nEDY=\n%s\nDiff=\n%s\n" \
        % (dY, EDY, (dY - EDY))

    # ********************************************************************************
    # Constraint:
    # EDX should match the gradient dL/dX back-propagated from the Matmul layer.
    # ********************************************************************************
    dX = matmul.gradient(dY)  # dL/dX: (N, M1)
    if not np.allclose(
            a=dX,
            b=EDX,
            atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
            rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
    ):
        Logger.error(
            "dX should match EDX. dX=\n%s\nEDX=\n%s\nDiff=\n%s\n",
            dX, EDX, (dX - EDX)
        )
        assert ENFORCE_STRICT_ASSERT

    # ================================================================================
    # Layer  gradient descent
    # ================================================================================
    # ********************************************************************************
    #  Constraint.
    #  W in the Matmul is updated by the gradient descent.
    # ********************************************************************************
    dS = matmul.update()  # [dL/dX: (N, M1), dL/dW: (M2, M1+1)]
    Logger.debug("W after is \n%s", matmul.W)
    if np.array_equal(before, matmul.W):
        Logger.warning(
            "W has not been updated. Before=\n%s\nAfter=\n%s\nDiff=\n%s\ndW=\n%s\n",
            before, matmul.W, (before - matmul.W), dS[1]
        )

    # ********************************************************************************
    #  Constraint.
    #  dS[0] == dX
    # ********************************************************************************
    assert np.array_equal(dS[0], dX)

    # ********************************************************************************
    # Constraint:
    # EDW should match the gradient dL/dW in the Matmul layer.
    # ********************************************************************************
    dW = dS[1]
    assert \
        np.allclose(
            a=dW,
            b=EDW,
            atol=GRADIENT_DIFF_ACCEPTANCE_VALUE,
            rtol=GRADIENT_DIFF_ACCEPTANCE_RATIO
        ), \
        "dW should match EDW. dW=\n%s\nEDW=\n%s\nDiff=\n%s\n" \
        % (dW, EDW, (dW - EDW))

    # ================================================================================
    # Layer  numerical gradient
    # ================================================================================
    if test_numerical_gradient:
        gn = matmul.gradient_numerical()  # [dL/dX: (N,M1), dL/dW: (M,M+1)]
        check_with_numerical_gradient(dS, gn, Logger)

    return dX


def backward(
        back_propagation,
        activation02,
        matmul02,
        activation01,
        matmul01,
        EDA02,
        EDY02,
        EDW02,
        EDA01,
        EDY01,
        EDW01,
        EDX,
        test_numerical_gradient
):

    # ================================================================================
    # Layer 01 backward path
    # 1. Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dL/dY.
    # 2. Gradient descent to update Wn+1 = Wn - lr * dL/dX.
    # ================================================================================
    before01 = copy.deepcopy(matmul01.W)

    dA01 = __backward(
        back_propagation,
        activation02,
        matmul02,
        EDA02,
        EDY02,
        EDW02,
        EDA01,
        test_numerical_gradient
    )

    dX = __backward(
        dA01,
        activation01,
        matmul01,
        EDA01,
        EDY01,
        EDW01,
        EDX,
        test_numerical_gradient
    )


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
    assert \
        isinstance(T, np.ndarray) and np.issubdtype(T.dtype, np.integer) and \
        T.ndim == 1 and T.shape[0] == N
    assert \
        isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT and \
        X.ndim == 2 and X.shape[0] == N and X.shape[1] == D
    assert \
        isinstance(W1, np.ndarray) and W1.dtype == TYPE_FLOAT and \
        W1.ndim == 2 and W1.shape[0] == M1 and W1.shape[1] == D+1
    assert \
        isinstance(W2, np.ndarray) and W2.dtype == TYPE_FLOAT and \
        W2.ndim == 2 and W2.shape[0] == M2 and W2.shape[1] == M1+1
    assert num_epochs > 0 and N > 0 and D > 0 and M1 > 1
    assert log_loss_function == softmax_cross_entropy_log_loss and M2 >= 2

    matmul01: Matmul
    matmul02: Matmul
    *network, = build(
        M1,
        W1,
        M2,
        W2,
        log_loss_function,
        optimizer,
        log_level
    )
    objective, prediction, loss, activation02, matmul02, activation01, matmul01 = network
    loss.T = T

    # ================================================================================
    # Train the classifier
    # ================================================================================
    num_no_progress: int = 0     # how many time when loss L not decreased.
    history: List[np.ndarray] = [objective(X)]

    for i in range(num_epochs):
        # --------------------------------------------------------------------------------
        # Forward path
        # --------------------------------------------------------------------------------
        *outputs, = forward(
            loss,
            activation02,
            matmul02,
            activation01,
            matmul01,
            objective,
            X
        )
        L, P, A02, Y02, A01, Y01 = outputs

        # --------------------------------------------------------------------------------
        # Verify loss
        # --------------------------------------------------------------------------------
        if not (i % 100): print(f"iteration {i} Loss {L}")
        Logger.info("%s: iteration[%s]. Loss is [%s]", name, i, L)

        # ********************************************************************************
        # Constraint: Objective/Loss L(Yn+1) after gradient descent < L(Yn)
        # ********************************************************************************
        if L >= history[-1] and i > 0:
            Logger.warning(
                "Iteration [%i]: Loss[%s] has not improved from the previous [%s] for %s times.",
                i, L, history[-1], num_no_progress + 1
            )
            # --------------------------------------------------------------------------------
            # Reduce the learning rate.
            # --------------------------------------------------------------------------------
            matmul01.lr = matmul01.lr * 0.95
            matmul02.lr = matmul02.lr * 0.95
            if (num_no_progress := num_no_progress + 1) > 50:
                Logger.error(
                    "The training has no progress more than %s times.", num_no_progress
                )
                break
        else:
            num_no_progress = 0

        history.append(L)

        # --------------------------------------------------------------------------------
        # Expected gradients
        # --------------------------------------------------------------------------------
        *gradients, = expected_gradients_relu(
            N, T, P, A02, Y02, A01, Y01,
            activation02,
            matmul02,
            activation01,
            matmul01
        )
        EDA02, EDY02, EDW02, EDA01, EDY01, EDW01, EDX = gradients

        # --------------------------------------------------------------------------------
        # Backward path
        # --------------------------------------------------------------------------------
        backward(
            loss.gradient(TYPE_FLOAT(1)),   # dL/dA02: (N, M2),
            activation02,
            matmul02,
            activation01,
            matmul01,
            EDA02,
            EDY02,
            EDW02,
            EDA01,
            EDY01,
            EDW01,
            EDX,
            test_numerical_gradient
        )

    if callback:
        callback(matmul01.W, matmul02.W)

    return matmul01.W, matmul02.W, objective, prediction


def test_two_layer_classifier(caplog):
    """Test case for layer matmul class
    """
    caplog.set_level(logging.DEBUG, logger=Logger.name)

    # Input X specification
    D = 2                       # Dimension of X WITHOUT bias

    # Layer 01. Output Y01=X@W1.T of shape (N,M1)
    M1 = 4                      # Nodes in the matmul 01
    W1 = weights.he(M1, D+1)    # Weights in the matmul 01 WITH bias (D+1)

    # Layer 02. Input A01 of shape (N,M1).
    # Output Y02=A01@W2.T of shape (N,M2)
    M2: int = 3                 # Number of categories to classify
    W2 = weights.he(M2, M1+1)   # Weights in the matmul 02 WITH bias (M1+1)

    optimizer = SGD(lr=0.2)

    # X data
    # X, T, V = linear_separable_sectors(n=N, d=D, m=M)
    X, T = venn_of_circle_a_not_b(
        radius=TYPE_FLOAT(1.0),
        ratio=TYPE_FLOAT(1.3),
        m=M2,
        n=10
    )
    N = X.shape[0]
    assert X.shape[0] > 0 and X.shape == (N, D)
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
        num_epochs=10,
        test_numerical_gradient=True,
        log_level=logging.DEBUG,
        callback=callback
    )

    profiler.disable()
    profiler.print_stats(sort="cumtime")
