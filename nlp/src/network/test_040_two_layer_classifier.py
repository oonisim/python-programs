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
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL,
)
from common.function import (
    softmax,
    relu,
    transform_X_T,
    softmax_cross_entropy_log_loss,
    compose,
    softmax,
    relu,
    transform_X_T,
    softmax_cross_entropy_log_loss,
    compose,
)
import common.weights as weights
from test.config import (
    GRADIENT_DIFF_CHECK_TRIGGER,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE,
)
from test.layer_validations import (
    expected_gradients_from_relu_neuron,
    expected_gradient_from_log_loss,
    validate_against_expected_gradient
)
from data import (
    spiral,
    venn_of_circle_a_not_b
)
from layer import (
    Standardization,
    Matmul,
    ReLU,
    CrossEntropyLogLoss
)
from layer.utility import (
    forward_outputs,
    backward_outputs
)
from optimizer import (
    Optimizer,
    SGD
)
from test.config import (
    ENFORCE_STRICT_ASSERT,
)
from test.layer_validations import (
    validate_against_numerical_gradient
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
    # Instantiate the 2nd ReLU layer
    # --------------------------------------------------------------------------------
    activation02 = ReLU(
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
    # Instantiate the 1st ReLU layer
    # --------------------------------------------------------------------------------
    activation01 = ReLU(
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
    predict = compose(
        matmul01.predict,
        activation01.predict,
        matmul02.predict,
        # TODO: Understand why including last activation make the prediction fail.
        # The venn diagram, (A and B and C and D)
        # activation02.predict,
    )

    return objective, predict, loss, activation02, matmul02, activation01, matmul01


def expected_gradients_relu(
        N,
        T,
        P,
        Y02,
        matmul02,
        Y01,
        matmul01
):
    # --------------------------------------------------------------------------------
    # Expected gradient EDA02 = dL/dA02 = (P-T)/N at the log loss.
    # --------------------------------------------------------------------------------
    EDA02 = expected_gradient_from_log_loss(P=P, T=T, N=N)
    EDY02, EDW02, EDA01 = expected_gradients_from_relu_neuron(EDA02, Y02, matmul02)
    EDY01, EDW01, EDX = expected_gradients_from_relu_neuron(EDA01, Y01, matmul01)

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

    assert validate_against_expected_gradient(EDA, dA), \
        "dA should match EDA. dA=\n%s\nEDA=\n%s\nDiff=\n%s\n" \
        % (dA, EDA, (dA - EDA))

    # ********************************************************************************
    # Constraint:
    # EDY should match the gradient dL/dY back-propagated from the ReLU layer.
    # ********************************************************************************
    dY = activation.gradient(dA)  # dL/dY: (N, M2)
    assert validate_against_expected_gradient(EDY, dY), \
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
    dS = matmul.update()  # [dL/dW: (M2, M1+1)]
    Logger.debug("W after is \n%s", matmul.W)
    if np.array_equal(before, matmul.W):
        Logger.warning(
            "W has not been updated. Before=\n%s\nAfter=\n%s\nDiff=\n%s\ndW=\n%s\n",
            before, matmul.W, (before - matmul.W), dS[0]
        )

    # ********************************************************************************
    # Constraint:
    # EDW should match the gradient dL/dW in the Matmul layer.
    # ********************************************************************************
    dW = dS[0]
    assert validate_against_expected_gradient(EDW, dW), \
        "dW should match EDW. dW=\n%s\nEDW=\n%s\nDiff=\n%s\n" \
        % (dW, EDW, (dW - EDW))

    # ================================================================================
    # Layer  numerical gradient
    # ================================================================================
    if test_numerical_gradient:
        gn = matmul.gradient_numerical()  # [dL/dX: (N,M1), dL/dW: (M,M+1)]
        validate_against_numerical_gradient([dX] + dS, gn, Logger)

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
        isinstance(T, np.ndarray) and T.ndim == 1 and T.shape[0] == N
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
        Y01, A01, Y02, A02, L = forward_outputs(
            [
                matmul01,
                activation01,
                matmul02,
                activation02,
                loss
            ],
            X
        )
        P = softmax(relu(np.matmul(matmul02.X, matmul02.W.T)))  # (N,M2)

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
            # Reduce the learning rate can make the situation worse.
            # When reduced the lr every time L >= history, the (L >= history) became successive
            # and eventually exceeded 50 successive non-improvement ending in failure.
            # Keep the learning rate make the L>=history more frequent but still up to 3
            # successive events, and the training still kept progressing.
            # --------------------------------------------------------------------------------
            num_no_progress += 1
            if num_no_progress > 5:
                matmul01.lr = matmul01.lr * 0.95
                matmul02.lr = matmul02.lr * 0.99

            if num_no_progress > 50:
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
            N, T, P, Y02, matmul02, Y01, matmul01
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

    return matmul01.W, matmul02.W, objective, prediction, history


def test_two_layer_classifier(caplog):
    """Test case for layer matmul class
    """
    caplog.set_level(logging.WARNING, logger=Logger.name)

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
