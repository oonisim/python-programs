"""Binary classifier test cases"""
import cProfile
import copy
import logging
from typing import (
    List,
    Callable
)

import numpy as np

import common.weights as weights
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL
)
from common.function import (
    softmax,
    transform_X_T,
    sigmoid,
    sigmoid_cross_entropy_log_loss,
    softmax_cross_entropy_log_loss,
)
from data import (
    linear_separable,
    linear_separable_sectors
)
from layer import (
    Matmul,
    CrossEntropyLogLoss
)
from optimizer import (
    Optimizer,
    SGD
)
from test.layer_validations import (
    validate_against_expected_gradient
)
from test.layer_validations import (
    validate_against_numerical_gradient
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
        test_numerical_gradient: bool = False,
        log_level: int = logging.ERROR,
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
        log_level: logging level
        callback: callback function to invoke at the each epoch end.
    """
    name = __name__
    assert isinstance(T, np.ndarray) and np.issubdtype(T.dtype, np.integer) and T.ndim == 1 and T.shape[0] == N
    assert isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT and X.ndim == 2 and X.shape[0] == N and X.shape[1] == D
    assert isinstance(W, np.ndarray) and W.dtype == TYPE_FLOAT and W.ndim == 2 and W.shape[0] == M and W.shape[1] == D+1
    assert num_epochs > 0 and N > 0 and D > 0

    assert (
        (log_loss_function == sigmoid_cross_entropy_log_loss and M == 1) or
        (log_loss_function == softmax_cross_entropy_log_loss and M >= 2)
    )

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    loss = CrossEntropyLogLoss(
        name="loss",
        num_nodes=M,
        log_loss_function=log_loss_function,
        log_level=log_level
    )

    # --------------------------------------------------------------------------------
    # Instantiate a Matmul layer
    # --------------------------------------------------------------------------------
    matmul = Matmul(
        name="matmul",
        num_nodes=M,
        W=W,
        optimizer=optimizer,
        log_level=log_level
    )
    matmul.objective = loss.function

    num_no_progress: int = 0     # how many time when loss L not decreased.
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

        if not (i % 50): print(f"iteration {i} Loss {L}")
        Logger.info("%s: iteration[%s]. Loss is [%s]", name, i, L)

        # --------------------------------------------------------------------------------
        # Constraint: 1. Objective/Loss L(Yn+1) after gradient descent < L(Yn)
        # --------------------------------------------------------------------------------
        if L >= history[-1] and (i % 20) == 1:
            Logger.warning(
                "Iteration [%i]: Loss[%s] has not improved from the previous [%s].",
                i, L, history[-1]
            )
            if (num_no_progress:= num_no_progress+1) > 20:
                Logger.error(
                    "The training has no progress more than %s times.", num_no_progress
                )
                # break
        else:
            num_no_progress = 0

        history.append(L)

        # --------------------------------------------------------------------------------
        # Expected dL/dW.T = X.T @ dL/dY = X.T @ (P-T) / N, and dL/dX = dL/dY @ W
        # P = sigmoid(X) or softmax(X)
        # dL/dX = dL/dY * W is to use W BEFORE updating W.
        # --------------------------------------------------------------------------------
        P = None
        if log_loss_function == sigmoid_cross_entropy_log_loss:
            # P = sigmoid(np.matmul(X, W.T))
            P = sigmoid(np.matmul(matmul.X, matmul.W.T))
            P = P - T.reshape(-1, 1)    # T(N,) -> T(N,1) to align with P(N,1)
            assert P.shape == (N, 1), "P.shape is %s T.shape is %s" % (P.shape, T.shape)

        elif log_loss_function == softmax_cross_entropy_log_loss:
            # matmul.X.shape is (N, D+1), matmul.W.T.shape is (D+1, M)
            P = softmax(np.matmul(matmul.X, matmul.W.T))    # (N, M)
            P[
                np.arange(N),
                T
            ] -= 1

        EDX = np.matmul(P/N, matmul.W)      # (N,M) @ (M, D+1) -> (N, D+1)
        EDX = EDX[::, 1:]                   # Hide the bias    -> (N, D)
        EDW = np.matmul(matmul.X.T, P/N).T  # ((D+1,N) @ (N, M)).T -> (M, D+1)

        # --------------------------------------------------------------------------------
        # Layer backward path
        # 1. Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dL/dY.
        # 2. Gradient descent to update Wn+1 = Wn - lr * dL/dX.
        # --------------------------------------------------------------------------------
        before = copy.deepcopy(matmul.W)
        dY = loss.gradient(float(1))
        dX = matmul.gradient(dY)

        # gradient descent and get the analytical gradients dS=[dL/dX, dL/dW]
        # dL/dX.shape = (N, D)
        # dL/dW.shape = (M, D+1)
        dS = matmul.update()
        dW = dS[0]
        # --------------------------------------------------------------------------------
        #  Constraint 1. W in the matmul has been updated by the gradient descent.
        # --------------------------------------------------------------------------------
        Logger.debug("W after is \n%s", matmul.W)
        assert not np.array_equal(before, matmul.W), "W has not been updated."

        if not validate_against_expected_gradient(EDX, dX):
            Logger.warning("Expected dL/dX \n%s\nDiff\n%s", EDX, EDX-dX)
        if not validate_against_expected_gradient(EDW, dW):
            Logger.warning("Expected dL/dW \n%s\nDiff\n%s", EDW, EDW-dW)

        if test_numerical_gradient:
            # --------------------------------------------------------------------------------
            # Numerical gradients gn=[dL/dX, dL/dW]
            # dL/dX.shape = (N, D)
            # dL/dW.shape = (M, D+1)
            # --------------------------------------------------------------------------------
            gn = matmul.gradient_numerical()
            validate_against_numerical_gradient([dX] + dS, gn, Logger)

        if callback:
            # if W.shape[1] == 1 else callback(W=np.average(matmul.W, axis=0))
            callback(W=matmul.W[0])

    return matmul.W


def _test_binary_classifier(
        M: int = 2,
        log_loss_function: Callable = softmax_cross_entropy_log_loss,
        num_epochs: int = 100
):
    """Test case for layer matmul class
    """
    N = 50
    D = 2
    W = weights.he(M, D+1)
    optimizer = SGD(lr=0.1)
    X, T, V = linear_separable(d=D, n=N)
    # X, T = transform_X_T(X, T)

    def callback(W):
        return W

    train_binary_classifier(
        N=N,
        D=D,
        M=M,
        X=X,
        T=T,
        W=W,
        log_loss_function=log_loss_function,
        optimizer=optimizer,
        num_epochs=num_epochs,
        test_numerical_gradient=True,
        callback=callback
    )


def test_sigmoid_classifier(caplog):
    _test_binary_classifier(
        M=1,
        log_loss_function=sigmoid_cross_entropy_log_loss
    )


def test_softmax_classifier(caplog):
    _test_binary_classifier(
        M=2,
        log_loss_function=softmax_cross_entropy_log_loss
    )


def test_categorical_classifier(
        M: int = 3,
        log_loss_function: Callable = softmax_cross_entropy_log_loss
):
    """Test case for layer matmul class
    """
    N = 10
    D = 2
    W = weights.he(M, D+1)
    optimizer = SGD(lr=0.1)
    X, T, V = linear_separable_sectors(n=N, d=D, m=M)
    assert X.shape == (N, D)
    X, T = transform_X_T(X, T)

    def callback(W):
        W

    profiler = cProfile.Profile()
    profiler.enable()

    train_binary_classifier(
        N=N,
        D=D,
        M=M,
        X=X,
        T=T,
        W=W,
        log_loss_function=log_loss_function,
        optimizer=optimizer,
        test_numerical_gradient=True,
        log_level=logging.WARNING,
        callback=callback
    )

    profiler.disable()
    profiler.print_stats(sort="cumtime")


from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS,
    _LOSS_FUNCTION,
    _COMPOSITE_LAYER_SPEC,
    _LOG_LEVEL
)
from optimizer import (
    SGD
)
from network.sequential import (
    SequentialNetwork
)
def test():
    M = 1
    D = 2
    N = 100

    X, T, V = linear_separable(d=D, n=N)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    sigmoid_classifier_specification = {
        _NAME: "softmax_classifier",
        _NUM_NODES: M,
        _LOG_LEVEL: logging.ERROR,
        _COMPOSITE_LAYER_SPEC: {
            "matmul01": Matmul.specification(
                name="matmul",
                num_nodes=M,
                num_features=D,
                weights_initialization_scheme="he",
                weights_optimizer_specification=SGD.specification(
                    lr=0.2,
                    l2=1e-3
                )
            ),
            "loss": CrossEntropyLogLoss.specification(
                name="loss",
                num_nodes=M,
                loss_function=sigmoid_cross_entropy_log_loss.__qualname__
            )
        }
    }
    logistic_classifier = SequentialNetwork(
        specification=sigmoid_classifier_specification,
    )

    for i in range(50):
        logistic_classifier.train(X=X, T=T)

    prediction = logistic_classifier.predict(np.array([-1., -1.]))
    np.isin(prediction, [0, 1])
    print(prediction)
