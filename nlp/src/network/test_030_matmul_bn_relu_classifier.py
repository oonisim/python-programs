"""Matmul layer test cases"""
from typing import (
    List,
    Callable
)
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
)
import common.weights as weights
from test.layer_validations import (
    validate_against_numerical_gradient
)
from data import (
    linear_separable_sectors
)
from layer import (
    Standardization,
    BatchNormalization,
    Matmul,
    ReLU,
    CrossEntropyLogLoss
)
from optimizer import (
    Optimizer,
    SGD
)

Logger = logging.getLogger(__name__)


def train_matmul_bn_relu_classifier(
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
        log_loss_function == softmax_cross_entropy_log_loss and M >= 2
    )

    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    loss: CrossEntropyLogLoss = CrossEntropyLogLoss(
        name="loss",
        num_nodes=M,
        log_loss_function=log_loss_function,
        log_level=log_level
    )

    # --------------------------------------------------------------------------------
    # Instantiate a ReLU layer
    # --------------------------------------------------------------------------------
    activation: ReLU = ReLU(
        name="relu",
        num_nodes=M,
        log_level=log_level
    )
    activation.objective = loss.function

    # --------------------------------------------------------------------------------
    # Instantiate a Matmul layer
    # --------------------------------------------------------------------------------
    bn: BatchNormalization = BatchNormalization(
        name=name,
        num_nodes=M,
        log_level=logging.WARNING
    )
    bn.objective = compose(activation.function, activation.objective)

    # --------------------------------------------------------------------------------
    # Instantiate a Matmul layer
    # --------------------------------------------------------------------------------
    matmul: Matmul = Matmul(
        name="matmul",
        num_nodes=M,
        W=W,
        optimizer=optimizer,
        log_level=log_level
    )
    matmul.objective = compose(bn.function, bn.objective)

    # --------------------------------------------------------------------------------
    # Instantiate a Normalization layer
    # Need to apply the same mean and std to the non-training data set.
    # --------------------------------------------------------------------------------
    # norm = Standardization(
    #     name="standardization",
    #     num_nodes=M,
    #     log_level=log_level
    # )
    # X = np.copy(X)
    # X = norm.function(X)

    # Network objective function f: L=f(X)
    objective = compose(matmul.function, matmul.objective)
    prediction = compose(
        matmul.predict,
        bn.predict,
        activation.predict
    )

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
        BN = bn.function(Y)
        A = activation.function(BN)
        L = loss.function(A)

        # ********************************************************************************
        # Constraint: Network objective L must match layer-by-layer output
        # ********************************************************************************
        assert L == objective(X) and L.shape == (), \
            f"Network objective L(X) %s must match layer-by-layer output %s." \
            % (objective(X), L)

        if not (i % 10): print(f"iteration {i} Loss {L}")
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
                matmul.lr = matmul.lr * 0.95

            if num_no_progress > 50:
                Logger.error(
                    "The training has no progress more than %s times.", num_no_progress
                )
                break
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
        dBN = activation.gradient(dA)       # dL/dBN
        dY = bn.gradient(dBN)               # dL/dY
        dX = matmul.gradient(dY)            # dL/dX

        # gradient descent and get the analytical gradients
        bn.update()

        dS = matmul.update()                # dL/dX, dL/dW
        # ********************************************************************************
        #  Constraint. W in the matmul has been updated by the gradient descent.
        # ********************************************************************************
        Logger.debug("W after is \n%s", matmul.W)
        assert not np.array_equal(before, matmul.W), "W has not been updated."

        if test_numerical_gradient:
            # --------------------------------------------------------------------------------
            # Numerical gradient
            # --------------------------------------------------------------------------------
            gn = matmul.gradient_numerical()
            validate_against_numerical_gradient([dX] + dS, gn, Logger)    # prepend dL/dX

        if callback:
            # if W.shape[1] == 1 else callback(W=np.average(matmul.W, axis=0))
            callback(W=matmul.W)

    return matmul.W, objective, prediction


def test_matmul_bn_relu_classifier(
        M: int = 3
):
    """Test case for layer matmul class
    """
    N = 10
    D = 2
    W = weights.he(M, D+1)
    optimizer = SGD(lr=0.5)
    X, T, V = linear_separable_sectors(n=N, d=D, m=M)
    assert X.shape == (N, D)
    X, T = transform_X_T(X, T)

    def callback(W):
        """Dummy callback"""
        W

    profiler = cProfile.Profile()
    profiler.enable()

    train_matmul_bn_relu_classifier(
        N=N,
        D=D,
        M=M,
        X=X,
        T=T,
        W=W,
        log_loss_function=softmax_cross_entropy_log_loss,
        optimizer=optimizer,
        test_numerical_gradient=True,
        callback=callback
    )

    profiler.disable()
    profiler.print_stats(sort="cumtime")
