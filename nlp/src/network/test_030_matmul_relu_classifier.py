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
from data import (
    linear_separable_sectors
)
from test.utilities import (
    build_matmul_relu_objective
)
from optimizer import (
    Optimizer,
    SGD
)
from test.layer_validations import (
    validate_relu_neuron_training
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
        test_numerical_gradient: bool = False,
        log_level: int = logging.WARNING,
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
    # Network
    # --------------------------------------------------------------------------------
    *network_components, = build_matmul_relu_objective(
        M,
        D,
        W=W,
        optimizer=optimizer,
        log_loss_function=softmax_cross_entropy_log_loss,
        log_level=log_level
    )
    matmul, activation, loss = network_components

    # --------------------------------------------------------------------------------
    # Set objective functions at each layer
    # --------------------------------------------------------------------------------
    activation.objective = loss.function
    matmul.objective = compose(activation.function, loss.function)

    # --------------------------------------------------------------------------------
    # Network objective function. L = network.objective(X)
    # --------------------------------------------------------------------------------
    objective = compose(matmul.function, matmul.objective)

    # --------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------
    history = validate_relu_neuron_training(
        matmul=matmul,
        activation=activation,
        loss=loss,
        X=X,
        T=T,
        num_epochs=num_epochs,
        test_numerical_gradient=test_numerical_gradient
    )

    return matmul.W


def test_matmul_relu_classifier(
):
    """Test case for layer matmul class
    """
    M: int = 3
    N = 10
    D = 2
    W = weights.he(M, D+1)
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
        test_numerical_gradient=True,
        callback=callback
    )

    profiler.disable()
    profiler.print_stats(sort="cumtime")
