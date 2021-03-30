import cProfile
import logging
from typing import (
    Callable
)

import numpy as np

from common.constants import (
    TYPE_FLOAT
)
from common.functions import (
    compose,
    softmax_cross_entropy_log_loss
)
from common.utilities import (
    random_string
)
from layer import (
    Matmul,
    ReLU,
    CrossEntropyLogLoss,
    Parallel
)
from optimizer import (
    Optimizer
)
from test.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES
)
from test.layer_validations import (
    validate_relu_neuron_training
)
from test.utilities import (
    build_matmul_relu_objective
)

Logger = logging.getLogger("test_030_objective")
Logger.setLevel(logging.DEBUG)


def test_050_parallel_instantiation():
    """
    Objective:
        Verify the initialized layer instance provides its properties.
    Expected:
        * name, num_nodes, M, log_level are the same as initialized.
        * X, T, dX, objective returns what is set.
        * N, M property are provided after X is set.
        * Y, dY properties are provided after they are set.
    """
    name = "test_050_parallel_instantiation"
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(2, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)

        *network_components, = build_matmul_relu_objective(
            M,
            D,
            log_loss_function=softmax_cross_entropy_log_loss
        )

        matmul: Matmul
        activation: ReLU
        loss: CrossEntropyLogLoss
        matmul, activation, loss = network_components

        layers = [matmul, activation]

        inference = Parallel(
            name=name,
            num_nodes=M,  # NOT including bias if the 1st layer is matmul
            layers=layers
        )
        inference.objective = loss.function

        assert inference.name == name
        assert inference.num_nodes == inference.M == M

        X = np.random.randn(N, D)
        inference.X = X
        assert np.array_equal(inference.X, X)
        assert inference.N == N == X.shape[0]

        inference._dX = X
        assert np.array_equal(inference.dX, X)

        T = np.random.randint(0, M, N)
        inference.T = T
        assert np.array_equal(inference.T, T)

        inference._Y = np.dot(X, X.T)
        assert np.array_equal(inference.Y, np.dot(X, X.T))

        inference._dY = np.array(0.9)
        assert inference._dY == np.array(0.9)

        inference.logger.debug("This is a pytest")

        assert inference.objective == loss.function


def test_050_parallel_builder_to_succeed():
    """
    Objective:
        Verify the Matmul.build()
    Expected:
        build() parse the spec and succeed
    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        # ----------------------------------------------------------------------
        # Validate the correct specification.
        # NOTE: Invalidate one parameter at a time from the correct one.
        # Otherwise not sure what you are testing.
        # ----------------------------------------------------------------------
        valid_spec = Parallel.build_specification_template()
        try:
            Parallel.build(parameters=valid_spec)
        except Exception as e:
            raise \
                RuntimeError("Matmul.build() must succeed with %s" % valid_spec) \
                from e

    profiler.disable()
    profiler.print_stats(sort="cumtime")


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

    *network_components, = build_matmul_relu_objective(
        M,
        D,
        W=W,
        optimizer=optimizer,
        log_loss_function=softmax_cross_entropy_log_loss,
        log_level=log_level
    )

    # --------------------------------------------------------------------------------
    # Network
    # --------------------------------------------------------------------------------
    matmul: Matmul
    activation: ReLU
    loss: CrossEntropyLogLoss
    matmul, activation, loss = network_components

    layers = [matmul]
    inference = Parallel(
        name=name,
        num_nodes=M,  # NOT including bias if the 1st layer is matmul
        layers=layers
    )
    inference.objective = loss.function
    objective = compose(inference.function, inference.objective)

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
    for line in history:
        print(line)

    return matmul.W


def test_050_parallel_training():
    """
    Objective:
        Verify the forward and backward paths at matmul.

    Expected:
        Forward path:
        1. Matmul function(X) == X @ W.T
        2. Numerical gradient should be the same with numerical Jacobian

        Backward path:
        3. Analytical gradient dL/dX == dY @ W
        4. Analytical dL/dW == X.T @ dY
        5. Analytical gradients are similar to the numerical gradient ones

        Gradient descent
        6. W is updated via the gradient descent.
        7. Objective L is decreasing via the gradient descent.

    """
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(2, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)

        *network_components, = build_matmul_relu_objective(
            M,
            D,
            log_loss_function=softmax_cross_entropy_log_loss,
            log_level=logging.WARNING
        )

        # --------------------------------------------------------------------------------
        # Network
        # --------------------------------------------------------------------------------
        matmul: Matmul
        activation: ReLU
        loss: CrossEntropyLogLoss
        matmul, activation, loss = network_components

        layers = [matmul, activation]
        function = compose(*[matmul.function, activation.function])
        predict = compose(*[matmul.predict, activation.predict])

        inference = Parallel(
            name=name,
            num_nodes=M,  # NOT including bias if the 1st layer is matmul
            layers=layers
        )
        inference.objective = loss.function
        objective = compose(inference.function, inference.objective)

        # --------------------------------------------------------------------------------
        # Training
        # --------------------------------------------------------------------------------
        X = np.random.rand(N, D)
        T = np.random.randint(0, 2, N)
        history = validate_relu_neuron_training(
            matmul=matmul,
            activation=activation,
            loss=loss,
            X=X,
            T=T,
            num_epochs=5,
            test_numerical_gradient=True,
        )

    profiler.disable()
    profiler.print_stats(sort="cumtime")
