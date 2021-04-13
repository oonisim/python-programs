"""Sequenctial layer test cases
"""
import cProfile
import logging
from typing import (
    Callable
)
import copy
import numpy as np
from common.weights import he
from common.constant import (
    TYPE_FLOAT
)
from common.function import (
    compose,
    softmax_cross_entropy_log_loss
)
from common.utility import (
    random_string
)
from layer.identity import Identity
from layer import (
    Matmul,
    ReLU,
    CrossEntropyLogLoss,
    Sequential
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
    validate_relu_neuron_training,
    validate_against_expected_gradient,
    forward_outputs,
    backward_outputs
)
from test.utilities import (
    build_matmul_relu_objective,
)
from optimizer import SGD
Logger = logging.getLogger("test_030_objective")
Logger.setLevel(logging.DEBUG)


def test_050_sequential_instantiation_to_fail():
    """
    Objective:
        Verify the layer class validates the parameters
    Expected:
        Initialization parses invalid parameters and fails.
    """
    name = "test_050_instantiation_to_fail"
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(2, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)

        *network_components, = build_matmul_relu_objective(
            M,
            D,
            log_loss_function=softmax_cross_entropy_log_loss
        )

        matmul: Matmul
        activation: ReLU
        loss: CrossEntropyLogLoss
        matmul, activation, loss = network_components

        function = compose(*[matmul.function, activation.function])
        predict = compose(*[matmul.predict, activation.predict])

        # --------------------------------------------------------------------------------
        # Test the valid configurations first and then change to invalid one by one.
        # Otherwise you are not sure what you are testing.
        # --------------------------------------------------------------------------------
        try:
            inference = Sequential(
                name="test_050_sequential",
                num_nodes=M,  # NOT including bias if the 1st layer is matmul
                layers=[matmul, activation]
            )
            inference.objective = loss.function

        except Exception as e:
            raise RuntimeError("Instantiation must succeed with the correct configurations")

        # --------------------------------------------------------------------------------
        # To fail
        # --------------------------------------------------------------------------------
        try:
            inference = Sequential(
                name="",
                num_nodes=2,  # NOT including bias if the 1st layer is matmul
                layers=[matmul, activation]
            )
            raise RuntimeError("Constraint: Instantiation with a invalid name must fail")
        except AssertionError:
            pass

        try:
            inference = Sequential(
                name=name,
                num_nodes=np.random.randint(-100, 1),
                layers=[matmul, activation]
            )
            raise RuntimeError("Constraint: Instantiation with a invalid num_nodes must fail")
        except AssertionError:
            pass

        # Constraint: Number of nodes of Sequential must match the first matmul layer's
        try:
            inference = Sequential(
                name=name,
                num_nodes=M+1,
                layers=[matmul, activation]
            )
            raise RuntimeError("Constraint: Instantiation with a num_nodes != matmul.M must fail")
        except AssertionError:
            pass

        try:
            inference = Sequential(
                name=name,
                num_nodes=M,
                layers=[]
            )
            raise RuntimeError("Constraint: Instantiation with a empty layers must fail")
        except AssertionError:
            pass

        try:
            inference = Sequential(
                name=name,
                num_nodes=M,
                layers=[]
            )
            raise RuntimeError("Constraint: Instantiation with a empty layers must fail")
        except AssertionError:
            pass


def test_050_sequential_instance_property_access_to_fail():
    """
    Objective:
        Verify the layer class validates the parameters have been initialized before accessed.
    Expected:
        Instance detects the access to the non-initialized parameters and fails.
    """
    msg = "Access to uninitialized property must fail"
    name = "test_050_sequential_instance_property_access_to_fail"
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(2, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)

        *network_components, = build_matmul_relu_objective(
            M,
            D,
            log_loss_function=softmax_cross_entropy_log_loss
        )

        matmul: Matmul
        activation: ReLU
        loss: CrossEntropyLogLoss
        matmul, activation, loss = network_components

        function = compose(*[matmul.function, activation.function])
        predict = compose(*[matmul.predict, activation.predict])

        try:
            inference = Sequential(
                name=name,
                num_nodes=M,  # NOT including bias if the 1st layer is matmul
                layers=[matmul, activation]
            )

        except Exception as e:
            raise RuntimeError("Instantiation must succeed with the correct configurations")

        # --------------------------------------------------------------------------------
        # To fail
        # --------------------------------------------------------------------------------
        try:
            print(inference.X)
            raise RuntimeError("Access to uninitialized X must fail before calling function(X)")
        except AssertionError:
            pass

        try:
            inference.X = int(1)
            raise RuntimeError("Set non-float to X must fail")
        except AssertionError:
            pass

        try:
            print(inference.N)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(inference.D)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(inference.dX)
            raise RuntimeError("msg")
        except AssertionError:
            pass

        try:
            print(inference.Y)
            raise RuntimeError(msg)
        except AssertionError:
            pass
        try:
            inference._Y = int(1)
            print(inference.Y)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(inference.dY)
            raise RuntimeError(msg)
        except AssertionError:
            pass
        try:
            inference._dY = int(1)
            print(inference.dY)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            print(inference.T)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            inference.T = float(1)
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            inference.objective(np.array(1.0))
            raise RuntimeError(msg)
        except AssertionError:
            pass

        try:
            inference.objective = "hoge"
            raise RuntimeError(msg)
        except AssertionError:
            pass

        assert inference.name == name
        assert inference.num_nodes == M


def test_050_sequential_instance_property_access_to_success():
    """
    Objective:
        Verify the layer class validates the parameters have been initialized.
    Expected:
        Instance detects the access to the initialized parameters and succeed.
    """
    name = "test_050_sequential_instance_property_access_to_success"
    for _ in range(NUM_MAX_TEST_TIMES):
        name = random_string(np.random.randint(1, 10))
        M: int = np.random.randint(2, NUM_MAX_NODES)
        D: int = np.random.randint(1, NUM_MAX_FEATURES)

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
        function = compose(*[matmul.function, activation.function])
        predict = compose(*[matmul.predict, activation.predict])

        inference = Sequential(
            name=name,
            num_nodes=M,  # NOT including bias if the 1st layer is matmul
            layers=layers
        )
        inference.objective = loss.function

        # --------------------------------------------------------------------------------
        # To pass
        # --------------------------------------------------------------------------------
        try:
            if not inference.name == name: raise RuntimeError("inference.name == name should be true")
        except AssertionError:
            raise RuntimeError("Access to name should be allowed as already initialized.")

        try:
            if not inference.M == M: raise RuntimeError("inference.M == M should be true")
        except AssertionError:
            raise RuntimeError("Access to M should be allowed as already initialized.")

        try:
            if not isinstance(inference.logger, logging.Logger):
                raise RuntimeError("isinstance(inference.logger, logging.Logger) should be true")
        except AssertionError:
            raise RuntimeError("Access to logger should be allowed as already initialized.")

        try:
            if not (
                (len(inference.layers) == inference.num_layers == len(layers))
            ):
                raise RuntimeError("inference.M == M should be true")
        except AssertionError:
            raise RuntimeError("Access to layers should be allowed as already initialized.")


def test_050_sequential_instantiation():
    """
    Objective:
        Verify the initialized layer instance provides its properties.
    Expected:
        * name, num_nodes, M, log_level are the same as initialized.
        * X, T, dX, objective returns what is set.
        * N, M property are provided after X is set.
        * Y, dY properties are provided after they are set.
    """
    name = "test_050_sequential_instantiation"
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

        inference = Sequential(
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


def test_050_sequential_builder_to_succeed():
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
        valid_spec = Sequential.specification_template()
        try:
            Sequential.build(parameters=valid_spec)
        except Exception as e:
            raise \
                RuntimeError("Matmul.build() must succeed with %s" % valid_spec) \
                from e

    profiler.disable()
    profiler.print_stats(sort="cumtime")


def _build_sequential(name, M, D, W, optimizer, log_loss_function, log_level):
    *sequential_network_components, = build_matmul_relu_objective(
        M,
        D,
        W=W,
        optimizer=optimizer,
        log_loss_function=log_loss_function,
        log_level=log_level
    )

    matmul_seq: Matmul
    loss_seq: CrossEntropyLogLoss
    matmul_seq, _, loss_seq = sequential_network_components
    sequence = Sequential(
        name=name,
        num_nodes=M,  # NOT including bias if the 1st layer is matmul
        layers=[matmul_seq, Identity(name="identity", num_nodes=M), loss_seq]
    )
    sequence.objective = lambda x: np.sum(x)

    return sequence


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
    X_seq = np.copy(X)
    X_non_seq = np.copy(X)

    # --------------------------------------------------------------------------------
    # Network with sequential
    # --------------------------------------------------------------------------------
    sequence = _build_sequential(name, M, D, W, optimizer, log_loss_function, log_level)

    # --------------------------------------------------------------------------------
    # Network without sequential
    # --------------------------------------------------------------------------------
    *non_sequential_network_components, = build_matmul_relu_objective(
        M,
        D,
        W,
        optimizer=optimizer,
        log_loss_function=log_loss_function,
        log_level=log_level
    )
    matmul_non_seq, _, loss_non_seq = non_sequential_network_components
    layers_non_seq = [matmul_non_seq, loss_non_seq]

    # --------------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------------
    history = []
    for _ in range(num_epochs):
        loss_non_seq.T = T
        sequence.T = T

        # --------------------------------------------------------------------------------
        # Forward
        # --------------------------------------------------------------------------------
        EY, EL = forward_outputs(layers=layers_non_seq, X=X_non_seq)
        L = sequence.function(X_seq)
        history.append(L)

        Y = sequence.layers[0].Y
        Logger.debug("Loss is %s", L)

        assert validate_against_expected_gradient(expected=EL, actual=L), \
            "Expected L \n%s\nDiff\n%s" % (EL, EL-L)
        assert validate_against_expected_gradient(expected=EY, actual=Y), \
            "Expected dL/dY \n%s\nDiff\n%s" % (EY, EY-Y)

        # --------------------------------------------------------------------------------
        # Backward
        # --------------------------------------------------------------------------------
        dX = sequence.gradient(TYPE_FLOAT(1))
        EDA, EDX = backward_outputs(layers=layers_non_seq, dY=TYPE_FLOAT(1))
        assert validate_against_expected_gradient(EDX, dX), \
            "Expected dL/dX \n%s\nDiff\n%s" % (EDX, EDX-dX)

        # --------------------------------------------------------------------------------
        # Gradient descent
        # --------------------------------------------------------------------------------
        EDS = matmul_non_seq.update()
        EDW = EDS[0]

        dS = sequence.update()
        dW = dS[0][0]
        assert validate_against_expected_gradient(EDW, dW), \
            "Expected dL/dW \n%s\nDiff\n%s" % (EDW, EDW-dW)

    return sequence.layers[0].W


def test_050_sequential_training():
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
        optimizer = SGD(lr=1e-3, l2=1e-4)
        W = he(M, D+1)

        # --------------------------------------------------------------------------------
        # Training
        # --------------------------------------------------------------------------------
        X = np.random.rand(N, D)
        T = np.random.randint(0, 2, N)
        train_matmul_relu_classifier(
            N=N,
            D=D,
            M=M,
            X=X,
            T=T,
            W=W,
            log_loss_function=softmax_cross_entropy_log_loss,
            optimizer=optimizer,
            num_epochs=10
        )

    profiler.disable()
    profiler.print_stats(sort="cumtime")
