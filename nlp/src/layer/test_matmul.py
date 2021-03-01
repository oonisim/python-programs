"""Matmul layer test cases"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple
)
import logging
import copy
import cProfile
import numpy as np
from common import (
    OFFSET_LOG,
    OFFSET_DELTA,
    numerical_jacobian,
    weights,
    random_string,
    softmax,
    sigmoid,
    cross_entropy_log_loss,
    logistic_log_loss
)
from layer import (
    Matmul,
    CrossEntropyLogLoss
)
from optimizer import(
    SGD
)
from data.classifications import (
    linear_separable
)

Logger = logging.getLogger("test_030_objective")
Logger.setLevel(logging.DEBUG)

MAX_TEST_TIMES = 10
N = 10
D = 2
M = 1
X, T, V = linear_separable(d=D, n=N)
print(f"X.shape {X.shape} T.shape {T.shape} W {V}")
Logger.debug(f"X\n{X}")


def train_binary_classifier():
    """Test case for binary classification with matmul + logistic log loss.
    """
    def objective_logloss(X: np.ndarray) -> Union[float, np.ndarray]:
        """Dummy objective_logloss function to calculate the loss L"""
        assert X.ndim == 0, "The output of the log loss should be of shape ()"
        return X

    name = train_binary_classifier
    # --------------------------------------------------------------------------------
    # Instantiate a CrossEntropyLogLoss layer
    # --------------------------------------------------------------------------------
    logloss = CrossEntropyLogLoss(
        name="loss",
        num_nodes=M,
        activation=sigmoid,
        log_level=logging.WARNING
    )
    logloss.objective = objective_logloss

    # --------------------------------------------------------------------------------
    # Instantiate a Matmul layer
    # --------------------------------------------------------------------------------
    W = weights.he(M, D + 1)
    sgd = SGD(lr=0.1)

    matmul = Matmul(
        name="matmul",
        num_nodes=M,
        W=W,
        optimizer=sgd,
        log_level=logging.WARNING
    )
    matmul.objective = logloss.function

    history: List[float] = []
    for i in range(MAX_TEST_TIMES):
        # --------------------------------------------------------------------------------
        # Layer forward path
        # Calculate the matmul output Y=f(X), and get the loss L = objective(Y)
        # Test the numerical gradient dL/dX=matmul.gradient_numerical().
        # --------------------------------------------------------------------------------
        Y = matmul.function(X)
        logloss.T = T
        L = logloss.function(Y)
        history.append(L)
        Logger.info("%s: iteration[%s]. Loss is [%s]", name, i, L)
        if L > history[-1]:
            Logger.warning(
                "Loss [%s] should decrease but increased from previous [%s]",
                L, history[-1]
            )

        # --------------------------------------------------------------------------------
        # Numerical gradient
        # --------------------------------------------------------------------------------
        gn = matmul.gradient_numerical()

        # --------------------------------------------------------------------------------
        # Layer backward path
        # Calculate the analytical gradient dL/dX=matmul.gradient(dL/dY) with a dummy dL/dY.
        # Confirm the numerical gradient (dL/dX, dL/dW) are closer to the analytical ones.
        # --------------------------------------------------------------------------------
        before = copy.deepcopy(matmul.W)
        dY = logloss.gradient(float(1))
        dX = matmul.gradient(dY)

        # --------------------------------------------------------------------------------
        # Gradient update.
        # Run the gradient descent to update Wn+1 = Wn - lr * dL/dX.
        # Confirm the new objective L(Yn+1) < L(Yn) with the Wn+1.
        # Confirm W in the matmul has been updated by the gradient descent.
        # --------------------------------------------------------------------------------
        dS = matmul.update()  # Analytical dL/dX, dL/dW
        assert not np.array_equal(before, matmul.W), "W has not been updated. \n%s\n"
        assert np.all(np.abs(dS[0] - gn[0]) < 0.001), \
            "dL/dX analytical gradient \n%s \nneed to close to numerical gradient \n%s\n" \
            % (dS[0], gn[0])
        assert np.all(np.abs(dS[1] - gn[1]) < 0.001), \
            "dL/dW analytical gradient \n%s \nneed to close to numerical gradient \n%s\n" \
            % (dS[1], gn[1])

        Logger.info("W after is \n%s", matmul.W)


def test_binary_classification(graph=False):
    """Test case for layer matmul class
    """
    profiler = cProfile.Profile()
    profiler.enable()

    train_binary_classifier()

    profiler.disable()
    profiler.print_stats(sort="cumtime")
