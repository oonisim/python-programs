import logging
import numpy as np
from common import (
    standardize,
    logarithm,
    sigmoid,
    softmax,
    cross_entropy_log_loss,
    softmax_cross_entropy_log_loss
)

from common.test_config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    REFORMULA_DIFF_ACCEPTANCE_VALUE
)


Logger = logging.getLogger(__name__)
Logger.setLevel(logging.DEBUG)


def test_010_softmax_cross_entropy_log_loss_2d():
    """
    Objective:
        Test case for softmax_cross_entropy_log_loss(X, T) = -T * log(softmax(X))

        For the input X of shape (N,M) and T in index format of shape (N,),
        calculate the softmax log loss and verify the values are as expected.

    Expected:
        For  P = softmax(X) = 1 / (1 + exp(-X))
        _P = P[
          np.arange(N),
          T
        ] selects the probability p for the correct input x.
        Then -log(_P) should be almost same with softmax_cross_entropy_log_loss(X, T).
        Almost because finite float precision always has rounding errors.
    """
    u = REFORMULA_DIFF_ACCEPTANCE_VALUE

    # --------------------------------------------------------------------------------
    # [Test case 01]
    # X:(N,M)=(1, 2). X=(x0, x1) where x0 == x1 == 0.5 by which softmax(X) generates equal
    # probability P=(p0, p1) where p0 == p1.
    # Expected:
    #   softmax(X) generates the same with X.
    #   softmax_cross_entropy_log_loss(X, T) == -log(0.5)
    # --------------------------------------------------------------------------------
    X = np.array([[0.5, 0.5]])
    T = np.array([1])
    E = -logarithm(np.array([0.5]))

    P = softmax(X)
    assert np.array_equal(X, P)
    J = softmax_cross_entropy_log_loss(X, T)
    assert np.all(np.abs(E - J) < u), \
        "Expected abs(E-J) < %s but \n%s\nE=\n%s\nT=%s\nX=\n%s\nJ=\n%s\n" \
        % (u, (np.abs(E - J) < u), E, T, X, J)

    # --------------------------------------------------------------------------------
    # [Test case 01]
    # For X:(N,M)
    # --------------------------------------------------------------------------------
    for _ in range(NUM_MAX_TEST_TIMES):
        # X(N, M), and T(N,) in index label format
        N = num_nodes = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M = num_nodes = np.random.randint(2, NUM_MAX_NODES)

        X = np.random.rand(N, M)
        T = np.random.randint(0, M, N)
        Logger.debug("T is %s\nX is \n%s\n" % (T, X))

        # ----------------------------------------------------------------------
        # Expected value E = -logarithm(_P)
        # ----------------------------------------------------------------------
        P = softmax(X)
        _P = P[
            np.arange(N),
            T
        ]   # Probability of p for the correct input x, which generates j=-log(p)

        E = -logarithm(_P)

        # ----------------------------------------------------------------------
        # Actual J should be close to E.
        # ----------------------------------------------------------------------
        J = softmax_cross_entropy_log_loss(X, T)
        assert np.all(np.abs(E-J) < u), \
            "Expected abs(E-J) < %s but \n%s\nE=\n%s\nT=%s\nX=\n%s\nJ=\n%s\n" \
            % (u, (np.abs(E-J) < u), E, T, X, J)

        # ----------------------------------------------------------------------
        # L = cross_entropy_log_loss(P, T) should be close to J
        # ----------------------------------------------------------------------
        L = cross_entropy_log_loss(P, T)
        assert np.all(np.abs(L-J) < u), \
            "Expected abs(L-J) < %s but \n%s\nL=\n%s\nT=%s\nX=\n%s\nJ=\n%s\n" \
            % (u, (np.abs(L-J) < u), L, T, X, J)
