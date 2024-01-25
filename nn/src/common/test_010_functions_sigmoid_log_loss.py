import logging
import numpy as np
from common.constant import (
    TYPE_FLOAT,
    TYPE_LABEL,
)
from common.function import (
    logarithm,
    sigmoid,
    logistic_log_loss,
    cross_entropy_log_loss,
    sigmoid_cross_entropy_log_loss,
    transform_X_T,
)
from testing.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    REFORMULA_DIFF_ACCEPTANCE_VALUE,
)


logging.basicConfig(level=logging.DEBUG)
Logger = logging.getLogger(__name__)


def test_010_sigmoid_cross_entropy_log_loss_2d(caplog):
    """
    Objective:
        Test case for sigmoid_cross_entropy_log_loss(X, T) =
        -( T * log(sigmoid(X)) + (1 -T) * log(1-sigmoid(X)) )

        For the input X of shape (N,1) and T in index format of shape (N,1),
        calculate the sigmoid log loss and verify the values are as expected.

    Expected:
        For  Z = sigmoid(X) = 1 / (1 + exp(-X)) and T=[[1]]
        Then -log(Z) should be almost same with sigmoid_cross_entropy_log_loss(X, T).
        Almost because finite float precision always has rounding errors.
    """
    # caplog.set_level(logging.DEBUG, logger=Logger.name)
    u = REFORMULA_DIFF_ACCEPTANCE_VALUE

    # --------------------------------------------------------------------------------
    # [Test case 01]
    # X:(N,M)=(1, 1). X=(x0) where x0=0 by which sigmoid(X) generates 0.5.
    # Expected:
    #   sigmoid_cross_entropy_log_loss(X, T) == -log(0.5)
    # --------------------------------------------------------------------------------
    X = np.array([[TYPE_FLOAT(0.0)]])
    T = np.array([TYPE_LABEL(1)])
    X, T = transform_X_T(X, T)
    E = -logarithm(np.array([TYPE_FLOAT(0.5)]))

    J, P = sigmoid_cross_entropy_log_loss(X, T)
    assert E.shape == J.shape
    assert np.all(E == J), \
        "Expected (E==J) but \n%s\nE=\n%s\nT=%s\nX=\n%s\nJ=\n%s\n" \
        % (np.abs(E - J), E, T, X, J)
    assert P == 0.5

    # --------------------------------------------------------------------------------
    # [Test case 02]
    # For X:(N,1)
    # --------------------------------------------------------------------------------
    for _ in range(NUM_MAX_TEST_TIMES):
        # X(N, M), and T(N,) in index label format
        N = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M = 1   # always 1 for binary classification 0 or 1.

        X = np.random.randn(N, M).astype(TYPE_FLOAT)
        T = np.random.randint(0, M, N).astype(TYPE_LABEL)
        X, T = transform_X_T(X, T)
        Logger.debug("T is %s\nX is \n%s\n", T, X)

        # ----------------------------------------------------------------------
        # Expected value EJ for J and Z for P
        # Note:
        #   To handle both index label format and OHE label format in the
        #   Loss layer(s), X and T are transformed into (N,1) shapes in
        #   transform_X_T(X, T) for logistic log loss.
        # DO NOT squeeze Z nor P.
        # ----------------------------------------------------------------------
        Z = sigmoid(X)
        EJ = np.squeeze(-(T * logarithm(Z) + TYPE_FLOAT(1-T) * logarithm(TYPE_FLOAT(1-Z))), axis=-1)

        # **********************************************************************
        # Constraint: Actual J should be close to EJ.
        # **********************************************************************
        J, P = sigmoid_cross_entropy_log_loss(X, T)
        assert EJ.shape == J.shape
        assert np.all(np.abs(EJ-J) < u), \
            "Expected abs(EJ-J) < %s but \n%s\nEJ=\n%s\nT=%s\nX=\n%s\nJ=\n%s\n" \
            % (u, np.abs(EJ-J), EJ, T, X, J)
        
        # **********************************************************************
        # Constraint: Actual P should be close to Z.
        # **********************************************************************
        assert np.all(np.abs(Z-P) < u), \
            "EP \n%s\nP\n%s\nEP-P \n%s\n" % (Z, P, Z-P)

        # ----------------------------------------------------------------------
        # L = cross_entropy_log_loss(P, T) should be close to J
        # ----------------------------------------------------------------------
        L = cross_entropy_log_loss(P=Z, T=T, f=logistic_log_loss)
        assert L.shape == J.shape
        assert np.all(np.abs(L-J) < u), \
            "Expected abs(L-J) < %s but \n%s\nL=\n%s\nT=%s\nX=\n%s\nJ=\n%s\n" \
            % (u, np.abs(L-J), L, T, X, J)
