import sys
import logging
import numpy as np
from common.constant import (
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_LABEL,
    OFFSET_LOG,
)
from common.function import (
    standardize,
    logarithm,
    sigmoid,
    softmax,
    cross_entropy_log_loss,
    logistic_log_loss
)

from testing.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    MAX_ACTIVATION_VALUE,
    ACTIVATION_DIFF_ACCEPTANCE_VALUE,
    LOSS_DIFF_ACCEPTANCE_VALUE,
    LOSS_DIFF_ACCEPTANCE_RATIO
)


Logger = logging.getLogger(__name__)
Logger.setLevel(logging.DEBUG)


def test_010_standardize():
    """
    Objective:
        Verify the standardize() function.
    Expected:
        standardize(X) = (X - np.mean(X)) / np.std(X)  (when np.std(X) != 0)
    """
    name = "test_010_standardize"
    keepdims = True
    u = 1e-6

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        X = np.random.uniform(
            -MAX_ACTIVATION_VALUE,
            MAX_ACTIVATION_VALUE,
            (N, M)
        ).astype(TYPE_FLOAT)
        Logger.debug("%s: X \n%s\n", name, X)

        # Constraint: standardize(X) == (X - np.mean(A)) / np.std(X)
        ddof = 1 if N > 1 else 0
        sd = np.std(X, axis=0, keepdims=keepdims, ddof=ddof)
        if np.all(sd > 0):
            # Expected
            mean = np.mean(X, axis=0)
            E = (X - mean) / sd
            # Actual
            A, __mean, __sd, _ = standardize(X, keepdims=keepdims)

            # Constraint. mean/sd should be same
            assert np.all(np.abs(mean - __mean) < u)
            assert np.all(np.abs(sd - __sd) < u)
            assert np.all(np.abs(E-A) < u), \
                f"X\n{X}\nstandardized\n{E}\nneeds\n{A}\n"


def test_010_standardize_sd_is_zero():
    """
    Objective:
        Verify the standardize() function when SD is zero.
    Expected:
        standardized == (X-mean) == 0
    """
    name = "test_010_standardize"
    keepdims = True
    u = TYPE_FLOAT(1e-6)

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        row = np.random.uniform(
            -MAX_ACTIVATION_VALUE,
            MAX_ACTIVATION_VALUE,
            M
        ).astype(TYPE_FLOAT)
        X = np.ones((N, M)).astype(TYPE_FLOAT) * row
        assert X.dtype == TYPE_FLOAT
        Logger.debug("%s: X \n%s\n", name, X)

        # Constraint: standardize(X) == (X - np.mean(A)) / np.std(X)
        ddof = 1 if N > 1 else 0
        sd = np.std(X, axis=0, keepdims=keepdims, ddof=ddof)
        assert np.allclose(sd, TYPE_FLOAT(0), atol=u, rtol=0)

        # Expected
        mean = np.mean(X, axis=0).astype(TYPE_FLOAT)
        E = (X - mean).astype(TYPE_FLOAT)
        assert np.allclose(E, TYPE_FLOAT(0), atol=u, rtol=0)

        # Actual
        A, __mean, __sd, _ = standardize(X, keepdims=keepdims)

        # Constraint. mean/sd should be same
        assert np.allclose(mean, __mean, atol=u)
        assert np.allclose(__sd, TYPE_FLOAT(1.0), atol=u)
        assert np.all(np.abs(E-A).astype(TYPE_FLOAT) < u), \
            f"X\n{X}\nstandardized\n{E}\nneeds\n{A}\n"


def test_010_standardize_eps(caplog):
    """
    Objective:
        Verify the standardize() function with eps
    Expected:
        standardize(X) = (X - np.mean(X)) / sqrt(variance + eps)
    """
    name = "test_010_standardize"
    keepdims = True

    # Test eps
    u = TYPE_FLOAT(1e-3)
    for i in range(NUM_MAX_TEST_TIMES):
        eps = np.random.uniform(1e-12, 1e-7)
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        X = np.random.uniform(
            -MAX_ACTIVATION_VALUE,
            MAX_ACTIVATION_VALUE,
            (N, M)
        ).astype(TYPE_FLOAT)
        Logger.debug("%s: X \n%s\n", name, X)

        # Constraint: standardize(X) == (X - np.mean(A)) / np.std(X)
        ddof = 1 if N > 1 else 0
        variance = np.var(X, axis=0, keepdims=keepdims, ddof=ddof)
        sd = np.sqrt(variance + eps)
        if np.all(sd > 0):
            # Expected
            mean = np.mean(X, axis=0)
            E = (X - mean) / sd
            # Actual

            if (i % 20) == 0:
                backup = eps
                eps = TYPE_FLOAT(0)  # Test eps == 0 at 5 % of times
                npsd = np.std(X, axis=0, keepdims=keepdims, ddof=ddof)
                if np.all(npsd > TYPE_FLOAT(0)):
                    # **********************************************************************
                    # Constraint: numpy sd matches __sd from standardize()
                    # **********************************************************************
                    A, __mean, __sd, _ = standardize(X, keepdims=keepdims, eps=eps)
                    assert np.allclose(a=__sd, b=npsd, atol=u, rtol=TYPE_FLOAT(0))
                else:
                    eps = backup
                    continue

            A, __mean, __sd, _ = standardize(X, keepdims=keepdims, eps=eps)

            # **********************************************************************
            # Constraint. mean/sd should be same.
            # **********************************************************************
            assert np.allclose(mean, __mean, atol=u)
            assert np.allclose(sd, __sd, atol=u), \
                "expected sd\n%s\nactual\n%s\ndiff=\n%s\n" % (sd, __sd, (sd - __sd))
            assert np.allclose(E, A, atol=u), \
                f"X\n{X}\nstandardized\n{E}\nndiff\n{A-E}\n"


def test_010_standardize_sd_is_zero_eps():
    """
    Objective:
        Verify the standardize() function when SD is zero.
    Expected:
        standardized == (X-mean)/sqrt(eps)
    """
    name = "test_010_standardize"
    keepdims = True
    u = TYPE_FLOAT(1e-6)
    eps = TYPE_FLOAT(1e-8)

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        row = np.random.uniform(
            -MAX_ACTIVATION_VALUE,
            MAX_ACTIVATION_VALUE,
            M
        ).astype(TYPE_FLOAT)
        X = np.ones(shape=(N, M), dtype=TYPE_FLOAT) * row
        Logger.debug("%s: X \n%s\n", name, X)

        # Constraint: standardize(X) == (X - np.mean(A)) / np.std(X)
        ddof = TYPE_INT(1) if N > 1 else TYPE_INT(0)
        sd = np.std(X, axis=0, keepdims=keepdims, ddof=ddof)
        assert np.allclose(sd, TYPE_FLOAT(0), atol=u, rtol=0)

        # Expected
        mean = np.mean(X, axis=0, dtype=TYPE_FLOAT)
        E = (X - mean) / np.sqrt(eps, dtype=TYPE_FLOAT)

        # Actual
        A, __mean, __sd, _ = standardize(X, keepdims=keepdims, eps=eps)

        # Constraint. mean/sd should be same
        assert np.allclose(mean, __mean, atol=u, rtol=TYPE_FLOAT(0))
        assert np.allclose(__sd, np.sqrt(eps), atol=u, rtol=TYPE_FLOAT(0))
        assert np.all(np.abs(E-A) < u), \
            f"X\n{X}\nstandardized\n{E}\nneeds\n{A}\n"


def test_010_sigmoid():
    """Test Case for sigmoid
    """
    u = ACTIVATION_DIFF_ACCEPTANCE_VALUE
    assert sigmoid(np.array(TYPE_FLOAT(0), dtype=TYPE_FLOAT)) == TYPE_FLOAT(0.5)
    x = np.array([0.0, 0.6, 0., -0.5]).reshape((2, 2)).astype(TYPE_FLOAT)
    t = np.array([
        0.5,
        0.6456563062257954529091,
        0.5,
        0.3775406687981454353611
    ]).reshape((2, 2)).astype(TYPE_FLOAT)
    assert np.all(np.abs(t - sigmoid(x)) < u), \
        f"delta (t-x) is expected < {u} but {x-t}"


def test_010_softmax():
    """Test Case for sigmoid
    """
    u = ACTIVATION_DIFF_ACCEPTANCE_VALUE
    P = softmax(np.array([2.44756739, 2.13945115]).astype(TYPE_FLOAT))
    E = np.array([0.57642539, 0.42357461]).astype(TYPE_FLOAT)
    assert np.all(np.abs(P-E) < u)

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        X = MAX_ACTIVATION_VALUE * np.random.randn(N, M).astype(TYPE_FLOAT)
        np.all(np.isfinite(softmax(X)))

