import sys
import logging
import numpy as np
from common.constant import (
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
        assert np.allclose(sd, 0.0, atol=u, rtol=0)

        # Expected
        mean = np.mean(X, axis=0).astype(TYPE_FLOAT)
        E = (X - mean).astype(TYPE_FLOAT)
        assert np.allclose(E, 0.0, atol=u, rtol=0)

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
    u = 1e-3
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
                eps = 0  # Test eps == 0 at 5 % of times
                npsd = np.std(X, axis=0, keepdims=keepdims, ddof=ddof)
                if np.all(npsd > 0.0):
                    # **********************************************************************
                    # Constraint: numpy sd matches __sd from standardize()
                    # **********************************************************************
                    A, __mean, __sd, _ = standardize(X, keepdims=keepdims, eps=eps)
                    assert np.allclose(a=__sd, b=npsd, atol=u, rtol=0)
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
    u = 1e-6
    eps = 1e-8

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        row = np.random.uniform(
            -MAX_ACTIVATION_VALUE,
            MAX_ACTIVATION_VALUE,
            M
        ).astype(TYPE_FLOAT)
        X = np.ones((N, M)) * row
        Logger.debug("%s: X \n%s\n", name, X)

        # Constraint: standardize(X) == (X - np.mean(A)) / np.std(X)
        ddof = 1 if N > 1 else 0
        sd = np.std(X, axis=0, keepdims=keepdims, ddof=ddof)
        assert np.allclose(sd, 0.0, atol=u, rtol=0)

        # Expected
        mean = np.mean(X, axis=0)
        E = (X - mean) / np.sqrt(eps)
        assert np.allclose(E, 0.0, atol=u, rtol=0)

        # Actual
        A, __mean, __sd, _ = standardize(X, keepdims=keepdims, eps=eps)

        # Constraint. mean/sd should be same
        assert np.allclose(mean, __mean, atol=u, rtol=0)
        assert np.allclose(__sd, np.sqrt(eps), atol=u, rtol=0)
        assert np.all(np.abs(E-A) < u), \
            f"X\n{X}\nstandardized\n{E}\nneeds\n{A}\n"


def test_010_sigmoid():
    """Test Case for sigmoid
    """
    u = ACTIVATION_DIFF_ACCEPTANCE_VALUE
    assert sigmoid(np.array(0.0)) == 0.5
    x = np.array([0.0, 0.6, 0., -0.5]).reshape((2,2))
    t = np.array([0.5, 0.6456563062257954529091, 0.5, 0.3775406687981454353611]).reshape((2,2))
    assert np.all(np.abs(t - sigmoid(x)) < u), f"delta (t-x) is expected < {u} but {x-t}"


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
        X = MAX_ACTIVATION_VALUE * np.random.randn(N, M)
        np.all(np.isfinite(softmax(X)))


def test_010_cross_entropy_log_loss():
    """
    Objective:
        Verify the cross_entropy_log_loss() function.
    Expected:
    """
    u = LOSS_DIFF_ACCEPTANCE_RATIO
    # --------------------------------------------------------------------------------
    # [Scalar test case]
    # For scalar P=1, T=1 for -t * log(P) -> log(1+k)
    # For scalar P=0, T=1 for -t * log(k) -> log(k)
    # For scalar P=1, T=0 for -t * log(k) -> 0
    # --------------------------------------------------------------------------------
    assert np.abs(cross_entropy_log_loss(P=1.0, T=1) - logarithm(1.0)) < u
    assert np.abs(cross_entropy_log_loss(P=0.0, T=1) - (-1 * logarithm(0.0))) < u
    assert cross_entropy_log_loss(P=1.0, T=0) < u

    for _ in range(NUM_MAX_TEST_TIMES):
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1 and 1D OHE array T = P,
        # sum(-t * log(p)) -> 0 (log(1) = 0)
        length = np.random.randint(2, NUM_MAX_NODES)  # length > 1
        index = np.random.randint(0, length)

        P = np.zeros(length)
        P[index] = 1.0
        T = index
        Z = cross_entropy_log_loss(P, T)    # log(P=1+k) -> log(k)
        assert np.all(np.abs(Z - logarithm(float(1))) < u), \
            f"cross_entropy_log_loss(1,1) is expected to be 0 but {Z}"

        # --------------------------------------------------------------------------------
        # [1D test case]
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1.
        # For 1D OHE array T [0, 0, ..., 0, 1, ...] where Tj = 1 and i != j
        # sum(-t * log(p)) -> log(k)
        # --------------------------------------------------------------------------------
        P = np.zeros(length)
        T = np.zeros(length, dtype=TYPE_LABEL)

        P[index] = 1.0
        while (position:= np.random.randint(0, length)) == index: pass
        T[position] = 1

        # Z will not get to np.inf as the function avoid it by adding a small number k
        # log(+k)
        E = -1 * logarithm(float(0))
        Z = cross_entropy_log_loss(P, T)
        assert (Z -E) < u, \
            f"cross_entropy_log_loss(1=0,T=0) is expected to be inf but {Z}"

        # --------------------------------------------------------------------------------
        # [2D test case]
        # P:(N, M) is probability matrix where Pnm = p, 0 <=n<N-1, 0<=m<M-1
        # T:(N,)   is index label where Tn=m is label as integer e.g. m=3 for 3rd label.
        # Pnm = log(P[i][j] + k)
        # --------------------------------------------------------------------------------
        p = np.random.uniform(OFFSET_LOG, 1-OFFSET_LOG)     # prevent log(0)
        N = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M = np.random.randint(2, NUM_MAX_NODES)
        T = np.random.randint(0, M, N)  # N rows of labels, max label value is M-1
        P = np.zeros((N, M))
        P[
            range(N),                    # Set p at random row position
            T
        ] = p

        E = np.full(T.shape, -logarithm(p))
        L = cross_entropy_log_loss(P, T)
        assert np.all(np.abs(E-L) < u * -logarithm(p)), \
            f"Loss deviation (E-L) is expected to be < {u} but {np.abs(E-L)}"

