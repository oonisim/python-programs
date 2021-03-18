import logging
import numpy as np
from common import (
    standardize,
    logarithm,
    sigmoid,
    softmax,
    cross_entropy_log_loss,
    OFFSET_LOG,
    OFFSET_DELTA
)

from common.test_config import (
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
        standardize(X) == (X - np.mean(A)) / np.std(X)  (when np.std(X) != 0)
    """
    name = "test_010_standardize"
    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M: int = np.random.randint(2, NUM_MAX_NODES)
        X = np.random.randint(0, 1000, (N, M)).astype(float)
        Logger.debug("%s: X \n%s\n", name, X)

        # Constraint: standardize(X) == (X - np.mean(A)) / np.std(X)
        sd = np.std(X, axis=0)
        if np.all(sd > 0):
            # Expected
            mean = np.mean(X, axis=0)
            E = (X - mean) / sd
            # Actual
            A, __mean, __sd = standardize(X)

            # Constraint. mean/sd should be same
            assert np.all(np.abs(mean - __mean) < 1e-6)
            assert np.all(np.abs(sd - __sd) < 1e-6)
            assert np.all(np.abs(E-A) < 1e-6), \
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
    P = softmax(np.array([2.44756739, 2.13945115]))
    E = np.array([0.57642539, 0.42357461])
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
        T = np.zeros(length, dtype=int)

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

