import random
import numpy as np
from common import (
    numerical_jacobian,
    logarithm,
    sigmoid,
    softmax,
    cross_entropy_log_loss,
    gn,
    OFFSET_LOG,
    OFFSET_DELTA
)

from layer.test_config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE
)


def test_010_sigmoid(h:float = 1e-5):
    """Test Case for sigmoid
    """
    assert sigmoid(np.array(0.0)) == 0.5
    x = np.array([0.0, 0.6, 0., -0.5]).reshape((2,2))
    t = np.array([0.5, 0.6456563062257954529091, 0.5, 0.3775406687981454353611]).reshape((2,2))
    assert np.all(np.abs(t - sigmoid(x)) < h), f"delta (t-x) is expected < {h} but {x-t}"


def test_010_softmax(h:float = 1e-5):
    """Test Case for sigmoid
    """
    P = softmax(np.array([2.44756739, 2.13945115]))
    E = np.array([0.57642539, 0.42357461])
    assert np.all(np.abs(P-E) < h)


def test_010_cross_entropy_log_loss(h: float = 1e-5):
    """Test case for cross_entropy_log_loss
    log(P=1) -> 0
    """
    # --------------------------------------------------------------------------------
    # [Scalar test case]
    # For scalar P=1, T=1 for -t * log(P) -> log(1+k)
    # For scalar P=0, T=1 for -t * log(k) -> log(k)
    # For scalar P=1, T=0 for -t * log(k) -> 0
    # --------------------------------------------------------------------------------
    assert np.abs(cross_entropy_log_loss(P=1.0, T=1) - logarithm(1)) < h
    assert np.abs(cross_entropy_log_loss(P=0.0, T=1) - (-1 * logarithm(0))) < h
    assert cross_entropy_log_loss(P=1.0, T=0) < h

    for _ in range(NUM_MAX_TEST_TIMES):
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1 and 1D OHE array T = P,
        # sum(-t * log(p)) -> 0 (log(1) = 0)
        length = np.random.randint(2, NUM_MAX_NODES)  # length > 1
        index = np.random.randint(0, length)

        P = np.zeros(length)
        P[index] = 1.0
        T = index
        Z = cross_entropy_log_loss(P, T)    # log(P=1+k) -> log(k)
        assert np.all(np.abs(Z - logarithm(1)) < h), \
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
        E = -1 * logarithm(0)
        Z = cross_entropy_log_loss(P, T)
        assert (Z -E) < h, \
            f"cross_entropy_log_loss(1=0,T=0) is expected to be inf but {Z}"

        # --------------------------------------------------------------------------------
        # [2D test case]
        # P:(N, M) is probability matrix where Pnm = p, 0 <=n<N-1, 0<=m<M-1
        # T:(N,)   is index label where Tn=m is label as integer e.g. m=3 for 3rd label.
        # Pnm = log(P[i][j] + k)
        # --------------------------------------------------------------------------------
        p = np.random.uniform(0, 1)
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
        assert np.all(np.abs(E-L) < h), \
            f"Loss deviation (E-L) is expected to be < {h} but {np.abs(E-L)}"


def test_gn(X, T):
    N = 1000
    left = 5
    right = 10

    X = np.linspace(left, right, N)
    T0 = np.zeros(N)
    T1 = np.ones(N)
    gn(X, 0)
