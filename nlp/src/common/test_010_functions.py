import random
import numpy as np
from common import (
    numerical_jacobian,
    sigmoid,
    softmax,
    cross_entropy_log_loss,
    OFFSET_FOR_LOG,
    OFFSET_FOR_DELTA
)

NUM_MAX_NODES: int = 1000
NUM_MAX_BATCH_SIZE: int = 1000


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


def test_010_cross_entropy_log_loss(h: float = 1e-5, k=OFFSET_FOR_LOG):
    """Test case for cross_entropy_log_loss
    log(P=1) -> 0
    """
    # --------------------------------------------------------------------------------
    # [Scalar test case]
    # For scalar P=1, T=1 for -t * log(P) -> 0 (log(1) = 0)
    # For scalar P=0, T=1 for -t * log(e) -> log(e)
    # For scalar P=1, T=0 for -t * log(e) -> 0
    # --------------------------------------------------------------------------------
    assert cross_entropy_log_loss(P=1.0, T=1) < h
    assert np.abs(cross_entropy_log_loss(P=0.0, T=1) - (-1 * np.log(e))) < h
    assert cross_entropy_log_loss(P=1.0, T=0) < h

    for _ in range(100):
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1 and 1D OHE array T = P,
        # sum(-t * log(p)) -> 0 (log(1) = 0)
        length = np.random.randint(2, 100)  # length > 1
        index = np.random.randint(0, length)

        P = np.zeros(length)
        P[index] = 1
        T = P
        Z = cross_entropy_log_loss(P, T)    # log(P=1) -> 0
        assert np.all(Z < h), \
            f"cross_entropy_log_loss(1,1) is expected to be 0 but {Z}"

        # --------------------------------------------------------------------------------
        # [1D test case]
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1.
        # For 1D OHE array T [0, 0, ..., 0, 1, ...] where Tj = 1 and i != j
        # sum(-t * log(p)) -> log(e)
        # --------------------------------------------------------------------------------
        P = np.zeros(length)
        T = np.zeros(length)

        P[index] = 1
        while (position:= np.random.randint(0, length)) == index: pass
        T[position] = 1

        # Z will not get to np.inf as the function avoid it by adding a small number e
        # log(+e)
        E = -1 * np.log(e)
        Z = cross_entropy_log_loss(P, T)
        assert (Z -E) < h, \
            f"cross_entropy_log_loss(1=0,T=0) is expected to be inf but {Z}"

        # --------------------------------------------------------------------------------
        # [2D test case]
        # P:(N, M) is probability matrix where Pnm = p, 0 <=n<N-1, 0<=m<M-1
        # T:(N,)   is index label where Tn=m is label as integer e.g. m=3 for 3rd label.
        # Pnm = log(P[i][j] + e)
        # --------------------------------------------------------------------------------
        p = np.random.uniform(0, 1)
        N = np.random.randint(2, NUM_MAX_BATCH_SIZE)
        M = np.random.randint(2, NUM_MAX_NODES)
        T = np.random.randint(0, M, N)  # N rows of labels, max label value is M-1
        P = np.zeros((N, M))
        P[
            range(N),                    # Set p at random row position
            T
        ] = p

        E = np.full(T.shape, -np.log(p+e))
        L = cross_entropy_log_loss(P, T)
        assert np.all(np.abs(E-L) < h), \
            f"Loss deviation (E-L) is expected to be < {h} but {np.abs(E-L)}"