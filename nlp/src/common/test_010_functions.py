import random
import numpy as np
from common import (
    numerical_jacobian,
    sigmoid,
    softmax,
    cross_entropy_log_loss
)

NUM_MAX_NODES: int = 1000
NUM_MAX_BATCH_SIZE: int = 1000


def test_001_sigmoid(h:float = 1e-5):
    """Test Case for sigmoid
    """
    assert sigmoid(np.array(0)) == 0.5
    x = np.array([0.0, 0.6, 0., -0.5]).reshape((2,2))
    t = np.array([0.5, 0.6456563062257954529091, 0.5, 0.3775406687981454353611]).reshape((2,2))
    assert np.all(np.abs(t - sigmoid(x)) < h), f"delta (t-x) is expected < {h} but {x-t}"


def test_001_cross_entropy_log_loss(h: float = 1e-5):
    """Test case for cross_entropy_log_loss
    log(P=1) -> 0
    """
    for _ in range(100):
        # --------------------------------------------------------------------------------
        # [Scalar test case]
        # For scalar P=1, T=1 for -t * log(t) -> 0 (log(1) = 0)
        # For scalar P=0, T=1 for -t * log(e) -> log(e)
        # For scalar P=1, T=0 for -t * log(e) -> 0
        # --------------------------------------------------------------------------------
        e = 1e-7
        assert cross_entropy_log_loss(P=1., T=1) < h
        assert np.abs(cross_entropy_log_loss(P=0., T=1, e=e) - (-1 * np.log(e)))< h
        assert cross_entropy_log_loss(P=1., T=0, e=e) < h

        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1 and 1D OHE array T = P,
        # sum(-t * log(t)) -> 0 (log(1) = 0)
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
        # sum(-t * log(t)) -> -np.inf (log(1) = -np.inf)
        # --------------------------------------------------------------------------------
        P = np.zeros(length)
        T = np.zeros(length)

        while (position:= np.random.randint(0, length)) == index: pass
        T[position] = 1

        # Z will not get to np.inf as the function avoid it by adding a small number e
        # log(+e)
        E = -1 * np.log(1e-7)
        Z = cross_entropy_log_loss(P, T)
        assert (Z -E) < h, \
            f"cross_entropy_log_loss(1=0,T=0) is expected to be inf but {Z}"

        # --------------------------------------------------------------------------------
        # [2D test case]
        # P:(N, M) is probability matrix where Pnm = p, 0 <=n<N-1, 0<=m<M-1
        # T:(N,)   is index label where Tn=m is label as integer e.g. m=3 for 3rd label.
        # Pnm = log(P[i][j] + e)
        # --------------------------------------------------------------------------------
        e = 1e-7
        p = np.random.uniform(0, 1)
        N = np.random.randint(2, NUM_MAX_BATCH_SIZE)
        M = np.random.randint(2, NUM_MAX_NODES)
        T = np.random.randint(0, M, N)  # N rows of labels, max label value is M-1
        P = np.zeros((N, M))
        P[
            range(N),                    # Set p at random row position
            T
        ] = p

        E = np.full(T.shape, np.log(p+e))
        L = cross_entropy_log_loss(P, T, e=e)
        assert np.all(np.abs(E-L) < h), \
            f"Loss deviation (E-L) is expected to be < {h} but {np.abs(E-L)}"


def test_002_numerical_gradient_avg(h:float = 1e-5):
    """Test Case for numerical gradient calculation for average function
    A Jacobian matrix whose element 1/N is expected where N is X.size
    """
    def f(X: np.ndarray):
        return np.average(X)

    for _ in range(100):
        # Batch input X of shape (N, M)
        n = np.random.randint(low=1, high=10)
        m = np.random.randint(low=1, high=10)
        X = np.random.randn(n,m)

        # Expected gradient matrix of shape (N,M) as label T
        T = np.full(X.shape, 1 / X.size)
        # Jacobian matrix of shame (N,M), each element of which is df/dXij
        J = numerical_jacobian(f, X)

        assert np.all(np.abs(T - J) < h), \
            f"(T - Z) < {h} is expected but {np.abs(T - J)}."


def test_002_numerical_gradient_sigmoid(h:float = 1e-5):
    """Test Case for numerical gradient calculation
    """
    # y=sigmoid(x) -> dy/dx = y(1-y)
    # 0.5 = sigmoid(0) -> dy/dx = 0.25
    for _ in range(100):
        x = np.random.uniform(low=-10, high=10, size=1)
        y = sigmoid(x)
        t = np.multiply(y, (1-y))
        z = numerical_jacobian(sigmoid, x)

        assert np.all((t - z) < h), f"delta (t-x) is expected < {h} but {x-t}"

