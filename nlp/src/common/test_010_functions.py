from common import (
    numerical_jacobian,
    sigmoid,
    softmax,
    cross_entropy_log_loss
)
import numpy as np


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
        length = np.random.randint(2, 4)  # length > 1
        index = np.random.randint(0, length)
        # --------------------------------------------------------------------------------
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Xi = 1 and 1D OHE array T = P,
        # sum(-t * log(t)) -> 0 (log(1) = 0)
        # --------------------------------------------------------------------------------
        P = np.zeros(length)
        P[index] = 1
        T = P
        Z = cross_entropy_log_loss(P, T)    # log(P=1) -> 0
        assert np.all(Z < h), \
            f"cross_entropy_log_loss(1,1) is expected to be 0 but {Z}"

        # --------------------------------------------------------------------------------
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Xi = 1.
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


def test_002_numerical_gradient_avg(h:float = 1e-5):
    """Test Case for numerical gradient calculation for average function
    A Jacobian matrix whose element 1/N is expected where N is X.size
    """
    def f(X: np.ndarray):
        return np.average(X)

    for _ in range(100):
        # Batch input X of shape (N, M)
        n = np.random.randint(low=1, high=10, size=1)
        m = np.random.randint(low=1, high=10, size=1)
        X = np.random.randn(np.asscalar(n),np.asscalar(m))

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


