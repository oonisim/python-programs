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


