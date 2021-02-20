"""Test cases for numerical_jacobian function
WARNING:
    DO NOT reuse X/P and T e.g. T = P or T = X.
    Python passes references to an object and (X=X+h) in numerical_jacobian
    will change T if you do T = X, causing bugs because T=[0+1e-6, 1+1e-6]
    can be T=[1,1] for T.dtype=int.

"""
import random
from functools import partial
import numpy as np
import pytest_check as check    # https://pypi.org/project/pytest-check/
from common import (
    numerical_jacobian,
    sigmoid,
    softmax,
    cross_entropy_log_loss,
    OFFSET_FOR_LOG,
    OFFSET_FOR_DELTA
)

NUM_MAX_BATCH_SIZE: int = 20 + 1
NUM_MAX_NODES: int = 20 + 1


def test_020_numerical_jacobian_avg(h:float = 1e-5):
    """Test Case for numerical gradient calculation for average function
    A Jacobian matrix whose element 1/N is expected where N is X.size
    """
    def f(X: np.ndarray):
        return np.average(X)

    for _ in range(100):
        # Batch input X of shape (N, M)
        n = np.random.randint(low=1, high=NUM_MAX_BATCH_SIZE)
        m = np.random.randint(low=1, high=NUM_MAX_NODES)
        X = np.random.randn(n,m)

        # Expected gradient matrix of shape (N,M) as label T
        T = np.full(X.shape, 1 / X.size)
        # Jacobian matrix of shame (N,M), each element of which is df/dXij
        J = numerical_jacobian(f, X)

        assert np.all(np.abs(T - J) < h), \
            f"(T - Z) < {h} is expected but {np.abs(T - J)}."


def test_020_numerical_jacobian_sigmoid(h:float = 1e-5):
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


def test_020_cross_entropy_log_loss_scalar(h: float = 1e-5, e=OFFSET_FOR_LOG):
    """Test case for cross_entropy_log_loss
    Args:
        h: threshold value above which the delte between Expectation E and Actual A
           is regarded as an error.
        e: A small value to add as log(x+e) to avoid log(0) -> -inf.
           The value must match with the one used in the cross_entropy_log_loss.
    -dlog(x)/dx = -1/x
    """
    # --------------------------------------------------------------------------------
    # [Scalar test case]
    # 1. For scalar P=1, T=1 for -t * log(1+e) -> log(1+e) : dlog(P+e)/dP -> 1 / (1+e)
    # 2. For scalar P=0, T=1 for -t * log(e) -> log(e)     : dlog(P+e)/dP -> 1 / (e)
    # 3. For scalar P=1, T=0 for -t * log(e) -> 0: loss = log(e) -> dlog(e)/dP -> 0
    # --------------------------------------------------------------------------------
    def f(P: np.ndarray, T: np.ndarray):
        return np.sum(cross_entropy_log_loss(P, T))

    p1 = 1.0
    t1 = int(1)
    g1 = numerical_jacobian(partial(f, T=1), p1)
    # Expected numerical gradient
    e1 = (-t1 * np.log(p1+h+e) + t1 * np.log(p1-h+e)) / (2 * h)

    assert np.abs(e1 - g1) < h, \
        f"Delta expected to be < {h} but {np.abs(e1 - g1)}"

    # Analytical gradient that E1 needs to be close to.
    a1 = - 1 / (p1 + e)
    check.less(np.abs(a1 - g1), 0.001, "a1-g1 %s\n" % np.abs(a1-g1))

    # --------------------------------------------------------------------------------
    # Not test P=0 because the gradient log(e +/- h) in numerical gradient value is
    # too large for h=1e-7 because log(1e-5)=-11.512925464970229, causing the gradient
    # to explode.
    # --------------------------------------------------------------------------------
    # p2 = 0.0
    # g2 = numerical_jacobian(partial(f, T=1), p2)
    # E2 = - 1 / e
    # assert np.abs(E2 - g2) < h, \
    #    f"Delta expected to be < {h} but {np.abs(E2 - g2)}"

    p3 = 1.0
    g3 = numerical_jacobian(partial(f, T=0), p3)
    E3 = 0      # dlog(e)/dP = 0 (derivative of constant is 0)
    assert np.abs(E3 - g3) < h, \
        f"Delta expected to be < {h} but {np.abs(E3 - g3)}"

    for _ in range(100):
        p = np.random.uniform(0.5,1)
        g = numerical_jacobian(partial(f, T=1), p)
        ex = (-t1 * np.log(p+h+e) + t1 * np.log(p-h+e)) / (2 * h)
        assert np.abs(ex-g) < h, \
            f"Delta expected to be < {h} but {np.abs(ex-g)}"

        a = - 1 / (p+e)
        check.less(np.abs(a-g), 0.001, "ex-a %s\n" % np.abs(a-g))


def test_020_cross_entropy_log_loss_1d(h: float = 1e-5, e=OFFSET_FOR_LOG):
    """Test case for cross_entropy_log_loss for 1D
    log(P=1) -> 0
    dlog(x)/dx = 1/x
    """

    def f(P: np.ndarray, T: np.ndarray):
        return np.sum(cross_entropy_log_loss(P, T))

    for _ in range(100):
        # --------------------------------------------------------------------------------
        # [1D test case]
        # P:[0, 0, ..., 1, 0, ...] where Pi = 1
        # T:[0, 0, ..., 1, 0, ...] is OHE label where Ti=1
        # sum(-t * log(p+e)) -> log(1+e)
        # dlog(P+e)/dP -> -1 / (1+e)
        # --------------------------------------------------------------------------------
        length = np.random.randint(2, NUM_MAX_NODES)    # length > 1
        index = np.random.randint(0, length)            # location of the truth

        P1 = np.zeros(length)
        P1[index] = 1.0
        T1 = np.zeros(length).astype(int)
        T1[index] = 1

        # --------------------------------------------------------------------------------
        # The Jacobian G shape is the same with P.shape.
        # G:[0, 0, ...,g, 0, ...] where Gi is numerical gradient close to -1/(1+e).
        # --------------------------------------------------------------------------------
        E1 = np.zeros_like(P1)
        E1[index] = (-1 * np.log(1.0 + h + e) + 1 * np.log(1.0 - h + e)) / (2 * h)

        G1 = numerical_jacobian(partial(f, T=T1), P1)
        assert G1.shape == E1.shape
        assert np.all(np.abs(E1-G1) < h), \
            f"Delta expected to be < {h} but \n{np.abs(E1-G1)}"

        A1 = np.zeros_like(P1)
        A1[index] = -1 / (1.0+e)
        check.equal(np.all(np.abs(A1 - G1) < 0.01), True, "A1-G1 %s\n" % np.abs(A1-G1))

        # --------------------------------------------------------------------------------
        # [1D test case]
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1.
        # For 1D OHE array T [0, 0, ..., 0, 1, ...] where Tj = 1 and i != j
        # sum(-t * log(0+e)) -> log(+e)
        # dlog(P)/dP -> -1 / e
        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # Not test P=0 because the gradient log(e +/- h) in numerical gradient value is
        # too large for h=1e-7 because log(1e-5)=-11.512925464970229, causing the gradient
        # to explode.
        # --------------------------------------------------------------------------------
        # P2 = np.zeros(length)
        # T2 = np.zeros(length)

        # P2[index] = 1
        # while (position:= np.random.randint(0, length)) == index: pass
        # T2[position] = 1
        # E2 = np.full(P2.shape, -1 / e)
        # G2 = numerical_jacobian(partial(f, T=T2), P2)

        # assert E2.shape == G2.shape, \
        #     f"Jacobian shape is expected to be {E2.shape} but {G2.shape}."
        # assert np.all(np.abs(E2 - G2) < h), \
        #     f"Delta expected to be < {h} but \n{np.abs(E2-G2)}"


def test_020_cross_entropy_log_loss_2d(h: float = 1e-5, e=OFFSET_FOR_LOG):
    """Test case for cross_entropy_log_loss for 2D
    log(P=1) -> 0
    dlog(x)/dx = 1/x
    """
    def f(P: np.ndarray, T: np.ndarray):
        """Loss function"""
        # For P.ndim==2 of shape (N, M), cross_entropy_log_loss() returns (N,).
        # Each of which has the loss for P[n].
        # If divided by P.shape[0] or N, the loss gets 1/N, which is wrong.
        # This is not a gradient function but a loss function.
        # return np.sum(cross_entropy_log_loss(P, T)) / P.shape[0]

        return np.sum(cross_entropy_log_loss(P, T))

    for _ in range(100):
        # --------------------------------------------------------------------------------
        # [2D test case]
        # P:(N, M) is probability matrix where Pnm = p, 0 <=n<N-1, 0<=m<M-1
        # T:(N,)   is index label where Tn=m is label as integer e.g. m=3 for 3rd label.
        # Pnm = log(P[i][j] + e)
        # L = -log(p + e), -> dlog(P)/dP -> -1 / (p+e)
        #
        # Keep p value away from 0. As p gets close to 0, the log(p+/-h) gets large e.g
        # -11.512925464970229, hence log(p+/-h) / 2h explodes.
        # --------------------------------------------------------------------------------
        p = np.random.uniform(0.5, 0.9)
        N = np.random.randint(2, NUM_MAX_BATCH_SIZE)
        M = np.random.randint(2, NUM_MAX_NODES)
        T = np.random.randint(0, M, N)  # N rows of labels, max label value is M-1
        P = np.zeros((N, M))
        P[
            range(N),                    # Set p at random row position
            T
        ] = p
        E = np.zeros_like(P)
        E[
            range(N),                    # Set p at random row position
            T
        ] = (-1 * np.log(p + h + e) + 1 * np.log(p - h + e)) / (2 * h)

        G = numerical_jacobian(partial(f, T=T), P)
        assert E.shape == G.shape, \
            f"Jacobian shape is expected to be {E.shape} but {G.shape}."
        assert np.all(np.abs(E-G) < h), \
            f"Delta expected to be < {h} but \n{np.abs(E-G)}"

        A = np.zeros_like(P)
        A[
            range(N),                    # Set p at random row position
            T
        ] = -1 / (p+e)

        check.equal(np.all(np.abs(A-G) < 0.001), True, "A-G %s\n" % np.abs(A-G))
