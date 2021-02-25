"""Test cases for numerical_jacobian function
WARNING:
    DO NOT reuse X/P and T e.g. T = P or T = X.
    Python passes references to an object and (X=X+h) in numerical_jacobian
    will change T if you do T = X, causing bugs because T=[0+1e-6, 1+1e-6]
    can be T=[1,1] for T.dtype=int.

"""
import random
import copy
from math import e
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

from common.test_config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    MAX_ACTIVATION_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO
)


def test_020_numerical_jacobian_avg(h:float = 1e-5):
    """Test Case for numerical gradient calculation for average function
    A Jacobian matrix whose element 1/N is expected where N is X.size
    """
    def f(X: np.ndarray):
        return np.average(X)

    for _ in range(NUM_MAX_TEST_TIMES):
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
    for _ in range(NUM_MAX_TEST_TIMES):
        x = np.random.uniform(low=-10, high=10, size=1)
        y = sigmoid(x)
        t = np.multiply(y, (1-y))
        z = numerical_jacobian(sigmoid, x)

        assert np.all((t - z) < h), f"delta (t-x) is expected < {h} but {x-t}"


def test_020_cross_entropy_log_loss_scalar(h: float = 1e-5, k=OFFSET_FOR_LOG):
    """Test case for cross_entropy_log_loss
    Args:
        h: threshold value above which the delta between Expectation E and Actual A
           is regarded as an error.
        k: A small value to add as log(x+k) to avoid log(0) -> -inf.
           The value must match with the one used in the cross_entropy_log_loss.
    -dlog(x)/dx = -1/x
    """
    # --------------------------------------------------------------------------------
    # [Scalar test case]
    # 1. For scalar P=1, T=1 for -t * log(1+k) -> log(1+k) : dlog(P+k)/dP -> 1 / (1+k)
    # 2. For scalar P=0, T=1 for -t * log(k) -> log(k)     : dlog(P+k)/dP -> 1 / (k)
    # 3. For scalar P=1, T=0 for -t * log(k) -> 0          : df/dP -> 0
    # --------------------------------------------------------------------------------
    def f(P: np.ndarray, T: np.ndarray):
        return np.sum(cross_entropy_log_loss(P, T))

    p1 = np.array(1.0, dtype=float)
    t1 = np.array(1, dtype=int)
    g1 = numerical_jacobian(partial(f, T=1), p1)
    # Expected numerical gradient
    e1 = (-t1 * np.log(p1+h+k) + t1 * np.log(p1-h+k)) / (2 * h)

    assert np.abs(e1 - g1) < h, \
        f"Delta expected to be < {h} but {np.abs(e1 - g1)}"

    # Analytical gradient that E1 needs to be close to.
    a1 = - 1 / (p1 + k)
    check.less_equal(np.abs(a1 - g1), 0.001, "a1-g1 %s\n" % np.abs(a1-g1))

    # --------------------------------------------------------------------------------
    # Not test P=0 because the gradient log(k +/- h) in numerical gradient value is
    # too large for h=1e-7 because log(1e-5)=-11.512925464970229, causing the gradient
    # to explode.
    # --------------------------------------------------------------------------------
    # p2 = 0.0
    # g2 = numerical_jacobian(partial(f, T=1), p2)
    # E2 = - 1 / k
    # assert np.abs(E2 - g2) < h, \
    #    f"Delta expected to be < {h} but {np.abs(E2 - g2)}"

    p3 = 1.0
    g3 = numerical_jacobian(partial(f, T=0), p3)
    E3 = 0      # dlog(k)/dP = 0 (derivative of constant is 0)
    assert np.abs(E3 - g3) < h, \
        f"Delta expected to be < {h} but {np.abs(E3 - g3)}"

    for _ in range(NUM_MAX_TEST_TIMES):
        p = np.random.uniform(0.2, 1)
        g = numerical_jacobian(partial(f, T=1), p)
        ex = (-t1 * np.log(p+h+k) + t1 * np.log(p-h+k)) / (2 * h)
        assert np.abs(ex-g) < h, \
            f"Delta expected to be < {h} but {np.abs(ex-g)}"

        a = - 1 / (p+k)
        check.less_equal(np.abs(a-g), 0.001, "ex-a %s\n" % np.abs(a-g))


def test_020_cross_entropy_log_loss_1d(h: float = 1e-5, k=OFFSET_FOR_LOG):
    """Test case for cross_entropy_log_loss for 1D
    log(P=1) -> 0
    dlog(x)/dx = 1/x
    """

    def f(P: np.ndarray, T: np.ndarray):
        return np.sum(cross_entropy_log_loss(P, T))

    # --------------------------------------------------------------------------------
    # p = 1
    # --------------------------------------------------------------------------------
    P1 = np.zeros(10)
    P1[5] = 1.0
    T1 = np.zeros(10).astype(int)
    T1[5] = 1

    E1 = np.zeros_like(P1)
    E1[5] = (-1 * np.log(1.0 + h + k) + 1 * np.log(1.0 - h + k)) / (2 * h)

    G1 = numerical_jacobian(partial(f, T=T1), P1)
    assert G1.shape == E1.shape
    assert np.all(np.abs(E1 - G1) < h), \
        f"Delta expected to be < {h} but \n{np.abs(E1 - G1)}"

    A1 = np.zeros_like(P1)
    A1[5] = -1 / (1.0 + k)
    check.equal(np.all(np.abs(A1 - G1) < 0.001), True, "A1-G1 %s\n" % np.abs(A1 - G1))

    # --------------------------------------------------------------------------------
    # p = np uniform()
    # --------------------------------------------------------------------------------
    for _ in range(NUM_MAX_TEST_TIMES):
        M = length = np.random.randint(2, NUM_MAX_NODES)    # length > 1
        T1 = np.random.randint(0, length)            # location of the truth

        p = np.random.uniform(0.2, 1.0)
        P1 = np.zeros(length)
        P1[T1] = p

        # --------------------------------------------------------------------------------
        # The Jacobian G shape is the same with P.shape.
        # G:[0, 0, ...,g, 0, ...] where Gi is numerical gradient close to -1/(1+k).
        # --------------------------------------------------------------------------------
        E1 = np.zeros_like(P1)
        E1[T1] = (-1 * np.log(p + h + k) + 1 * np.log(p - h + k)) / (2 * h)

        G1 = numerical_jacobian(partial(f, T=T1), P1)
        assert G1.shape == E1.shape
        assert np.all(np.abs(E1-G1) < h), \
            f"Delta expected to be < {h} but \n{np.abs(E1-G1)}"

        A1 = np.zeros_like(P1)
        A1[T1] = -1 / (p+k)
        check.equal(np.all(np.abs(A1 - G1) < 0.001), True, "A1-G1 %s\n" % np.abs(A1-G1))

    for _ in range(NUM_MAX_TEST_TIMES):
        # --------------------------------------------------------------------------------
        # [1D test case]
        # P:[0, 0, ..., 1, 0, ...] where Pi = 1
        # T:[0, 0, ..., 1, 0, ...] is OHE label where Ti=1
        # sum(-t * log(p+k)) -> log(1+k)
        # dlog(P+k)/dP -> -1 / (1+k)
        # --------------------------------------------------------------------------------
        length = np.random.randint(2, NUM_MAX_NODES)    # length > 1
        index = np.random.randint(0, length)            # location of the truth

        p = np.random.uniform(0.2, 1.0)
        P1 = np.zeros(length)
        P1[index] = p
        T1 = np.zeros(length).astype(int)   # OHE index
        T1[index] = int(1)

        # --------------------------------------------------------------------------------
        # The Jacobian G shape is the same with P.shape.
        # G:[0, 0, ...,g, 0, ...] where Gi is numerical gradient close to -1/(1+k).
        # --------------------------------------------------------------------------------
        E1 = np.zeros_like(P1)
        E1[index] = (-1 * np.log(p + h + k) + 1 * np.log(p - h + k)) / (2 * h)

        G1 = numerical_jacobian(partial(f, T=T1), P1)
        assert G1.shape == E1.shape
        assert np.all(np.abs(E1-G1) < h), \
            f"Delta expected to be < {h} but \n{np.abs(E1-G1)}"

        A1 = np.zeros_like(P1)
        A1[index] = -1 / (p+k)
        check.equal(np.all(np.abs(A1 - G1) < 0.001), True, "A1-G1 %s\n" % np.abs(A1-G1))

        # --------------------------------------------------------------------------------
        # [1D test case]
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1.
        # For 1D OHE array T [0, 0, ..., 0, 1, ...] where Tj = 1 and i != j
        # sum(-t * log(0+k)) -> log(+k)
        # dlog(P)/dP -> -1 / k
        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # Not test P=0 because the gradient log(k +/- h) in numerical gradient value is
        # too large for h=1e-7 because log(1e-5)=-11.512925464970229, causing the gradient
        # to explode.
        # --------------------------------------------------------------------------------
        # P2 = np.zeros(length)
        # T2 = np.zeros(length)

        # P2[index] = 1
        # while (position:= np.random.randint(0, length)) == index: pass
        # T2[position] = 1
        # E2 = np.full(P2.shape, -1 / k)
        # G2 = numerical_jacobian(partial(f, T=T2), P2)

        # assert E2.shape == G2.shape, \
        #     f"Jacobian shape is expected to be {E2.shape} but {G2.shape}."
        # assert np.all(np.abs(E2 - G2) < h), \
        #     f"Delta expected to be < {h} but \n{np.abs(E2-G2)}"


def test_020_cross_entropy_log_loss_2d(h: float = 1e-5, k=OFFSET_FOR_LOG):
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

    for _ in range(NUM_MAX_TEST_TIMES):
        # --------------------------------------------------------------------------------
        # [2D test case]
        # P:(N, M) is probability matrix where Pnm = p, 0 <=n<N-1, 0<=m<M-1
        # T:(N,)   is index label where Tn=m is label as integer k.g. m=3 for 3rd label.
        # Pnm = log(P[i][j] + k)
        # L = -log(p + k), -> dlog(P)/dP -> -1 / (p+k)
        #
        # Keep p value away from 0. As p gets close to 0, the log(p+/-h) gets large e.g
        # -11.512925464970229, hence log(p+/-h) / 2h explodes.
        # --------------------------------------------------------------------------------
        p = np.random.uniform(0.2, 1)
        N = np.random.randint(1, NUM_MAX_BATCH_SIZE)
        M = np.random.randint(2, NUM_MAX_NODES)
        # label index, not OHE
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
        ] = (-1 * np.log(p + h + k) + 1 * np.log(p - h + k)) / (2 * h)

        G = numerical_jacobian(partial(f, T=T), P)
        assert E.shape == G.shape, \
            f"Jacobian shape is expected to be {E.shape} but {G.shape}."
        assert np.all(np.abs(E-G) < h), \
            f"Delta expected to be < {h} but \n{np.abs(E-G)}"

        A = np.zeros_like(P)
        A[
            range(N),                    # Set p at random row position
            T
        ] = -1 / (p+k)

        check.equal(np.all(np.abs(A-G) < 0.001), True, "A-G %s\n" % np.abs(A-G))


# ================================================================================
# Softmax + log loss
# ================================================================================
def test_020_softmax_scalar(h=OFFSET_FOR_DELTA, k=OFFSET_FOR_LOG, r=GRADIENT_DIFF_ACCEPTANCE_RATIO):
    """Test case for softmax plus log loss with a scalar input
    Analytica gradient dj/dx -> softmax(x) - t
    j = cross_entropy_log_loss(softmax(x), T)
    dlog(x)/dx -> 1/x, dexp(x)/dx -> exp(x)

    Args:
        h: threshold value above which the delta between Expectation E and Actual A
           is regarded as an error.
        k: A small value to add as log(x+k) to avoid log(0) -> -inf.
           The value must match with the one used in the cross_entropy_log_loss.
    """
    # --------------------------------------------------------------------------------
    # [Scalar test case]
    # p = softmax(a)
    # 1. For scalar a, T=1 for -t * log(p+k)           : df/da
    # 3. For scalar a, T=0 for -t * log(p+k) -> 0      : df/da -> 0
    # --------------------------------------------------------------------------------
    def L(X: np.ndarray, T: np.ndarray):
        """Loss function"""
        return np.sum(cross_entropy_log_loss(softmax(X), T))

    def gn(a, t):
        """Expected numerical gradient"""
        if t == 0: return 0
        return (-np.log(softmax(a) + h + k) + np.log(softmax(a) - h + k)) / (2*h)

    for _ in range(NUM_MAX_TEST_TIMES):
        # activation value from an activation function (ReLU, etc)
        T = np.random.randint(0, 1)
        A = np.random.uniform(float(-MAX_ACTIVATION_VALUE), float(MAX_ACTIVATION_VALUE))
        G = numerical_jacobian(partial(L, T=T), A)
        E = gn(A, T)
        assert np.abs(E - G) <= np.abs(r * E), \
            f"Delta expected to be < {np.abs(r * E)} ratio of E but {np.abs((E - G))}"

        # --------------------------------------------------------------------------------
        # Analytical gradient dL/dA = dL/dJ * dJ/dA. dJ/dA = -T/P at the log loss.
        # For one input A, dJ/dA is always 0 when T = 0.
        # --------------------------------------------------------------------------------
        dA = softmax(A) - T if T == 1 else 0
        check.less_equal(
            np.abs(dA-G), np.abs(dA * r),
            "T %s A % s dA %s G %s dA-G %s \n" % (T, A, dA, G, np.abs(dA-G))
        )


def test_020_softmax_1d(r: float = GRADIENT_DIFF_ACCEPTANCE_RATIO, k=OFFSET_FOR_LOG):
    """Test case for softmax for 1D
    Verify the delta between the analytical and numerical gradient is small (<h).
    """
    def L(X: np.ndarray, T: np.ndarray):
        """Loss function for the softmax input A (activations)
        Args:
            X: Softmax input A
            T: Labels
        Returns:
            L: Loss value of from the softmax with log loss.
        """
        return np.sum(cross_entropy_log_loss(softmax(X), T))

    for _ in range(NUM_MAX_TEST_TIMES):
        N: int = np.random.randint(1, NUM_MAX_BATCH_SIZE)   # Batch size
        M: int = np.random.randint(2, NUM_MAX_NODES)        # Number of activations
        T = np.random.randint(0, M, N)  # N rows of labels, max label value is M-1
        A = np.random.uniform(-float(MAX_ACTIVATION_VALUE), float(MAX_ACTIVATION_VALUE), (N, M))
        G = numerical_jacobian(partial(L, T=T), A)

        # Analytical gradient P-T
        dA = softmax(A)
        dA[
            np.arange(N),
            T
        ] -= 1

        threshold = r * dA   # With 5 % difference of dL/dA
        assert np.all(np.abs(dA - G) <= np.abs(threshold)), \
            f"For T {T} and A: \n{A}\n the delta between G{G} and dA: \n{dA}\nwas expected to be < {threshold} but \n{np.abs(dA-G)}"
