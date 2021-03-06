"""Test cases for numerical_jacobian function
WARNING:
    DO NOT reuse X/P and T e.g. T = P or T = X.
    Python passes references to an object and (X=X+h) in numerical_jacobian
    will change T if you do T = X, causing bugs because T=[0+1e-6, 1+1e-6]
    can be T=[1,1] for T.dtype=int.

"""
import logging
from functools import partial
import numpy as np
import pytest_check as check    # https://pypi.org/project/pytest-check/
from common import (
    numerical_jacobian,
    logarithm,
    sigmoid,
    softmax,
    cross_entropy_log_loss,
    OFFSET_LOG,
    OFFSET_DELTA,
    BOUNDARY_SIGMOID
)

from common.test_config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    MAX_ACTIVATION_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO
)


Logger = logging.getLogger(__name__)
Logger.setLevel(logging.ERROR)


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


def test_020_numerical_jacobian_sigmoid(u: float = 1e-4):
    """Test Case for numerical gradient calculation
    The domain of X is -BOUNDARY_SIGMOID < x < BOUNDARY_SIGMOID

    Args:
          u: Acceptable threshold value
    """
    # y=sigmoid(x) -> dy/dx = y(1-y)
    # 0.5 = sigmoid(0) -> dy/dx = 0.25
    for _ in range(NUM_MAX_TEST_TIMES):
        x = np.random.uniform(low=-BOUNDARY_SIGMOID, high=BOUNDARY_SIGMOID, size=1)
        y = sigmoid(x)
        analytical = np.multiply(y, (1-y))
        numerical = numerical_jacobian(sigmoid, x)

        difference = np.abs(analytical - numerical)
        acceptance = np.abs(analytical * GRADIENT_DIFF_ACCEPTANCE_RATIO)
        assert np.all(difference < max(u, acceptance)), \
            f"Needs difference < {max(u, acceptance)} but {difference}\nx is {x}"


def test_020_cross_entropy_log_loss_1d(h: float = OFFSET_DELTA, u: float = 1e-4):
    """
    Objective:
        Test the categorical log loss values for P in 1 dimension.

    Constraints:
        1. The numerical gradient gn = (-t * logarithm(p+h) + t * logarithm(p-h)) / 2h.
        2. The numerical gradient gn is within +/- u within the analytical g = -T/P.

    P: Probabilities from softmax of shape (M,)
    M: Number of nodes in the cross_entropy_log_loss layer.
    T: Labels

    Note:
        log(P=1) -> 0
        dlog(x)/dx = 1/x
    """
    def f(P: np.ndarray, T: np.ndarray):
        return np.sum(cross_entropy_log_loss(P, T))

    # --------------------------------------------------------------------------------
    # For (P, T): P[index] = True/1, OHE label T[index] = 1 where T = [0,0,0,...1,...0]
    # --------------------------------------------------------------------------------
    M: int = np.random.randint(2, NUM_MAX_NODES)
    index: int = np.random.randint(0, M)            # Position of the true label in P
    P1 = np.zeros(M)
    P1[index] = float(1.0)
    T1 = np.zeros(M, dtype=int)
    T1[index] = int(1)

    EGN1 = np.zeros_like(P1)    # Expected numerical gradient
    EGN1[index] = (-1 * logarithm(1.0 + h) + 1 * logarithm(1.0 - h)) / (2 * h)
    GN1 = numerical_jacobian(partial(f, T=T1), P1)

    # The numerical gradient gn = (-t * logarithm(p+h) + t * logarithm(p-h)) / 2h
    assert GN1.shape == EGN1.shape
    assert np.all(EGN1 == GN1), \
        f"Expected GN1==EGN1 but GN1-EGN1=\n{np.abs(GN1-EGN1)}"

    # The numerical gradient gn is within +/- u within the analytical g = -T/P
    G1 = np.zeros_like(P1)
    G1[T1 == 1] = -1 * (T1[index] / P1[index])
    # G1[T1 != 0] = 0
    check.equal(np.all(np.abs(G1 - GN1) < u), True, "G1-GN1 %s\n" % np.abs(G1 - GN1))

    # --------------------------------------------------------------------------------
    # For (P, T): P[index] = np uniform(), index label T=index
    # --------------------------------------------------------------------------------
    for _ in range(NUM_MAX_TEST_TIMES):
        M = np.random.randint(2, NUM_MAX_NODES)    # M > 1
        T1 = np.random.randint(0, M)            # location of the truth
        P1 = np.zeros(M)
        while not (x := np.random.uniform(low=-BOUNDARY_SIGMOID, high=BOUNDARY_SIGMOID)): pass
        p = softmax(x)
        P1[T1] = p

        # --------------------------------------------------------------------------------
        # The Jacobian G shape is the same with P.shape.
        # G:[0, 0, ...,g, 0, ...] where Gi is numerical gradient close to -1/(1+k).
        # --------------------------------------------------------------------------------
        EGN1 = np.zeros_like(P1)
        EGN1[T1] = -1 * (logarithm(p+h) - logarithm(p-h)) / (2 * h)
        GN1 = numerical_jacobian(partial(f, T=T1), P1)

        # The numerical gradient gn = (-t * logarithm(p+h) + t * logarithm(p-h)) / 2h
        assert GN1.shape == EGN1.shape
        assert np.all(np.abs(EGN1-GN1) < h), \
            f"Delta expected to be < {h} but \n{np.abs(EGN1-GN1)}"

        G1 = np.zeros_like(P1)
        G1[T1] = -1 / p

        # The numerical gradient gn is within +/- u within the analytical g = -T/P
        check.equal(np.all(np.abs(G1-GN1) < u), True, "G1-GN1 %s\n" % np.abs(G1-GN1))

    for _ in range(NUM_MAX_TEST_TIMES):
        # --------------------------------------------------------------------------------
        # [1D test case]
        # P:[0, 0, ..., 1, 0, ...] where Pi = 1
        # T:[0, 0, ..., 1, 0, ...] is OHE label where Ti=1
        # sum(-t * log(p+k)) -> log(1+k)
        # dlog(P+k)/dP -> -1 / (1+k)
        # --------------------------------------------------------------------------------
        M = np.random.randint(2, NUM_MAX_NODES)    # M > 1
        index = np.random.randint(0, M)            # location of the truth
        while not (x := np.random.uniform(low=-BOUNDARY_SIGMOID, high=BOUNDARY_SIGMOID)): pass
        p = softmax(x)
        P1 = np.zeros(M)
        P1[index] = p
        T1 = np.zeros(M).astype(int)   # OHE index
        T1[index] = int(1)

        # --------------------------------------------------------------------------------
        # The Jacobian G shape is the same with P.shape.
        # --------------------------------------------------------------------------------
        EGN1 = np.zeros_like(P1)
        EGN1[index] = (-1 * logarithm(p+h) + 1 * logarithm(p-h)) / (2 * h)
        GN1 = numerical_jacobian(partial(f, T=T1), P1)
        assert GN1.shape == EGN1.shape
        assert np.all(np.abs(EGN1-GN1) < h), \
            f"Delta expected to be < {h} but \n{np.abs(EGN1-GN1)}"

        G1 = np.zeros_like(P1)
        G1[index] = -1 / p
        check.equal(np.all(np.abs(G1-GN1) < u), True, "G1-GN1 %s\n" % np.abs(G1-GN1))

        # --------------------------------------------------------------------------------
        # [1D test case]
        # For 1D OHE array P [0, 0, ..., 1, 0, ...] where Pi = 1.
        # For 1D OHE array T [0, 0, ..., 0, 1, ...] where Tj = 1 and i != j
        # sum(-t * logarithm(0)) -> log(offset)
        # dlog(P)/dP -> -T / P
        # --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        # Not test P=0 because the gradient log(k +/- h) in numerical gradient value is
        # too large for h=1e-7 because log(1e-5)=-11.512925464970229, causing the gradient
        # to explode.
        # --------------------------------------------------------------------------------
        # P2 = np.zeros(M)
        # T2 = np.zeros(M)

        # P2[index] = 1
        # while (position:= np.random.randint(0, M)) == index: pass
        # T2[position] = 1
        # E2 = np.full(P2.shape, -1 / k)
        # G2 = numerical_jacobian(partial(f, T=T2), P2)

        # assert E2.shape == G2.shape, \
        #     f"Jacobian shape is expected to be {E2.shape} but {G2.shape}."
        # assert np.all(np.abs(E2 - G2) < h), \
        #     f"Delta expected to be < {h} but \n{np.abs(E2-G2)}"


def test_020_cross_entropy_log_loss_2d(h: float = OFFSET_DELTA):
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
        # Pnm = log(P[i][j])
        # L = -log(p), -> dlog(P)/dP -> -1 / (p)
        #
        # Keep p value away from 0. As p gets close to 0, the log(p+/-h) gets large e.g
        # -11.512925464970229, hence log(p+/-h) / 2h explodes.
        # --------------------------------------------------------------------------------
        while not (x := np.random.uniform(low=-BOUNDARY_SIGMOID, high=BOUNDARY_SIGMOID)): pass
        p = softmax(x)
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
        ] = (-1 * logarithm(p+h) + 1 * logarithm(p-h)) / (2 * h)

        G = numerical_jacobian(partial(f, T=T), P)
        assert E.shape == G.shape, \
            f"Jacobian shape is expected to be {E.shape} but {G.shape}."
        assert np.all(np.abs(E-G) < h), \
            f"Delta expected to be < {h} but \n{np.abs(E-G)}"

        A = np.zeros_like(P)
        A[
            range(N),                    # Set p at random row position
            T
        ] = -1 / p

        check.equal(np.all(np.abs(A-G) < 0.0001), True, "A-G %s\n" % np.abs(A-G))


# ================================================================================
# Softmax + log loss
# ================================================================================
def test_020_softmax_1d(r: float = GRADIENT_DIFF_ACCEPTANCE_RATIO, u: float = 1e-4):
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

        assert \
            np.all(np.abs(dA - G) < u) or \
            (np.all(np.abs(dA - G) < np.abs(GRADIENT_DIFF_ACCEPTANCE_RATIO * G))), \
            f"For T {T} and A: \n{A}\nthe delta between G\n{G}\n and dA: \n{dA}\nwas expected to be < {u} but \n{np.abs(dA-G)}"
