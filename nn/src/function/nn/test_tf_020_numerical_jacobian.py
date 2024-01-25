"""Test cases for numerical_jacobian function
WARNING:
    DO NOT reuse X/P and T e.g. T = P or T = X.
    Python passes references to an object and (X=X+h) in numerical_jacobian
    will change T if you do T = X, causing bugs because T=[0+1e-6, 1+1e-6]
    can be T=[1,1] for T.dtype=TYPE_LABEL.

"""
import logging
import os
import numpy as np
import tensorflow as tf
from common.constant import (
    TYPE_FLOAT,
    TYPE_LABEL,
    OFFSET_LOG,
    OFFSET_DELTA,
    BOUNDARY_SIGMOID,
    TYPE_NN_FLOAT,
    TYPE_NN_INT,
)
from function.nn.tf import (
    Function
)
from testing.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES,
    MAX_ACTIVATION_VALUE,
    GRADIENT_DIFF_ACCEPTANCE_RATIO,
    GRADIENT_DIFF_ACCEPTANCE_VALUE
)

# https://stackoverflow.com/questions/59265920
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Logger = logging.getLogger(__name__)
Logger.setLevel(logging.ERROR)


# ================================================================================
# Matmul 2D
# ================================================================================
def test_020_numerical_jacobian_test_020_matmul(caplog):
    """
    Objective:
        Verify the numerical_jacobian function returns correct dL/dX and dL/dW
        for L = objective(matmul(X, W)).
    Expected:
        For Y:(N,M) = matmul(X, W.T) where X:(N,D) and W:(M,D)
        dL/dW.T:(D,M) = X.T(D,N) @ dL/dY:(N,M)
        dL/dX:(N,D) = dL/dY:(N,M) @ W:(M,D)

        where dL/dY = ones(shape=(N,M))
    """
    for _ in tf.range(NUM_MAX_TEST_TIMES):
        N = tf.random.uniform(
            shape=(), minval=2, maxval=NUM_MAX_BATCH_SIZE, dtype=TYPE_NN_INT
        )
        D = tf.random.uniform(
            shape=(), minval=2, maxval=NUM_MAX_FEATURES, dtype=TYPE_NN_INT
        )
        M = tf.random.uniform(
            shape=(), minval=2, maxval=NUM_MAX_NODES, dtype=TYPE_NN_INT
        )
        X = tf.random.uniform(
            shape=(N, D), minval=-1, maxval=1, dtype=TYPE_NN_FLOAT
        )
        W = tf.Variable(tf.random.uniform(
            shape=(M, D), minval=-1, maxval=1, dtype=TYPE_NN_FLOAT
        ))

        def L(x):
            loss = Function.sum(
                x, axis=None, keepdims=False
            )
            return loss

        def function_X(x):
            return tf.linalg.matmul(
                a=x, b=W, transpose_a=False, transpose_b=True, adjoint_a=False, adjoint_b=False,
                a_is_sparse=False, b_is_sparse=False
            )

        def objective_X(x):
            return L(function_X(x))

        def function_W(w):
            return tf.linalg.matmul(
                a=X, b=w, transpose_a=False, transpose_b=True, adjoint_a=False, adjoint_b=False,
                a_is_sparse=False, b_is_sparse=False
            )

        def objective_W(w):
            return L(function_W(w))

        dY = tf.ones(shape=(N, M), dtype=TYPE_NN_FLOAT)

        # ********************************************************************************
        # Constraint: dL/dX:(N,D) = dL/dY:(N,M) @ W:(M,D)
        # ********************************************************************************
        EDX = tf.linalg.matmul(a=dY, b=W, transpose_a=False, transpose_b=False)
        dX = Function.numerical_jacobian(objective_X, X)
        try:
            tf.debugging.assert_near(
                dX,
                EDX,
                atol=tf.constant(1e-5),
                message="Expected W \n%s\nActual\n%s\nDiff %s" % (EDX, dX, (EDX - dX))
            )
        except tf.errors.InvalidArgumentError as e:
            raise AssertionError(str(e))

        # ********************************************************************************
        # Constraint: dL/dW.T:(D,M) = X.T(D,N) @ dL/dY:(N,M)
        # ********************************************************************************
        EDW = tf.transpose(     # Expected dL/dY:(M,D)
            tf.linalg.matmul(a=X, b=dY, transpose_a=True, transpose_b=False)    # (D,M)
        )
        dW = Function.numerical_jacobian(objective_W, W)
        try:
            tf.debugging.assert_near(
                dW,
                EDW,
                atol=tf.constant(1e-5),
                message="Expected W \n%s\nActual\n%s\nDiff %s" % (EDW, dW, (EDW-dW))
            )
        except tf.errors.InvalidArgumentError as e:
            raise AssertionError(str(e))
