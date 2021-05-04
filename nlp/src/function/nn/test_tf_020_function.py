"""Test cases for numerical_jacobian function
WARNING:
    DO NOT reuse X/P and T e.g. T = P or T = X.
    Python passes references to an object and (X=X+h) in numerical_jacobian
    will change T if you do T = X, causing bugs because T=[0+1e-6, 1+1e-6]
    can be T=[1,1] for T.dtype=TYPE_LABEL.

"""
import logging
import os

import tensorflow as tf

from common.constant import (
    TYPE_NN_FLOAT,
    TYPE_NN_INT,
)
from common.function import (
    sigmoid_cross_entropy_log_loss
)
from function.nn.tf import (
    Function
)
from testing.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_BATCH_SIZE,
    NUM_MAX_FEATURES
)

# https://stackoverflow.com/questions/59265920
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Logger = logging.getLogger(__name__)
Logger.setLevel(logging.ERROR)


# ================================================================================
# Matmul 2D
# ================================================================================
def test_020_sigmoid_cross_entropy_log_loss(caplog):
    """
    Objective:
        Verify sigmoid_cross_entropy_log_loss function.
    Expected:
    """
    for _ in tf.range(NUM_MAX_TEST_TIMES):
        N = tf.random.uniform(
            shape=(), minval=2, maxval=NUM_MAX_BATCH_SIZE, dtype=TYPE_NN_INT
        )
        D = tf.random.uniform(
            shape=(), minval=2, maxval=NUM_MAX_FEATURES, dtype=TYPE_NN_INT
        )
        X = tf.random.uniform(
            shape=(N, 1), minval=-1, maxval=1, dtype=TYPE_NN_FLOAT
        )
        T = tf.random.uniform(shape=(N, 1), minval=0, maxval=2, dtype=TYPE_NN_INT)

        J, P = Function.sigmoid_cross_entropy_log_loss(X, T)
        EJ, EP = sigmoid_cross_entropy_log_loss(X=X.numpy(), T=T.numpy())

        # ********************************************************************************
        # Constraint: J and EJ are close
        # ********************************************************************************
        try:
            tf.debugging.assert_near(
                J,
                EJ,
                atol=tf.constant(1e-5),
                message="Expected J \n%s\nActual\n%s\nDiff %s" % (EJ, J, (EJ-J))
            )
        except tf.errors.InvalidArgumentError as e:
            raise AssertionError(str(e))

        # ********************************************************************************
        # Constraint: P and EP are close
        # ********************************************************************************
        try:
            tf.debugging.assert_near(
                P,
                EP,
                atol=tf.constant(1e-5),
                message="Expected P \n%s\nActual\n%s\nDiff %s" % (EP, P, (EP-P))
            )
        except tf.errors.InvalidArgumentError as e:
            raise AssertionError(str(e))
