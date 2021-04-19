"""Objective
Test cases for function.nn.base module
"""
import os
import logging
import numpy as np
import tensorflow as tf
from common.constant import (
    TYPE_FLOAT,
    TYPE_LABEL,
    OFFSET_LOG,
)
import function.common.base as base

from test.config import (
    NUM_MAX_TEST_TIMES,
    NUM_MAX_NODES,
    NUM_MAX_BATCH_SIZE,
    MAX_ACTIVATION_VALUE,
    ACTIVATION_DIFF_ACCEPTANCE_VALUE,
    LOSS_DIFF_ACCEPTANCE_VALUE,
    LOSS_DIFF_ACCEPTANCE_RATIO
)


Logger = logging.getLogger(__name__)
Logger.setLevel(logging.DEBUG)


def test_010_base_nn_is_float_scalar(caplog):
    """
    Objective:
        Verify the is_float_scalar() function.

    Constraints:
    1. True for numpy instance of np.float32, np.float64, np.float/python float
    2. True for numpy ndarray of shape () of type np.float32, np.float64, np.float/python float

    """
    # https://stackoverflow.com/a/66071396/4281353
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # ********************************************************************************
    # Constraint:
    # True for numpy instance of np.float32, np.float64, np.float/python float
    # ********************************************************************************
    np_float_primitive_scalar = np.float32(np.random.randn())
    msg = f"{np_float_primitive_scalar} should be true"
    assert base.Function.is_float_scalar(np_float_primitive_scalar), msg

    tf_float_scalar = tf.constant(np_float_primitive_scalar)
    msg = f"{tf_float_scalar} should be true"
    assert base.Function.is_float_scalar(tf_float_scalar), msg

    # ********************************************************************************
    # Constraint:
    # True for numpy ndarray of shape () of type np.float32, np.float64, np.float/python float
    # ********************************************************************************
    np_float_ndarray_scalar = np.array(np.float32(np.random.randn()))
    msg = f"{np_float_ndarray_scalar} should be true"
    assert base.Function.is_float_scalar(np_float_ndarray_scalar), msg

    tf_float_tensor_scalar = tf.constant(np_float_ndarray_scalar)
    msg = f"{tf_float_tensor_scalar} should be true"
    assert base.Function.is_float_scalar(tf_float_tensor_scalar), msg
    assert base.Function.is_float_scalar(tf.random.normal(shape=())), msg
