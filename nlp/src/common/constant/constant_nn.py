from typing import (
    TypeVar
)
import numpy as np
import tensorflow as tf
# --------------------------------------------------------------------------------
# Float number type to use. For NN, 32 bit is enough.
# There is no way to change the default float type
# https://github.com/numpy/numpy/issues/6860
# --------------------------------------------------------------------------------
TYPE_INT = np.int32
TYPE_LABEL = np.int32
TYPE_FLOAT = np.float64       # alias of Python float
assert not isinstance(TYPE_FLOAT, (float, np.float))
# TYPE_FLOAT = np.float64   # Cannot use due to Numpy default is float.
TYPE_TENSOR = TypeVar('TYPE_TENSOR', np.ndarray, tf.Tensor)

TYPE_NN_INT = tf.int32
TYPE_NN_FLOAT = tf.float32

# --------------------------------------------------------------------------------
# Be mindful of the relation between h/OFFSET_DELTA and k/OFFSET_LOG
# --------------------------------------------------------------------------------
# Numerical gradient (f(x+h)-f(x-h))/2h may cause an invalid value for f.
# e.g log loss tries to avoid log(0)->np.inf by a small value k as in log(0+k).
# However if k < h, f(x-h) causes nan due to log(x < 0) as x needs to be > 0.
#
# The values depend on the floating storage size. 1e-10 for the delta
# to calculate the numerical gradient is too small for 32 bit.
# --------------------------------------------------------------------------------
# OFFSET_DELTA
# OFFSET_LOG:
#   To avoid log(0) -> inf by log(0+offset). Be as small as possible to minimize
#   the impact by the clipping value log(OFFSET_LOG), but large enough so that the
#   OFFSET_DELTA will not cause rounding errors due to too small delta for division.
#   because we need OFFSET_LOG > OFFSET_DELTA to avoid the numerical gradient from
#   causing np.nan by f(log(x + OFFSET_LOG - OFFSET_DELTA)).
# --------------------------------------------------------------------------------
OFFSET_DELTA = TYPE_FLOAT(1e-10) if TYPE_FLOAT == np.float64 else TYPE_FLOAT(1e-5)
OFFSET_LOG = OFFSET_DELTA * TYPE_FLOAT(10.0)
assert OFFSET_LOG > OFFSET_DELTA
# Avoid div by zero at (X-u) / sqrt(variance + eps)
OFFSET_STD = TYPE_FLOAT(1e-10) if TYPE_FLOAT == np.float64 else TYPE_FLOAT(1e-5)

# When True, set the element to the offset value only when it is below the offset,
# clipping element values at the offset.
# Otherwise add offset to all elements in a blanket manner shifting all values
# by the offset.
OFFSET_MODE_ELEMENT_WISE = True

# --------------------------------------------------------------------------------
# BOUNDARY_SIGMOID = -np.log(OFFSET_LOG) * safe_margin_ratio
# Because of the impact by adding k/OFFSET_LOG, the logistic log loss L=-T*log(Z+k)
# is ceiled by -np.log(OFFSET_LOG) where Z=sigmoid(X). As Z gets closer to k, that-
# is, X gets **similar** to log(k), dL/dX starts getting flatten eventually gets 0.
# Before it starts, need to stop X from getting closer to log(k).
# Hence X < -np.log(OFFSET_LOG) * safety_margin_ratio.
# --------------------------------------------------------------------------------
BOUNDARY_SIGMOID = -np.log(OFFSET_LOG) * TYPE_FLOAT(0.5)

# Threshold below which the gradient is regarded as saturated.
GRADIENT_SATURATION_THRESHOLD = TYPE_FLOAT(1e-10) if TYPE_FLOAT == np.float64 else TYPE_FLOAT(1e-5)

# Min difference between f(x+h) and f(x-h) at numerical gradient to avoid
# floating precision error. If f(x+h) - f(x-h) is small
GN_DIFF_ACCEPTANCE_VALUE = TYPE_FLOAT(2) * OFFSET_DELTA * GRADIENT_SATURATION_THRESHOLD
GN_DIFF_ACCEPTANCE_RATIO = TYPE_FLOAT(1e-15)

# --------------------------------------------------------------------------------
# Numpy optimization
# conda install numexpr
# conda install numba
#
# NOTE: Acceleration depends on conditions e.g. the size of the matrices
# https://stackoverflow.com/questions/59347796/
# For larger N x N matrices (aprox. size 20) a BLAS  is faster than Numba/Cython
# --------------------------------------------------------------------------------
ENABLE_NUMEXPR = False
ENABLE_NUMBA = False

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
LAYER_MAX_NUM_NODES: int = 1000
