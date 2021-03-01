import numpy as np

# --------------------------------------------------------------------------------
# Be mindful of the relation between h/OFFSET_DELTA and k/OFFSET_LOG
# --------------------------------------------------------------------------------
# Numerical gradient (f(x+h)-f(x-h))/2h may cause an invalid value for f.
# e.g log loss tries to avoid log(0)->np.inf by a small value k as in log(0+k).
# However if k < h, f(x-h) causes nan due to log(x < 0) as x needs to be > 0.
# --------------------------------------------------------------------------------
# OFFSET_DELTA = 1e-10
# OFFSET_LOG = (OFFSET_DELTA + 1e-7)  # Avoid log(0) -> inf by log(0+offset)
# --------------------------------------------------------------------------------
OFFSET_DELTA = 1e-9
OFFSET_LOG = 1e-7       # Avoid log(0) -> inf by log(x) where x > offset
OFFSET_STD = 1e-10      # Avoid div by zero at (X-u) / sqrt(variance + eps)

# --------------------------------------------------------------------------------
# BOUNDARY_SIGMOID = -np.log(OFFSET_LOG) * safe_margin_ratio
# Because of the impact by adding k/OFFSET_LOG, the logistic log loss L=-T*log(Z+k)
# is ceiled by -np.log(OFFSET_LOG) where Z=sigmoid(X). As Z gets closer to k, that-
# is, X gets **similar** to log(k), dL/dX starts getting flatten eventually gets 0.
# Before it starts, need to stop X from getting closer to log(k).
# Hence X < -np.log(OFFSET_LOG) * safety_margin_ratio.
# --------------------------------------------------------------------------------
BOUNDARY_SIGMOID = -np.log(OFFSET_LOG) * 0.5

# Threshold below which the gradient is regarded as saturated.
GRADIENT_SATURATION_THRESHOLD = (10 ** -5)
# TODO restore the original
# GRADIENT_SATURATION_THRESHOLD = 0

# Min difference between f(x+h) and f(x-h) at numerical gradient to avoid
# floating precision error. If f(x+h) - f(x-h) is small
GN_DIFF_ACCEPTANCE_VALUE = 2 * OFFSET_DELTA * GRADIENT_SATURATION_THRESHOLD
# TODO restore the original
# GN_DIFF_ACCEPTANCE_RATIO = 1e-10
GN_DIFF_ACCEPTANCE_RATIO = 0.0

# To enforce assertion failure, set False (True -> assert True)
ENFORCE_STRICT_ASSERT = (not False)
