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
OFFSET_LOG = 1e-7      # Avoid log(0) -> inf by log(x) where x > offset

# --------------------------------------------------------------------------------
# BOUNDARY_SIGMOID = -np.log(OFFSET_LOG) * safe_margin_ratio
# Because of the impact by adding k/OFFSET_LOG, the logistic log loss L=-T*log(Z+k)
# is ceiled by -np.log(OFFSET_LOG) where Z=sigmoid(X). As Z gets closer to k, that-
# is, X gets **similar** to log(k), dL/dX starts getting flatten eventually gets 0.
# Before it starts, need to stop X from getting closer to log(k).
# Hence X < -np.log(OFFSET_LOG) * safety_margin_ratio.
# --------------------------------------------------------------------------------
BOUNDARY_SIGMOID = -np.log(OFFSET_LOG) * 0.5

# Min difference between f(x+h) and f(x-h) at numerical gradient to avoid
# floating precision error. If f(x+h) - f(x-h) is small
MIN_DIFF_AT_GN = np.power(OFFSET_DELTA, 2)
GN_DIFF_ACCEPTANCE_VALUE = OFFSET_DELTA / (10 ** 4)     # 1/10000
GN_DIFF_ACCEPTANCE_RATIO = 1e-12
assert np.isfinite(MIN_DIFF_AT_GN)
