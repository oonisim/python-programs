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
# OFFSET_SIGMOID = np.log(OFFSET_LOG)

OFFSET_DELTA = 1e-10
OFFSET_LOG = 1e-8      # Avoid log(0) -> inf by log(x) where x > offset
OFFSET_SIGMOID = -np.log(OFFSET_LOG)