NUM_MAX_TEST_TIMES = 200
NUM_MAX_BATCH_SIZE: int = 5+1
NUM_MAX_NODES: int = 20+1
NUM_MAX_FEATURES: int = 20
MAX_ACTIVATION_VALUE = 5.0  # max output value from an activation function

# Accept the difference between the numerical gradient GN and analytical gradient G
# when it is less than the ratio of G, e.g for ratio=0.05, abd(G-GN) < (0.05 * GN),
# or below the threshold value.
#
# Skip the check when the gradient is in the saturation: < GRADIENT_DIFF_CHECK_TRIGGER
# Do not make it small enough. For gradient 0.001, 0.0011 would be good enough.
GRADIENT_DIFF_CHECK_TRIGGER = 0.005
GRADIENT_DIFF_ACCEPTANCE_RATIO = 0.3
GRADIENT_DIFF_ACCEPTANCE_VALUE = 1e-2
ACTIVATION_DIFF_ACCEPTANCE_VALUE = 1e-4
LOSS_DIFF_ACCEPTANCE_VALUE = 1e-4
LOSS_DIFF_ACCEPTANCE_RATIO = 0.001

# Accept the Numerical difference between re-formulated functions, e.g.
# log( exp(xi) / sum(exp(X)) ) = sum(exp(X)) - xi,
REFORMULA_DIFF_ACCEPTANCE_VALUE = 1e-10

# To enforce assertion failure, set False (True -> assert True)
ENFORCE_STRICT_ASSERT = (not False)
