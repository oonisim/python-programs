NUM_MAX_TEST_TIMES = 100
NUM_MAX_BATCH_SIZE: int = 1+1
NUM_MAX_NODES: int = 2+1
NUM_MAX_FEATURES: int = 2
MAX_ACTIVATION_VALUE = 5.0  # max output value from an activation function

# Accept the difference between the numerical gradient GN and analytical gradient G
# when it is less than the ratio of G, e.g for ratio=0.05, abd(G-GN) < (0.05 * GN)
# is acceptable.
GRADIENT_DIFF_ACCEPTANCE_RATIO = 0.15
GRADIENT_DIFF_ACCEPTANCE_VALUE = 1e-5

# Accept the Numerical difference between re-formulated functions, e.g.
# log( exp(xi) / sum(exp(X)) ) = sum(exp(X)) - xi,
REFORMULA_DIFF_ACCEPTANCE_VALUE = 1e-10
