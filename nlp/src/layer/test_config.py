NUM_MAX_TEST_TIMES = 100
NUM_MAX_BATCH_SIZE: int = 5+1
NUM_MAX_NODES: int = 10+1
NUM_MAX_FEATURES: int = 5
MAX_ACTIVATION_VALUE = 5.0  # max output value from an activation function

# Accept the difference between the numerical gradient GN and analytical gradient G
# when it is less than the ratio of G, e.g for ratio=0.05, abd(G-GN) < (0.05 * GN)
# is acceptable.
GRADIENT_DIFF_ACCEPTANCE_RATIO = 0.05

NUM_MAX_TEST_TIMES = 100
NUM_MAX_BATCH_SIZE: int = 5+1
NUM_MAX_NODES: int = 4+1
NUM_MAX_FEATURES: int = 5
MAX_ACTIVATION_VALUE = 5.0  # max output value from an activation function

# Accept the difference between the numerical gradient GN and analytical gradient G
# when it is less than the ratio of G, e.g for ratio=0.05, abd(G-GN) < (0.05 * GN)
# is acceptable.
GRADIENT_DIFF_ACCEPTANCE_RATIO = 0.05
