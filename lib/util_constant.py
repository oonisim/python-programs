import numpy as np


TYPE_INT = np.int32
TYPE_LABEL = np.int32
TYPE_FLOAT = np.float32     # alias of Python float
# assert not isinstance(TYPE_FLOAT, (float, np.float))
assert not isinstance(TYPE_FLOAT, float)

PI = TYPE_FLOAT(np.pi)
