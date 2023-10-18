"""Constant variable definition module
"""
import numpy as np


# --------------------------------------------------------------------------------
# Data Types
# Be specific with the storage size to avoid unexpected casting/rounding errors.
# --------------------------------------------------------------------------------
TYPE_INT = np.int32
TYPE_LABEL = np.int32
TYPE_FLOAT = np.float32     # alias of Python float
# assert not isinstance(TYPE_FLOAT, (float, np.float))
assert not isinstance(TYPE_FLOAT, float)

PI = TYPE_FLOAT(np.pi)
