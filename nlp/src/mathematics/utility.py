"""NumPy Utility"""
import numpy as np


def xslice(x, slices):
    """Extract multiple slices from an array-like and concatenate them.
    Args:
        x: array-like
        slices: slice or tuple of slice objects
    Return:
        Combined slices
    """
    if isinstance(slices, tuple):
        if isinstance(x, np.ndarray):
            return np.concatenate([x[_slice] for _slice in slices])
        else:
            return sum((x[s] if isinstance(s, slice) else [x[s]] for s in slices), [])
    elif isinstance(slices, slice):
        return x[slices]
    else:
        return [x[slices]]