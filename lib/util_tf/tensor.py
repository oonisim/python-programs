"""TensorFlow Tensor utility module
"""
import numpy as np
import tensorflow as tf


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def take_rows_by_indices(
        X: tf.Tensor,
        M:int,
        D: int,
        indices: tf.Tensor
) -> tf.Tensor:
    """Extract rows from (N,R,C) tensor with row indices.
    You want to extract from X:
    1. row with index=1 from the first batch (N=0)
    2. row with index=0 from the second batch (N=1)

    X: [
        # N=0
        [
            [ 0,  1,  2,  3],
            [ 4,  5,  6,  7],    # <--- row index 1
            [ 8,  9, 10, 11]
        ],
        # N=1
        [
            [12, 13, 14, 15],    # <--- row index 0
            [16, 17, 18, 19],
            [20, 21, 22, 23]
        ]
    ]
    Then indices = [1, 0]

    Args:
        X: N number of ```(M, D)``` matrix e.g. (M=3, D=4)
        M: number of rows in the matrix
        D: number of columns in the matrix
        indices: index value (0, ... M-1) to select a row from each matrix.

    Returns: Extracted rows in shape (N, D)
    """
    X = tf.reshape(tensor=X, shape=(-1, M, D))
    indices = tf.reshape(tensor=indices, shape=(-1))
    assert indices.dtype in (tf.dtypes.uint8, tf.dtypes.int32, tf.dtypes.int64), \
        f"indices of dtype must be one of {(tf.dtypes.uint8, tf.dtypes.int32, tf.dtypes.int64)}"
    assert X.shape[0] == indices.shape[0], \
        f"expected same number of matrices in X:{X.shape[0]} and indices:{indices.shape[0]}."

    # MatMul with OHE extracts a row by masking other rows with 0 in OHE
    return tf.einsum("nmd,nm->nd", X, tf.one_hot(indices=indices, depth=M, dtype=X.dtype))
