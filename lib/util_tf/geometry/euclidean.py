"""
Euclidean geometry utility module
"""
from typing import (
    Union
)
import numpy as np
import tensorflow as tf
from tensorflow.debugging import (
    Assert,
    assert_equal,
    assert_non_negative,
    assert_less,
)

from util_constant import (
    TYPE_FLOAT
)


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def convert_box_corner_coordinates_to_centre_w_h(
        ymin: Union[np.ndarray, tf.Tensor],
        xmin: Union[np.ndarray, tf.Tensor],
        ymax: Union[np.ndarray, tf.Tensor],
        xmax: Union[np.ndarray, tf.Tensor]
) -> tf.Tensor:
    """
    Convert box annotation of (ymin,xmin,ymax,xmax) of corner coordinates to
    (x_centre, y_centre, width, height)

    Args:
        ymin: left bottom corner y coordinate
        xmin: left bottom corner x coordinate
        ymax: right top corner y coordinate
        xmax: right top corner x coordinate
    Returns: tf.Tensor([x_centre, y_centre, width, height])
    """
    x = (xmin + xmax) / TYPE_FLOAT(2.0)
    y = (ymin + ymax) / TYPE_FLOAT(2.0)
    w = xmax - xmin
    h = ymax - ymin

    result: tf.Tensor = tf.stack(values=[x, y, w, h], axis=-1)
    assert_non_negative(x=result, message="expected all non negative")
    return result


