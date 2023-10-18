"""Euclidean test module"""
import tensorflow as tf
from tensorflow.debugging import (
    assert_equal,
    assert_non_negative,
    assert_less,
)
from tensorflow.errors import (
    InvalidArgumentError
)

from util_constant import (
    TYPE_FLOAT
)
from util_tf.geometry.euclidean import (
    convert_box_corner_coordinates_to_centre_w_h
)


def test_convert_box_corner_coordinates_to_centre_w_h():
    """
    Objective:
        Verify the function returns (x,y,w,h) as Tensor where (x,y) is the centre
        of the box and (w,h) are the width, height

    Test Conditions:
        1. (x,y,w,h)==(0,0,0,0) for (ymin=0,xmin=0,ymax=0,xmax=0)
        2. (x,y,w,h)==(1,1,2,2) for (ymin=0,xmin=0,ymax=2,xmax=2)
    """
    # Test condition #1
    box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
        ymin=TYPE_FLOAT(0),
        xmin=TYPE_FLOAT(0),
        ymax=TYPE_FLOAT(0),
        xmax=TYPE_FLOAT(0)
    )
    assert_equal(x=box, y=tf.constant([0, 0, 0, 0], dtype=TYPE_FLOAT))

    # Test condition #2
    box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
        ymin=TYPE_FLOAT(0),
        xmin=TYPE_FLOAT(0),
        ymax=TYPE_FLOAT(2),
        xmax=TYPE_FLOAT(2)
    )
    assert_equal(x=box, y=tf.constant([1, 1, 2, 2], dtype=TYPE_FLOAT))


def test_convert_box_corner_coordinates_to_centre_w_h_fail():
    """
    Objective:
        Verify the function returns (x,y,w,h) as Tensor where (x,y) is the centre
        of the box and (w,h) are the width, height

    Test Conditions:
        1. x >= 0 as centre of bbox is inside a cell where the top left coordinate is (0,0)
        2. y >= 0 as centre of bbox is inside a cell where the top left coordinate is (0,0)
        3. w >=0. w can be 0 if there is no object in the cell.
        3. h >=0. h can be 0 if there is no object in the cell.
    """
    try:
        # Test condition #1
        box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
            ymin=TYPE_FLOAT(0),
            xmin=TYPE_FLOAT(0),
            ymax=TYPE_FLOAT(1),
            xmax=TYPE_FLOAT(-1)
        )
        raise AssertionError("Should fail as x is negative")
    except InvalidArgumentError:
        pass

    try:
        # Test condition #2
        box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
            ymin=TYPE_FLOAT(0),
            xmin=TYPE_FLOAT(0),
            ymax=TYPE_FLOAT(-1),
            xmax=TYPE_FLOAT(1)
        )
        raise AssertionError("Should fail as y is negative")
    except InvalidArgumentError:
        pass

    try:
        # Test condition #3
        box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
            ymin=TYPE_FLOAT(0),
            xmin=TYPE_FLOAT(1),
            ymax=TYPE_FLOAT(1),
            xmax=TYPE_FLOAT(0)
        )
        raise AssertionError("Should fail as w is negative")
    except InvalidArgumentError:
        pass

    try:
        # Test condition #4
        box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
            ymin=TYPE_FLOAT(1),
            xmin=TYPE_FLOAT(0),
            ymax=TYPE_FLOAT(0),
            xmax=TYPE_FLOAT(1)
        )
        raise AssertionError("Should fail as h is negative")
    except InvalidArgumentError:
        pass

