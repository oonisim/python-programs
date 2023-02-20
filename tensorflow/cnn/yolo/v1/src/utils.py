"""
YOLO v1 utility module
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py

TODO:
    clarify if it is OK to clip values without consideration of the back propagation?
"""
import json
import logging
from typing import (
    List,
    Dict,
    Tuple,
    Callable,
    Optional,
    Union,
    Iterable,
)

import tensorflow as tf
from constant import (
    TYPE_FLOAT,
    TYPE_INT,
    YOLO_V1_IMAGE_WIDTH,
    YOLO_V1_IMAGE_HEIGHT,
)


from util_logging import (
    get_logger,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def intersection_over_union(
        boxes_preds, boxes_labels, box_format="midpoint"
) -> tf.Tensor:
    """
    Calculates intersection over union

    [Original YOLO v1 paper]
    Each bounding box consists of 5 predictions: x, y, w, h, and confidence.
    The (x, y) is the center of the box relative to the bounds of the grid cell.
    The w and h are predicted relative to the whole image.
    confidence is the IOU between the predicted box and any ground truth box.

    See towardsdatascience.com/yolo-made-simple-interpreting-the-you-only-look-once-paper-55f72886ab73

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        IOU(Intersection over union) for all examples
    """
    _name: str = "intersection_over_union()"
    assert boxes_preds.shape[1] == 4    # (x/0, y/1, w/2, h/3)
    assert boxes_labels.shape[1] == 4    # (x/0, y/1, w/2, h/3)
    assert boxes_preds.shape[0] == boxes_labels.shape[0]
    N: TYPE_INT = boxes_preds.shape[0]      # pylint: disable=invalid-name

    _logger.debug(
        "%s: sample prediction (x, y, w, h) = %s",
        _name, (boxes_preds[0][0], boxes_preds[0][1], boxes_preds[0][2], boxes_preds[0][3])
    )

    if box_format == "midpoint":
        # predicted box left x coordinate
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        # predicted box right x coordinate
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        # predicted box bottom y coordinate
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        # predicted box top y coordinate
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = tf.math.maximum(box1_x1, box2_x1)      # pylint: disable=invalid-name
    y1 = tf.math.maximum(box1_y1, box2_y1)      # pylint: disable=invalid-name
    x2 = tf.math.maximum(box1_x2, box2_x2)      # pylint: disable=invalid-name
    y2 = tf.math.maximum(box1_y2, box2_y2)      # pylint: disable=invalid-name
    assert x1.shape == (N, 1)
    _logger.debug(
        "%s: sample intersection corner coordinates (x1, y1, x2, y2) = %s",
        _name, (x1[0], y1[0], x2[0], y2[0])
    )

    # --------------------------------------------------------------------------------
    # Intersection
    # - YOLO v1 predicts bbox w/h as relative to image w/h, e.g. 0.6 * image w/h.
    # --------------------------------------------------------------------------------
    width: tf.Tensor = x2 - x1
    height: tf.Tensor = y2 - y1
    assert tf.math.reduce_all(width <= YOLO_V1_IMAGE_WIDTH + 1e-6), \
        "expected (width <= YOLO_V1_IMAGE_WIDTH + 1e-6), " \
        f"got {width[(width > YOLO_V1_IMAGE_WIDTH + 1e-6)]}"
    assert tf.math.reduce_all(height <= YOLO_V1_IMAGE_HEIGHT + 1e-6), \
        "expected (height <= YOLO_V1_IMAGE_WIDTH + 1e-6), " \
        f"got {height[(height > YOLO_V1_IMAGE_HEIGHT + 1e-6)]}"

    # Clip in case no intersection
    width = tf.clip_by_value(
        (x2 - x1), clip_value_min=TYPE_FLOAT(0), clip_value_max=TYPE_FLOAT(YOLO_V1_IMAGE_WIDTH)
    )
    height = tf.clip_by_value(
        (y2 - y1), clip_value_min=TYPE_FLOAT(0), clip_value_max=TYPE_FLOAT(YOLO_V1_IMAGE_HEIGHT)
    )
    intersection: tf.Tensor = tf.math.multiply(width, height)

    # --------------------------------------------------------------------------------
    # Union
    # --------------------------------------------------------------------------------
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union: tf.Tensor = (box1_area + box2_area - intersection + 1e-6)

    # --------------------------------------------------------------------------------
    # IOU between (0, 1)
    # --------------------------------------------------------------------------------
    IOU: tf.Tensor = tf.clip_by_value(      # pylint: disable=invalid-name
        # 1e-6 to avoid div by 0.
        tf.math.divide(intersection, union),
        clip_value_min=TYPE_FLOAT(0.0),
        clip_value_max=TYPE_FLOAT(1.0)
    )
    _logger.debug("%s: sample IOU = %s", _name, IOU[0])
    return IOU
