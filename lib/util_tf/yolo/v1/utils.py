"""
YOLO v1 utility module based on YOLOv1 from Scratch by Aladdin Persson.
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/utils.py

[References]
https://datascience.stackexchange.com/q/118656/68313

TODO:
    clarify if it is OK to clip values without consideration of the back propagation?
"""
# pylint: disable=too-many-statements
import logging

import tensorflow as tf

from util_tf.yolo.v1.constant import (
    TYPE_FLOAT,
    EPSILON,
)
from util_logging import (
    get_logger,
)

# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
MAX_EXPECTED_W_PREDICTION: TYPE_FLOAT = TYPE_FLOAT(5.0)
MAX_EXPECTED_H_PREDICTION: TYPE_FLOAT = TYPE_FLOAT(5.0)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def intersection_over_union(
        boxes_preds: tf.Tensor,
        boxes_labels: tf.Tensor,
) -> tf.Tensor:
    """
    Calculates intersection over union

    [Original YOLO v1 paper]
    ```
    Each bounding box consists of 5 predictions: cp, x, y, w, h where cp is confidence.
    The (x, y) is the center of the box relative to the bounds of the grid cell.
    The w and h are predicted relative to the whole image.
    confidence is the IOU between the predicted box and any ground truth box.

    We normalize the bounding box width and height by the image width and height
    so that they fall between 0 and 1
    ```
    e.g. (x, y, w, h) = (0.3, 0.7, 0.6, 1.1). 1.1. because the bounding box to
    surround the object can outgrow the image itself.

    Check if w, h within MAX_EXPECTED_W_PREDICTION and MAX_EXPECTED_H_PREDICTION.

    See:
    https://youtu.be/n9_XyCGr-MI?list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&t=281
    towardsdatascience.com/yolo-made-simple-interpreting-the-you-only-look-once-paper-55f72886ab73

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (x, y, w, h) in shape:(N, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (x, y, w, h) in shape:(N, 4)
    Returns:
        IOU(Intersection over union) for all examples in shape (N, 1)
    """
    _name: str = "intersection_over_union()"
    # Cannot use python bool in TF Graph mode
    # assert tf.rank(boxes_preds) == 2, \
    #     f"expected dims=4 with shape (N, 4) got {tf.rank(boxes_preds)} dimensions."
    # assert tf.rank(boxes_labels) == 2, \
    #     f"expected dims=4 with shape (N, 4) got {tf.rank(boxes_labels)} dimensions."
    # assert tf.shape(boxes_preds) == tf.shape(boxes_labels), \
    #     f"expected same shape, got {tf.shape(boxes_preds)} and {tf.shape(boxes_labels)}."
    # assert boxes_preds.shape[1] == boxes_labels.shape[1] == 4   # (x/0, y/1, w/2, h/3)

    # Check if w, h within MAX_EXPECTED_W_PREDICTION and MAX_EXPECTED_H_PREDICTION
    w_predicted: tf.Tensor = boxes_preds[..., 2]
    h_predicted: tf.Tensor = boxes_preds[..., 3]

    # Cannot use python bool in TF Graph mode
    # assert tf.math.reduce_all(w_predicted <= MAX_EXPECTED_W_PREDICTION + EPSILON), \
    #     "expected w_predicted <= MAX_EXPECTED_W_PREDICTION, " \
    #     f"got\n{w_predicted[(w_predicted > MAX_EXPECTED_W_PREDICTION + EPSILON)]}"
    # assert tf.math.reduce_all(h_predicted <= MAX_EXPECTED_H_PREDICTION + EPSILON), \
    #     "expected height <= MAX_EXPECTED_H_PREDICTION, " \
    #     f"got\n{h_predicted[(h_predicted > MAX_EXPECTED_H_PREDICTION + EPSILON)]}"

    _logger.debug(
        "%s: sample prediction (x, y, w, h) = %s",
        _name,
        (
            boxes_preds[..., 0, 0],
            boxes_preds[..., 0, 1],
            boxes_preds[..., 0, 2],
            boxes_preds[..., 0, 3])
    )

    N: int = boxes_preds.shape[0]      # pylint: disable=invalid-name
    _logger.debug("%s:total cells [%s]", N)

    # --------------------------------------------------------------------------------
    # Corner coordinates of Bounding Boxes and Ground Truth
    # --------------------------------------------------------------------------------
    # predicted box left x coordinate
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2.
    # predicted box right x coordinate
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2.
    # predicted box bottom y coordinate
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2.
    # predicted box top y coordinate
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2.

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2.
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2.
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2.
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2.

    # --------------------------------------------------------------------------------
    # Intersection
    # - YOLO v1 predicts bbox w/h as relative to image w/h, e.g. 0.6 * image w/h.
    # --------------------------------------------------------------------------------
    x1 = tf.math.maximum(box1_x1, box2_x1)      # pylint: disable=invalid-name
    y1 = tf.math.maximum(box1_y1, box2_y1)      # pylint: disable=invalid-name
    x2 = tf.math.maximum(box1_x2, box2_x2)      # pylint: disable=invalid-name
    y2 = tf.math.maximum(box1_y2, box2_y2)      # pylint: disable=invalid-name
    # assert tf.shape(x1) == (N, 1)
    _logger.debug(
        "%s: sample intersection corner coordinates (x1, y1, x2, y2) = %s",
        _name, (x1[..., 0, 0], y1[..., 0, 0], x2[..., 0, 0], y2[..., 0, 0])
    )

    # Clip with 0 in case no intersection,
    width: tf.Tensor = x2 - x1
    height: tf.Tensor = y2 - y1
    width = tf.clip_by_value(
        width, clip_value_min=TYPE_FLOAT(0), clip_value_max=TYPE_FLOAT(5.0)
    )
    height = tf.clip_by_value(
        height, clip_value_min=TYPE_FLOAT(0), clip_value_max=TYPE_FLOAT(5.0)
    )
    intersection: tf.Tensor = tf.math.multiply(width, height)

    # --------------------------------------------------------------------------------
    # Union
    # --------------------------------------------------------------------------------
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union: tf.Tensor = (box1_area + box2_area - intersection + EPSILON)

    # --------------------------------------------------------------------------------
    # IOU between (0, 1)
    # --------------------------------------------------------------------------------
    IOU: tf.Tensor = tf.clip_by_value(      # pylint: disable=invalid-name
        # EPSILON to avoid div by 0.
        tf.math.divide(intersection, union),
        clip_value_min=TYPE_FLOAT(0.0),
        clip_value_max=TYPE_FLOAT(1.0)
    )
    _logger.debug("%s: sample IOU = %s", _name, IOU[0])
    # assert tf.shape(IOU) == (N, 1), f"expected IOU shape {(N, 1)}, got {IOU.shape}"
    # assert tf.math.reduce_all(IOU <= TYPE_FLOAT(1.0+EPSILON)), \
    #     f"expected IOU <= 1.0, got\n{IOU[(IOU > TYPE_FLOAT(1.0+EPSILON))]}"

    return IOU
