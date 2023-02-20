import tensorflow as tf
from constant import (
    TYPE_FLOAT,
    TYPE_INT,
    YOLO_V1_IMAGE_WIDTH,
    YOLO_V1_IMAGE_HEIGHT,
)


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
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
        tensor: Intersection over union for all examples
    """
    assert boxes_preds.shape[1] == 4    # (x/0, y/1, w/2, h/3)
    assert boxes_labels.shape[1] == 4    # (x/0, y/1, w/2, h/3)
    assert boxes_preds.shape[0] == boxes_labels.shape[0]
    N: TYPE_INT = boxes_preds.shape[0]

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

    x1 = tf.math.maximum(box1_x1, box2_x1)
    y1 = tf.math.maximum(box1_y1, box2_y1)
    x2 = tf.math.maximum(box1_x2, box2_x2)
    y2 = tf.math.maximum(box1_y2, box2_y2)
    assert x1.shape == (N, 1)

    # --------------------------------------------------------------------------------
    # Intersection area
    # - YOLO v1 predicts bbox w/h as relative to image w/h, e.g. 0.6 * image w/h.
    # - Clip in case no intersection
    # --------------------------------------------------------------------------------
    width = tf.clip_by_value((x2 - x1), clip_value_min=TYPE_FLOAT(0), clip_value_max=TYPE_FLOAT(YOLO_V1_IMAGE_WIDTH))
    height = tf.clip_by_value((y2 - y1), clip_value_min=TYPE_FLOAT(0), clip_value_max=TYPE_FLOAT(YOLO_V1_IMAGE_HEIGHT))
    intersection = width * height

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # IOU between (0, 1)
    return tf.clip_by_value(
        # 1e-6 to avoid div by 0.
        intersection / (box1_area + box2_area - intersection + 1e-6),
        clip_value_min=TYPE_FLOAT(0.0),
        clip_value_max=TYPE_FLOAT(1.0)
    )
