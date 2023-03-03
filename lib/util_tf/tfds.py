"""TensorFlow Datasets utility module
"""
from typing import (
    Tuple,
)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.debugging import (
    Assert,
    assert_non_negative,
)

from util_constant import (
    TYPE_FLOAT
)


def convert_pascal_voc_bndbox_to_yolo_bbox(
        bndbox: tf.Tensor
) -> tf.data.Dataset:
    """
    Convert TFDS PASCAL VOC XML bndbox annotation to YOLO Darknet Bounding box annotation
    (x,y,w,h) where (x,y) is the center of bbox and (w,h) is relative to image size.

    NOTE:
        TFDS VOC bndbox is tfds.features.BBox whose format is (ymin, xmin, ymax, xmax)
        and normalized with image size.

        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/BBox
        tfds.features.BBox(
            ymin, xmin, ymax, xmax
        )
    ---
      bndbox = obj.find("bndbox")
      xmax = float(bndbox.find("xmax").text)
      xmin = float(bndbox.find("xmin").text)
      ymax = float(bndbox.find("ymax").text)
      ymin = float(bndbox.find("ymin").text)
      yield {
          "label": label,
          "pose": pose,
          "bbox": tfds.features.BBox(
              ymin / height, xmin / width, ymax / height, xmax / width
          ),
          "is_truncated": is_truncated,
          "is_difficult": is_difficult,
      }
      ---

    [YOLO Darknet Annotation]
    YOLO Darknet annotations are stored in text files. Similar to VOC XML, there is
    one annotation per image. Unlike the VOC format, a YOLO annotation has only a
    text file defining each object in an image, one per plain text file line.
    id: Pascal VOC class ID
    (x,y): center coordinate of the bounding box relative to image size.
    (w,h): (width, height) fo the bounding box relative to image size.
    ----
    # id,x,y,w,h
    0 0.534375 0.4555555555555555 0.3854166666666667 0.6166666666666666
    ----
    [TFDF VOC]
    https://www.tensorflow.org/datasets/catalog/voc

    Args:
        bndbox: coordinate of box as (ymin, xmin, ymax, xmax)
    Returns:
        YOLO Darknet bbox annotation (x,y,w,h) as Dataset
    """
    x = (bndbox[1] + bndbox[3]) / TYPE_FLOAT(2.0)   # (xmin+xmax)/2
    y = (bndbox[0] + bndbox[2]) / TYPE_FLOAT(2.0)   # (ymin+ymax)/2
    w = bndbox[3] - bndbox[1]                       # xmax-xmin
    h = bndbox[2] - bndbox[0]                       # ymax-ymin

    box: tf.Tensor = tf.stack(values=[x, y, w, h], axis=-1)
    assert_non_negative(x=box, message="expected all non negative")

    return tf.data.Dataset.from_tensors(box)


def test_convert_pascal_voc_bndbox_to_yolo_bbox():
    xmin: TYPE_FLOAT = TYPE_FLOAT(0)
    xmax: TYPE_FLOAT = TYPE_FLOAT(6)
    ymin: TYPE_FLOAT = TYPE_FLOAT(0)
    ymax: TYPE_FLOAT = TYPE_FLOAT(8)
    bndbox: tfds.features.BBox = tfds.features.BBox(
        ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax
    )

    yolo_bbox_dataset = convert_pascal_voc_bndbox_to_yolo_bbox(bndbox=bndbox)
    yolo_bbox_tensor = yolo_bbox_dataset.take(1).get_single_element()
    x_centre = TYPE_FLOAT(yolo_bbox_tensor[0])
    y_centre = TYPE_FLOAT(yolo_bbox_tensor[1])
    w = TYPE_FLOAT(yolo_bbox_tensor[2])
    h = TYPE_FLOAT(yolo_bbox_tensor[3])

    assert x_centre == TYPE_FLOAT(3)
    assert y_centre == TYPE_FLOAT(4)
    assert w == TYPE_FLOAT(6)
    assert h == TYPE_FLOAT(8)