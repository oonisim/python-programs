"""TensorFlow Datasets utility module
"""
from typing import (
    Tuple,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.debugging import (
    Assert
)


def tfds_convert_pascal_voc_bndbox_to_yolo_bbox(
        box: tf.Tensor
) -> tf.Tensor:
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
        YOLO Darknet bbox annotation (x,y,w,h) as Tensor
    """
    box = tf.reshape(tensor=bndbox, shape=(-1,len("xywh")))

    x = (box[:, 1] + box[:, 3]) / 2.0  # (xmin+xmax)/2
    y = (box[:, 0] + box[:, 2]) / 2.0  # (ymin+ymax)/2
    w = box[:, 3] - box[:, 1]          # xmax-xmin
    h = box[:, 2] - box[:, 0]          # ymax-ymin
    bbox = tf.concat(values=[x, y, w, h], axis=-1)

    # Assert(condition=tf.reduce_all(bbox >= 0.0), data=bndbox)
    return bbox
