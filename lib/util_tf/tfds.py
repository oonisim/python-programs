"""TensorFlow Datasets utility module
"""
from typing import (
    Tuple,
    List,
)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.debugging import (
    Assert,
    assert_equal,
    assert_non_negative,
    assert_less,
)

from util_constant import (
    TYPE_FLOAT,
)
from util_tf.yolo.v1.constant import (
    YOLO_V1_IMAGE_WIDTH,
    YOLO_V1_IMAGE_HEIGHT,
    YOLO_V1_LABEL_LENGTH,
    YOLO_V1_LABEL_INDEX_CP,
    YOLO_V1_LABEL_INDEX_X,
    YOLO_V1_LABEL_INDEX_Y,
    YOLO_V1_LABEL_INDEX_W,
    YOLO_V1_LABEL_INDEX_H,
    YOLO_V1_PREDICTION_NUM_CLASSES
)
from util_tf.geometry.euclidean import (
    convert_box_corner_coordinates_to_centre_w_h
)


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
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
    bndbox = tf.reshape(tensor=bndbox, shape=(-1, 4))
    xmin: tf.Tensor = bndbox[..., 1]
    xmax: tf.Tensor = bndbox[..., 3]
    ymin: tf.Tensor = bndbox[..., 0]
    ymax: tf.Tensor = bndbox[..., 2]
    box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
        ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax
    )
    return tf.data.Dataset.from_tensors(box)


def generate_yolo_v1_class_predictions(labels: tf.Tensor, dtype=tf.dtypes.float32):
    """
    Generate class prediction C in the YOLO v1 label of format (C,P) where C is 20 classes.
    It is one hot encoding to identify the object class id.

    TFDS Pascal VOC may have multiple object classes identified in single image, hence
    the shape of labels can be () or (n, 1) depending on the number of objects identified.

    Args:
        labels: Sparce indices to the identified classes in an image
        dtype: Output tensor dtype
    Returns: One Hot Encoding tensor of shape (n, 20)
    """
    # --------------------------------------------------------------------------------
    # Sparce label index of int64 (as in voc) to the class of truth (0, ..., 19).
    # The shape can be () or (n) based on the objects identified in the image.
    # --------------------------------------------------------------------------------
    labels: tf.Tensor = tf.reshape(
        tensor=labels,
        shape=(-1, 1)
    )
    # Number of objects / classes identified in the image
    num_objects = tf.shape(input=labels)[0]
    # tf.print("labels shape", tf.shape(labels), labels)

    # --------------------------------------------------------------------------------
    # Generate class prediction C in YOLO v1 label (C,P).
    # [Approach]
    # 1. Create a zeros tensor of shape (num_objects, YOLO_V1_PREDICTION_NUM_CLASSES).
    # 2. Set 1.0 to the (row, col) positions in the zeros tensor for each object.
    # Each row i corresponds with an object identified whose class label is labels[i].
    # Hence, the column j=labels[i] at the row is set to 1.
    # --------------------------------------------------------------------------------
    # (row=i, col=labels[i]) positions set to 1.0
    positions: tf.Tensor = tf.stack(
        values=[
            tf.reshape(tf.range(num_objects, dtype=labels.dtype), (-1, 1)),  # row i
            labels                                                           # column j=labels[i]
        ],
        axis=-1
    )
    classes = tf.tensor_scatter_nd_update(
        tensor=tf.zeros(shape=(num_objects, YOLO_V1_PREDICTION_NUM_CLASSES), dtype=dtype),
        indices=positions,
        updates=tf.ones(shape=(num_objects, 1))
    )
    return classes


def _generate_yolo_v1_label_from_pascal_voc(dataset) -> tf.Tensor:
    """
    Generate YOLO v1 label in format (C,P) where C is 20 classes and P is (cp,x,y,w,h)
    where (x,y) is the centre coordinate of a bounding box that locates an object and
    (w,h) is the (width,height) of the box. cp is confidence which is 1.0 for label.

    Args:
        dataset: VOC dataset (either train, validation, or test)
    Returns: tf.Tensor of shape (25,) that holds (C,P)
    """
    # --------------------------------------------------------------------------------
    # (x,y,w,h) in YOLO v1 label
    # From Pascal VOC bounding box to YOLO v1 bounding box
    # bndbox shape can be (4,) or (n, 4) based on the objects identified in the image.
    # --------------------------------------------------------------------------------
    bndbox = tf.reshape(tensor=dataset['objects']['bbox'], shape=(-1, 4))
    num_objects = tf.shape(bndbox)[0]
    # tf.print("number of objects in the image: ", num_objects)

    box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
        ymin=bndbox[..., 0],
        xmin=bndbox[..., 1],
        ymax=bndbox[..., 2],
        xmax=bndbox[..., 3]
    )
    x = box[..., 0:1]
    dtype = x.dtype
    # tf.print("box shape", tf.shape(box), box)

    # --------------------------------------------------------------------------------
    # Confidence score cp that has the same shape and type of x=box[..., 0:1].
    # cp of the ground truth bounding box is always 1.0
    # --------------------------------------------------------------------------------
    cp: tf.Tensor = tf.ones_like(
        input=x,
        dtype=dtype
    )

    # --------------------------------------------------------------------------------
    # Sparce label index of int64 (as in voc) to the class of truth (0, ..., 19).
    # The shape can be () or (n) based on the objects identified in the image.
    # --------------------------------------------------------------------------------
    indices = tf.reshape(
        tensor=dataset['objects']['label'],
        shape=(-1, 1)
    )
    tf.debugging.assert_equal(
        x=num_objects,
        y=tf.shape(indices)[0],
        message="expected same numbef of objects."
    )
    # tf.print("index shape", tf.shape(index), index)

    # --------------------------------------------------------------------------------
    # C=20 class predictions in YOLO v1 label
    # --------------------------------------------------------------------------------
    classes = generate_yolo_v1_class_predictions(labels=indices, dtype=dtype)

    # --------------------------------------------------------------------------------
    # YOLO v1 label
    # --------------------------------------------------------------------------------
    label = tf.concat([classes, cp, box], axis=-1)
    return label


def generate_yolo_v1_label_from_pascal_voc(dataset) -> tf.data.Dataset:
    label = _generate_yolo_v1_label_from_pascal_voc(dataset=dataset)
    return tf.data.Dataset.from_tensor_slices(label)


def _generate_yolo_v1_data_from_pascal_voc(dataset) -> tf.Tensor:
    """Generate dataset of (image, label) for YOLO v1 training or validation.
    TFDS Pascal VOC has the structure where there can be multiple objects identified
    in single image. To have one-to-one relation between image and label, repeats
    the image n times where n is the number of identified objects (labels).

    Args:
        dataset: VOC dataset (either train, validation, or test)
    Returns: Tensor of tuple(image, label)
    """
    labels = _generate_yolo_v1_label_from_pascal_voc(dataset=dataset)
    num_objects = tf.shape(labels)[0]
    # tf.print("labels shape ", tf.shape(labels))
    # tf.print("num_objects ", num_objects)

    resized = tf.image.resize(
        images=dataset['image'],
        size=(YOLO_V1_IMAGE_WIDTH, YOLO_V1_IMAGE_HEIGHT)
    )
    images = tf.repeat(
        input=(resized[tf.newaxis, ...]),
        repeats=num_objects,
        axis=0
    )

    # return images
    return tf.nest.map_structure(
        lambda left, right: (left, right), images, labels
    )


def generate_yolo_v1_data_from_pascal_voc(dataset) -> tf.data.Dataset:
    """Dataset version of _generate_yolo_v1_data_from_pascal_voc for flat_map"""
    data = _generate_yolo_v1_data_from_pascal_voc(dataset)
    return tf.data.Dataset.from_tensor_slices(data)


# --------------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------------
def test_convert_pascal_voc_bndbox_to_yolo_bbox():
    xmin: TYPE_FLOAT = tf.constant(0, dtype=TYPE_FLOAT)
    xmax: TYPE_FLOAT = tf.constant(6, dtype=TYPE_FLOAT)
    ymin: TYPE_FLOAT = tf.constant(0, dtype=TYPE_FLOAT)
    ymax: TYPE_FLOAT = tf.constant(8, dtype=TYPE_FLOAT)
    bndbox: tfds.features.BBox = tfds.features.BBox(
        ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax
    )

    yolo_bbox_dataset = convert_pascal_voc_bndbox_to_yolo_bbox(bndbox=bndbox)
    yolo_bbox_tensor = yolo_bbox_dataset.take(1).get_single_element()
    x_centre = TYPE_FLOAT(yolo_bbox_tensor[..., 0])
    y_centre = TYPE_FLOAT(yolo_bbox_tensor[..., 1])
    w = TYPE_FLOAT(yolo_bbox_tensor[..., 2])
    h = TYPE_FLOAT(yolo_bbox_tensor[..., 3])

    tf.debugging.assert_equal(
        x=x_centre,
        y=tf.ones_like(input=x_centre) * tf.constant(3.0), message="expected 3"
    )
    tf.debugging.assert_equal(
        x=y_centre,
        y=tf.ones_like(input=y_centre) * tf.constant(4.0), message="expected 4"
    )
    tf.debugging.assert_equal(
        x=w,
        y=tf.ones_like(input=w) * tf.constant(6.0), message="expected 6"
    )
    tf.debugging.assert_equal(
        x=h,
        y=tf.ones_like(input=w) * tf.constant(8.0), message="expected 8"
    )
