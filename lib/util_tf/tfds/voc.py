"""TensorFlow Datasets Pascal VOC utility module
https://www.tensorflow.org/datasets/catalog/voc
"""
import sys
sys.path.append("/Users/oonisim/home/repository/git/oonisim/python-programs/lib")

from typing import (
    Tuple,
    List,
    Dict,
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
    YOLO_GRID_SIZE,
    YOLO_V1_IMAGE_WIDTH,
    YOLO_V1_IMAGE_HEIGHT,
    YOLO_V1_LABEL_LENGTH,
    YOLO_V1_LABEL_INDEX_CP,
    YOLO_V1_LABEL_INDEX_X,
    YOLO_V1_LABEL_INDEX_Y,
    YOLO_V1_LABEL_INDEX_W,
    YOLO_V1_LABEL_INDEX_H,
    YOLO_V1_PREDICTION_NUM_PRED,
    YOLO_V1_PREDICTION_NUM_CLASSES
)
from util_tf.geometry.euclidean import (
    convert_box_corner_coordinates_to_centre_w_h
)


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
# Class labels of TFDS VOC Dataset. Other origins may use different.
PASCAL_VOC_CLASSES: List[str] = [
    "aeroplane",        # 0
    "bicycle",          # 1
    "bird",             # 2
    "boat",             # 3
    "bottle",           # 4
    "bus",              # 5
    "car",              # 6
    "cat",              # 7
    "chair",            # 8
    "cow",              # 9
    "diningtable",      # 10
    "dog",              # 11
    "horse",            # 12
    "motorbike",        # 13
    "person",           # 14
    "pottedplant",      # 15
    "sheep",            # 16
    "sofa",             # 17
    "train",            # 18
    "tvmonitor"         # 19
]
S = YOLO_GRID_SIZE
C = YOLO_V1_PREDICTION_NUM_CLASSES
P = YOLO_V1_PREDICTION_NUM_PRED


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def _convert_pascal_voc_bndbox_to_yolo_bbox(
        bndbox: tf.Tensor
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
    bndbox = tf.reshape(tensor=bndbox, shape=(-1, 4))
    xmin: tf.Tensor = bndbox[..., 1]
    xmax: tf.Tensor = bndbox[..., 3]
    ymin: tf.Tensor = bndbox[..., 0]
    ymax: tf.Tensor = bndbox[..., 2]
    box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
        ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax
    )
    return box


def convert_pascal_voc_bndbox_to_yolo_bbox(
        bndbox: tf.Tensor
) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensors(
        _convert_pascal_voc_bndbox_to_yolo_bbox(bndbox=bndbox)
    )


def generate_yolo_v1_class_predictions(labels: tf.Tensor):
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
    # The shape should be (n, 1) as there can be multiple objects identified in image.
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
    # DO NOT use tf.stack here which generate (n, 1, 2) instead of (n, 2)
    positions: tf.Tensor = tf.concat(
        values=[
            tf.reshape(tf.range(num_objects, dtype=labels.dtype), (-1, 1)),  # row i
            labels                                                           # column j=labels[i]
        ],
        axis=-1
    )
    classes = tf.tensor_scatter_nd_update(
        tensor=tf.zeros(shape=(num_objects, C), dtype=TYPE_FLOAT),
        indices=positions,
        updates=tf.ones(shape=(num_objects,))
    )
    return classes


def generate_yolo_v1_labels_from_pascal_voc(record: Dict) -> tf.Tensor:
    """
    Generate YOLO v1 labels in shape (S,S,(C+P)) per image where C is 20 classes and
    P is (cp,x,y,w,h) where (x,y) is the centre coordinate of a bounding box that
    locates an object and (w,h) is the (width,height) of the box.
    cp is confidence which is 1.0 for label.

    Note:
        Run as tf.py_function which only take a list as 'imp' argument and cannot
        take dictionary.

        PASCAL VOC image shape is not that of YOLO v1 which is (448,448,3). Hence,
        need to resize. However, no need to change (x,y) in bounding boxes as they
        are normalized by the image size so that they are between 0. and 1.

    Args:
        bndbox: bounding boxes
        label: index to the class
    Returns: tf.Tensor of shape (S,S,C+P)
    """
    bndbox: tf.Tensor = record['objects']['bbox']
    label: tf.Tensor = record['objects']['label']
    tf.debugging.assert_non_negative(
        x=YOLO_V1_PREDICTION_NUM_CLASSES-label, message="label < 20"
    )

    # --------------------------------------------------------------------------------
    # PASCAL VOC bndbox shape should be (n, 4) as multiple objects can exist in an image
    # --------------------------------------------------------------------------------
    bndbox = tf.reshape(tensor=bndbox, shape=(-1, 4))
    num_objects = tf.shape(bndbox)[0]
    # tf.print("number of objects in the image: ", num_objects)

    # --------------------------------------------------------------------------------
    # From Pascal VOC bounding box to YOLO v1 bounding box (x,y,w,h)
    # --------------------------------------------------------------------------------
    box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
        ymin=bndbox[..., 0],
        xmin=bndbox[..., 1],
        ymax=bndbox[..., 2],
        xmax=bndbox[..., 3]
    )
    # tf.print("box shape", tf.shape(box), box)

    # --------------------------------------------------------------------------------
    # Confidence score cp that has the same shape and type of x=box[..., 0:1].
    # cp of the ground truth bounding box is always 1.0
    # --------------------------------------------------------------------------------
    cp: tf.Tensor = tf.ones_like(
        input=box[..., 0:1],
        dtype=box.dtype
    )

    # --------------------------------------------------------------------------------
    # Sparce label index to the class of truth (0, ..., 19). int64 as in voc.
    # The shape should (n,1) as there can be multiple objects identified.
    # --------------------------------------------------------------------------------
    indices = tf.reshape(
        tensor=label,
        shape=(-1, 1)
    )

    tf.debugging.assert_equal(
        x=num_objects,
        y=tf.shape(indices)[0],
        message="expected bbox and labels have the same number of objects."
    )
    # tf.print("index shape", tf.shape(index), index)

    # --------------------------------------------------------------------------------
    # C=20 class predictions in YOLO v1 label
    # --------------------------------------------------------------------------------
    classes = generate_yolo_v1_class_predictions(labels=indices)

    # --------------------------------------------------------------------------------
    # YOLO v1 labels for all the identified bounding boxes
    # --------------------------------------------------------------------------------
    labels = tf.concat([classes, cp, box], axis=-1)

    # --------------------------------------------------------------------------------
    # YOLO v1 labels for an image of S x S grid where each cell in the  has a label.
    # NOTE:
    #   The centres of multiple bounding boxes can fall into single cell. Then, which
    #   bounding box take precedence is non-deterministic.
    #
    #   See https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update.
    # > The order in which updates are applied is nondeterministic, so the output
    # > will be nondeterministic if indices contains duplicates.
    # --------------------------------------------------------------------------------
    grid = tf.zeros(shape=(S, S, C+P), dtype=TYPE_FLOAT)

    # Coordinate (row, col) of the grid that includes the centre of a bounding box (x,y).
    # row = (0,...,S-1), col = (0,...,S-1)
    def fn_xy_to_grid_coordinate(x_y):
        grid_row = tf.cast(tf.math.floor(S * x_y[1]), dtype=tf.int32)   # y
        grid_col = tf.cast(tf.math.floor(S * x_y[0]), dtype=tf.int32)   # x
        assert_non_negative(x=grid_row, message="grid_row >= 0")
        assert_non_negative(x=grid_col, message="grid_col >= 0")

        return tf.stack([grid_row, grid_col], axis=-1)

    xy: tf.Tensor = tf.stack([box[..., 0], box[..., 1]], axis=-1)
    coordinates = tf.map_fn(
        fn=fn_xy_to_grid_coordinate,
        elems=xy,                               # shape:(n,2)
        fn_output_signature=tf.TensorSpec(
            shape=(2,),                         # Output shape of fn
            dtype=tf.dtypes.int32,
            name=None
        )
    )
    # Update cells in the S x S grid that have the centre of the bounding boxes.
    return tf.tensor_scatter_nd_update(
        tensor=grid,
        indices=coordinates,
        updates=labels
    )


def generate_yolo_v1_data_from_pascal_voc(record: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate (image, labels) for YOLO v1 where label has (S,S,C+P) shape to tell the
    binding box and class for each grid, hence there are S*S labels.

    Args:
        record: Single row/record from the PASCAL VOC Dataset
    Returns: Tensor of tuple(image, labels)
    """
    image: tf.Tensor = record['image']
    bndbox: tf.Tensor = record['objects']['bbox']
    label: tf.Tensor = record['objects']['label']

    # --------------------------------------------------------------------------------
    # Resize to YOLO v1 image
    # --------------------------------------------------------------------------------
    resized = tf.image.resize(
        images=image,
        size=(YOLO_V1_IMAGE_WIDTH, YOLO_V1_IMAGE_HEIGHT)
    )
    labels = generate_yolo_v1_labels_from_pascal_voc(record)
    # labels = tf.py_function(
    #     # tf.py_function inp argument only support List. Cannot pass a dictionary.
    #     # https://github.com/tensorflow/tensorflow/issues/27679
    #     # > This is a limitation of py_function which only supports a list of Tensor
    #     # inputs  and I don't need see a straightforward way to extend its implementation
    #     # to support dictionaries. FWIW, you could deconstruct the dictionary structure
    #     # ahead of invoking the py_function:
    #     func=generate_yolo_v1_labels_from_pascal_voc,
    #     inp=[
    #         bndbox,
    #         label
    #     ],
    #     Tout=TYPE_FLOAT
    # )
    return resized, labels


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


def main():
    voc, info = tfds.load(
        name='voc',
        data_dir="/Volumes/SSD/data/tfds/",
        with_info=True,
    )
    labels = voc['train'].take(21).map(
        generate_yolo_v1_labels_from_pascal_voc,
        num_parallel_calls=1,
        deterministic=True
    )
    for index, label in enumerate(labels):
        print(index, label.shape)
        print('-' * 80)


if __name__ == "__main__":
    main()