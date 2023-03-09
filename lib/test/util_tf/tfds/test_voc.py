"""
TFDS voc test module
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
import tensorflow_datasets as tfds

from util_constant import (
    TYPE_FLOAT
)
from util_tf.geometry.euclidean import (
    convert_box_corner_coordinates_to_centre_w_h
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
from util_tf.tfds.voc import (
    convert_pascal_voc_bndbox_to_yolo_bbox,
    generate_yolo_v1_data_from_pascal_voc
)


# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
PASCAL_VOC_SAMPLE_DATASET_DIR: str = "./pascal_voc_samples"


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


# --------------------------------------------------------------------------------
# Testing YOLO label (C,P) where P is (cp,x,y,w,h)
# --------------------------------------------------------------------------------
def test_generate_yolo_v1_data_from_pascal_voc():
    """
    Objective:
        Verify generate_yolo_v1_data_from_pascal_voc generates expected YOLO v1 label

    Test Conditions:
        1. PASCAL VOC bndbox (xmin, xmax) is converted into x and w where x is the
           centre coordinate of bounding box and w is the width.
        2. PASCAL VOC bndbox (ymin, ymax) is converted into y and h where y is the
           centre coordinate of bounding box and h is the height.
        3. cp in YOLO label in P=(cp,x,y,w,h) is 1.0.
        4. C[voc_label_index-1] = 1.0 where C is YOLO class predictions of size 20.
    """
    # --------------------------------------------------------------------------------
    # Test Pascal VOC dataset saved in the disk
    # Dataset from TFDS PASCAL VOC have been saved to PASCAL_VOC_SAMPLE_DATASET_DIR
    # using tf.data.Dataset.save() method.
    # --------------------------------------------------------------------------------
    voc_dataset: tf.data.Dataset = \
        tf.data.Dataset.load(path=PASCAL_VOC_SAMPLE_DATASET_DIR, compression="GZIP")
    voc_record_generator = voc_dataset.as_numpy_iterator()

    # --------------------------------------------------------------------------------
    # Generate YOLO Dataset(inputs, labels) from PASCAL VOC Dataset
    # --------------------------------------------------------------------------------
    yolo_dataset: tf.data.Dataset = \
        voc_dataset.flat_map(generate_yolo_v1_data_from_pascal_voc)
    yolo_data_generator = yolo_dataset.as_numpy_iterator()

    # --------------------------------------------------------------------------------
    # Loop through the PASCAL VOC Dataset records.
    # Each record in TFDS VOC may have multiple bounding boxes and labels as there may
    # be multiple objects identified in an image.
    # --------------------------------------------------------------------------------
    for voc_record in voc_record_generator:
        # --------------------------------------------------------------------------------
        # PASCAL VOC Labels. Note that the label value is from 1 to 20.
        # --------------------------------------------------------------------------------
        labels: np.ndarray = voc_record['objects']['label']

        # --------------------------------------------------------------------------------
        # Bounding Box
        # --------------------------------------------------------------------------------
        for position, bndbox in enumerate(voc_record['objects']['bbox']):
            # TO BE bounding box
            # Tensor([x,y,w,h])
            expected_yolo_box: tf.Tensor = convert_box_corner_coordinates_to_centre_w_h(
                ymin=bndbox[0],
                xmin=bndbox[1],
                ymax=bndbox[2],
                xmax=bndbox[3]
            )
            # YOLO data (input=image, label=(C,P))
            yolo_record: dict = next(yolo_data_generator)
            actual_yolo_label = yolo_record[1]

            # Test condition #1
            actual_x = actual_yolo_label[YOLO_V1_LABEL_INDEX_X]
            actual_w = actual_yolo_label[YOLO_V1_LABEL_INDEX_W]
            assert_equal(x=expected_yolo_box[0], y=actual_x, message="expected same x")
            assert_equal(x=expected_yolo_box[2], y=actual_w, message="expected same w")

            # Test condition #2
            actual_y = actual_yolo_label[YOLO_V1_LABEL_INDEX_Y]
            actual_h = actual_yolo_label[YOLO_V1_LABEL_INDEX_H]
            assert_equal(x=expected_yolo_box[1], y=actual_y, message="expected same y")
            assert_equal(x=expected_yolo_box[3], y=actual_h, message="expected same h")

            # Test condition #3
            actual_cp = actual_yolo_label[YOLO_V1_LABEL_INDEX_CP]
            assert_equal(x=actual_cp, y=1.0, message="expected cp==1.0")

            # Test condition #4
            # PASCAL VOC label = index to class (-1 as VOC label is from 1 to 20)
            index_to_class: int = labels[position] - 1
            assert_equal(
                x=actual_yolo_label[index_to_class],
                y=1.0,
                message=f"expected {index_to_class}-th class"
            )
