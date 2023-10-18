"""
Constant definitions
"""
# pylint: disable=invalid-name
import logging
import numpy as np
import tensorflow as tf

from util_constant import (
    TYPE_FLOAT,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
DEBUG_LEVEL: int = logging.INFO
DUMP: bool = False
logging.basicConfig(level=DEBUG_LEVEL)

# --------------------------------------------------------------------------------
# TYPES
# --------------------------------------------------------------------------------
ZERO: tf.Tensor = tf.constant(0, dtype=TYPE_FLOAT)
ONE: tf.Tensor = tf.constant(1, dtype=TYPE_FLOAT)
EPSILON = TYPE_FLOAT(1e-6)  # small enough value e.g. to avoid div by zero

YOLO_GRID_SIZE: int = 7

# --------------------------------------------------------------------------------
# YOLO v1 Input
# --------------------------------------------------------------------------------
YOLO_V1_IMAGE_WIDTH: int = 448
YOLO_V1_IMAGE_HEIGHT: int = 448
YOLO_V1_IMAGE_CHANNELS: int = 3

# --------------------------------------------------------------------------------
# YOLO Model
# --------------------------------------------------------------------------------
YOLO_V1_LEAKY_RELU_SLOPE: TYPE_FLOAT = TYPE_FLOAT(0.1)
YOLO_V1_BATCH_SIZE: int = 64
YOLO_V1_MOMENTUM: TYPE_FLOAT = TYPE_FLOAT(0.9)
YOLO_V1_DECAY: TYPE_FLOAT = TYPE_FLOAT(0.0005)

# Learning Rate
# [YOLO v1 paper]
# We train the network for about 135 epochs on the training
# and validation data sets from PASCAL VOC 2007 and
# 2012. When testing on 2012 we also include the VOC 2007
# test data for training. Throughout training we use a batch
# size of 64, a momentum of 0:9 and a decay of 0:0005.
# Our learning rate schedule is as follows: For the first
# epochs we slowly raise the learning rate from 10􀀀3 to 10􀀀2.
# If we start at a high learning rate our model often diverges
# due to unstable gradients. We continue training with 10􀀀2
# for 75 epochs, then 10􀀀3 for 30 epochs, and finally 10􀀀4
# for 30 epochs.
#
# NOTE:
# 1e-3 is too high and causes overfitting + model output explodes to inf
# with BN layer being used. Hence, reduce.
YOLO_V1_LR_1ST: TYPE_FLOAT = TYPE_FLOAT(1e-6)
# YOLO_V1_EPOCHS_1ST = 10
YOLO_V1_EPOCHS_1ST = 100
YOLO_V1_LR_2ND: TYPE_FLOAT = TYPE_FLOAT(1e-6)
YOLO_V1_EPOCHS_2ND = 75
YOLO_V1_LR_3RD: TYPE_FLOAT = TYPE_FLOAT(1e-7)
YOLO_V1_EPOCHS_3RD = 30
YOLO_V1_LR_4TH: TYPE_FLOAT = TYPE_FLOAT(1e-8)
YOLO_V1_EPOCHS_4TH = 30

# --------------------------------------------------------------------------------
# YOLO v1 Predictions
# YOLO v1 prediction format = (C=20, B*P for each grid cell.
# P = (cp=1, x=1, y=1, w=1, h=1)
# Total S * S grids, hence model prediction output = (S, S, (C+B*P)).
# --------------------------------------------------------------------------------
# Prediction shape = (C+B*P)
YOLO_V1_PREDICTION_NUM_CLASSES: int = 20   # number of classes
YOLO_V1_PREDICTION_NUM_BBOX: int = 2       # number of bbox per grid cell
YOLO_V1_PREDICTION_NUM_PRED: int = 5       # (cp, x, y, w, h)
YOLO_V1_PREDICTION_INDEX_CP1: int = 20      # Index to cp of the first bbox in P=(C+B*5)
# Index to x in the first BBox
YOLO_V1_PREDICTION_INDEX_X1: int = YOLO_V1_PREDICTION_INDEX_CP1 + 1
# Index to y in the first BBox
YOLO_V1_PREDICTION_INDEX_Y1: int = YOLO_V1_PREDICTION_INDEX_X1 + 1
# Index to w in the first BBox
YOLO_V1_PREDICTION_INDEX_W1: int = YOLO_V1_PREDICTION_INDEX_Y1 + 1
# Index to h in the first BBox
YOLO_V1_PREDICTION_INDEX_H1: int = YOLO_V1_PREDICTION_INDEX_W1 + 1
assert YOLO_V1_PREDICTION_INDEX_X1 == 21
assert YOLO_V1_PREDICTION_INDEX_H1 == 24

YOLO_V1_PREDICTION_INDEX_CP2: int = YOLO_V1_PREDICTION_INDEX_H1 + 1
YOLO_V1_PREDICTION_INDEX_X2: int = YOLO_V1_PREDICTION_INDEX_CP2 + 1
YOLO_V1_PREDICTION_INDEX_Y2: int = YOLO_V1_PREDICTION_INDEX_X2 + 1
YOLO_V1_PREDICTION_INDEX_W2: int = YOLO_V1_PREDICTION_INDEX_Y2 + 1
YOLO_V1_PREDICTION_INDEX_H2: int = YOLO_V1_PREDICTION_INDEX_W2 + 1
assert YOLO_V1_PREDICTION_INDEX_X2 == 26
assert YOLO_V1_PREDICTION_INDEX_H2 == 29

YOLO_V1_LABEL_LENGTH: int = 25
YOLO_V1_LABEL_INDEX_CP: int = 20
YOLO_V1_LABEL_INDEX_X: int = YOLO_V1_LABEL_INDEX_CP + 1
YOLO_V1_LABEL_INDEX_Y: int = YOLO_V1_LABEL_INDEX_X + 1
YOLO_V1_LABEL_INDEX_W: int = YOLO_V1_LABEL_INDEX_Y + 1
YOLO_V1_LABEL_INDEX_H: int = YOLO_V1_LABEL_INDEX_W + 1
