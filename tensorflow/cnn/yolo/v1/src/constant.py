# pylint: disable=invalid-name
import logging
import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
DEBUG_LEVEL: int = logging.INFO
DUMP: bool = False
logging.basicConfig(level=DEBUG_LEVEL)

# --------------------------------------------------------------------------------
# TYPES
# --------------------------------------------------------------------------------
TYPE_FLOAT = np.float32
TYPE_INT = np.int32
ZERO: tf.Tensor = tf.constant(0, dtype=TYPE_FLOAT)
ONE: tf.Tensor = tf.constant(1, dtype=TYPE_FLOAT)
EPSILON = TYPE_FLOAT(1e-6)  # small enough value e.g. to avoid div by zero

YOLO_GRID_SIZE: int = 7

# --------------------------------------------------------------------------------
# YOLO v1 Input
# --------------------------------------------------------------------------------
YOLO_V1_IMAGE_WIDTH: int = 448
YOLO_V1_IMAGE_HEIGHT: int = 448

# --------------------------------------------------------------------------------
# YOLO Model
# --------------------------------------------------------------------------------
YOLO_LEAKY_RELU_SLOPE: TYPE_FLOAT = TYPE_FLOAT(0.1)

# --------------------------------------------------------------------------------
# YOLO v1 Predictions
# YOLO v1 prediction format = (C=20, B*P for each grid cell.
# P = (cp=1, x=1, y=1, w=1, h=1)
# Total S * S grids, hence model prediction output = (S, S, (C+B*P)).
# --------------------------------------------------------------------------------
# Prediction shape = (C+B*P)
YOLO_PREDICTION_NUM_CLASSES: int = 20   # number of classes
YOLO_PREDICTION_NUM_BBOX: int = 2       # number of bbox per grid cell
YOLO_PREDICTION_NUM_PRED: int = 5       # (cp, x, y, w, h)
YOLO_PREDICTION_INDEX_CP1: int = 20      # Index to cp of the first bbox in P=(C+B*5)
# Index to x in the first BBox
YOLO_PREDICTION_INDEX_X1: int = YOLO_PREDICTION_INDEX_CP1 + 1
# Index to y in the first BBox
YOLO_PREDICTION_INDEX_Y1: int = YOLO_PREDICTION_INDEX_X1 + 1
# Index to w in the first BBox
YOLO_PREDICTION_INDEX_W1: int = YOLO_PREDICTION_INDEX_Y1 + 1
# Index to h in the first BBox
YOLO_PREDICTION_INDEX_H1: int = YOLO_PREDICTION_INDEX_W1 + 1
assert YOLO_PREDICTION_INDEX_X1 == 21
assert YOLO_PREDICTION_INDEX_H1 == 24

YOLO_PREDICTION_INDEX_CP2: int = YOLO_PREDICTION_INDEX_H1 + 1
YOLO_PREDICTION_INDEX_X2: int = YOLO_PREDICTION_INDEX_CP2 + 1
YOLO_PREDICTION_INDEX_Y2: int = YOLO_PREDICTION_INDEX_X2 + 1
YOLO_PREDICTION_INDEX_W2: int = YOLO_PREDICTION_INDEX_Y2 + 1
YOLO_PREDICTION_INDEX_H2: int = YOLO_PREDICTION_INDEX_W2 + 1
assert YOLO_PREDICTION_INDEX_X2 == 26
assert YOLO_PREDICTION_INDEX_H2 == 29

YOLO_LABEL_INDEX_CP: int = 20
YOLO_LABEL_INDEX_X: int = YOLO_LABEL_INDEX_CP + 1
YOLO_LABEL_INDEX_Y: int = YOLO_LABEL_INDEX_X + 1
YOLO_LABEL_INDEX_W: int = YOLO_LABEL_INDEX_Y + 1
YOLO_LABEL_INDEX_H: int = YOLO_LABEL_INDEX_W + 1
