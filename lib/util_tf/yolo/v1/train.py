"""YOLO v1 training module
"""
import logging
from typing import (
    Union,
    Optional,
)

import numpy as np
from tensorflow import keras    # pylint: disable=unused-import
import tensorflow as tf
from keras.losses import (
    Loss,
    # MeanSquaredError,
)

from constant import (
    DEBUG_LEVEL,
    DUMP,
    TYPE_FLOAT,
    TYPE_INT,
    ZERO,       # 0.0 of type TYPE_FLOAT
    ONE,        # 1.0 of type TYPE_FLOAT
    EPSILON,
    YOLO_GRID_SIZE,
    YOLO_V1_PREDICTION_NUM_CLASSES,
    YOLO_V1_PREDICTION_NUM_BBOX,
    YOLO_V1_PREDICTION_NUM_PRED,
    YOLO_V1_LABEL_INDEX_CP,
)
from util_logging import (
    get_logger,
)
from utils import (
    intersection_over_union,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__, level=DEBUG_LEVEL)


# --------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------
