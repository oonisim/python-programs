import json
import logging
from typing import (
    List,
    Dict,
    Tuple,
    Callable,
    Optional,
    Union,
    Iterable,
)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import (
    Loss,
    MeanSquaredError,
)

from util_logging import (
    get_logger,
)
from constant import (
    TYPE_FLOAT,
    EPSILON,
    YOLO_GRID_SIZE,
    YOLO_PREDICTION_NUM_CLASSES,
    YOLO_PREDICTION_NUM_BBOX,
    YOLO_PREDICTION_NUM_PRED,
    YOLO_PREDICTION_INDEX_CP1,
    YOLO_PREDICTION_INDEX_X1,
    YOLO_PREDICTION_INDEX_Y1,
    YOLO_PREDICTION_INDEX_W1,
    YOLO_PREDICTION_INDEX_H1,
    YOLO_PREDICTION_INDEX_CP2,
    YOLO_PREDICTION_INDEX_X2,
    YOLO_PREDICTION_INDEX_Y2,
    YOLO_PREDICTION_INDEX_W2,
    YOLO_PREDICTION_INDEX_H2,
    YOLO_LABEL_INDEX_CP,
    YOLO_LABEL_INDEX_X,
    YOLO_LABEL_INDEX_Y,
    YOLO_LABEL_INDEX_W,
    YOLO_LABEL_INDEX_H,
)
from utils import (
    intersection_over_union,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------------
class YOLOLoss(Loss):
    """YOLO v1 objective (loss) layer"""
    def __init__(
            self,
            S: int = YOLO_GRID_SIZE,                # pylint: disable=invalid-name
            B: int = YOLO_PREDICTION_NUM_BBOX,      # pylint: disable=invalid-name
            C: int = YOLO_PREDICTION_NUM_CLASSES,   # pylint: disable=invalid-name
            P: int = YOLO_PREDICTION_NUM_PRED,      # pylint: disable=invalid-name
            **kwargs
    ):
        """
        Initialization of the Loss layer
        Args:
            S: size of grid
            B: number of bounding box per grid cell
            C: number of classes
            P: number of predictions per bounding box (cp, x, y, w, h)
        """
        super().__init__(reduction=tf.keras.losses.Reduction)
        self.N: int = -1    # Number of batch records
        self.S: int = S
        self.B: int = B
        self.C: int = C
        self.P: int = P

        # --------------------------------------------------------------------------------
        # lambda parameters to prioritise the localization vs classification
        # --------------------------------------------------------------------------------
        # YOLO uses sum-squared error because it is easy to optimize,
        # however it does not perfectly align with our goal of maximizing
        # average precision. It weights localization error equally with
        # classification error which may not be ideal.
        # Also, in every image many grid cells do not contain any
        # object. This pushes the “confidence” scores of those cells
        # towards zero, often overpowering the gradient from cells
        # that do contain objects. This can lead to model instability,
        # causing training to diverge early on.
        # To remedy this, we increase the loss from bounding box
        # coordinate predictions and decrease the loss from confidence
        # predictions for boxes that don’t contain objects. We
        # use two parameters, lambda_coord=5 and lambda_noobj=0.5
        # --------------------------------------------------------------------------------
        self.lambda_coord: TYPE_FLOAT = TYPE_FLOAT(5.0)
        self.lambda_noobj: TYPE_FLOAT = TYPE_FLOAT(0.5)

    def build(self, input_shape):
        """build the layer state
        Args:
            input_shape: TensorShape, or list of instances of TensorShape
        """
        # Tell Keras the layer is built
        super().build(input_shape=input_shape)

    def get_config(self) -> dict:
        """
        Return serializable layer configuration from which the layer can be reinstantiated.
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_config
        """
        config = super().get_config().copy()
        config.update({
            'S': self.S,
            'B': self.B,
            'C': self.C,
            'P': self.P,
            'lambda_coord': self.lambda_coord,
            'lambda_noobj': self.lambda_noobj,
        })
        return config

    def call(
            self, y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor]
    ) -> tf.Tensor:
        """Loss forward process
        https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss

        Args:
            y_true: ground truth
            y_pred: prediction of shape (N, S, S, (C+B*P)) where N is batch size

        Returns: loss
        """
        # The model output shape for one input image should be
        # (S=7 * S=7 * (C+B*P)=30) or (S, S, (C+B*P)).
        assert isinstance(y_pred, (np.ndarray, tf.Tensor, tf.Variable))
        assert y_pred.shape[-1] in (
            (self.S * self.S * (self.C + self.B * self.P)),
            (self.C + self.B * self.P)
        )

        # The label shape for one input image should be
        # (S=7 * S=7 * (C+P)=25) or (S, S, (C+P)).
        assert isinstance(y_true, (np.ndarray, tf.Tensor))
        assert y_true.shape[-1] in (
            (self.S * self.S * (self.C + self.P)),
            (self.C + self.P)
        )

        # --------------------------------------------------------------------------------
        # Convert to tf.Variable to be able to update it.
        # y_pred into (N, S, S, (C+B*P)) of YOLO v1 model output.
        # y_true into (N, S, S, (C+P)
        # --------------------------------------------------------------------------------
        y_pred: tf.Variable = tf.Variable(y_pred).reshape((-1, self.S, self.S, self.C + self.B * self.P))
        y_true: tf.Tensor = tf.Variable(y_true).reshape((-1, self.S, self.S, self.C + self.P))
        assert y_pred.shape[0] == y_true.shape[0]
        self.N: tf.Tensor = y_pred.shape[0]
        _logger.debug("%s: batch size [%s]", self.N)

        # --------------------------------------------------------------------------------
        # IoU per predicted bounding box with target bbox
        # --------------------------------------------------------------------------------
        iou_b1 = intersection_over_union(
            y_pred[..., YOLO_PREDICTION_INDEX_X1:YOLO_PREDICTION_INDEX_H1+1],   # (x,y,w,h) for the first bbox
            y_true[..., YOLO_LABEL_INDEX_X:YOLO_LABEL_INDEX_H+1]                # (x,y,w,h) for label
        )
        iou_b2 = intersection_over_union(
            y_pred[..., YOLO_PREDICTION_INDEX_X2:YOLO_PREDICTION_INDEX_H2+1],
            y_true[..., YOLO_LABEL_INDEX_X:YOLO_LABEL_INDEX_H+1]                # (x,y,w,h) for label
        )
        # --------------------------------------------------------------------------------
        # Max IOU per grid cell (axis=-1)
        # best_box_j tells which bbox j (0 or 1) is the best box for a cell.
        # --------------------------------------------------------------------------------
        IOU: tf.Tensor = tf.math.reduce_max(input_tensor=[iou_b1, iou_b2], axis=-1, keepdims=True)
        best_box_j: tf.Tensor = tf.reshape(    # argmax drops the last dimension
            tensor=tf.math.argmax(input=[iou_b1, iou_b2], axis=-1, output_type=TYPE_FLOAT),
            shape=(self.N, self.S, self.S, 1)
        )
        assert IOU.shape == (self.N, self.S, self.S, 1), \
            f"expected shape {(self.N, self.S, self.S, 1)} got {IOU.shape}."
        assert (
                tf.math.reduce_all(best_box_j == tf.constant(value=0, dtype=TYPE_FLOAT)) or
                tf.math.reduce_all(best_box_j == tf.constant(value=1, dtype=TYPE_FLOAT))
        ), "expected best_box_j in (0, 1), got {}".format(
            best_box_j[tf.math.logical_and(
                best_box_j != tf.constant(value=0, dtype=TYPE_FLOAT),
                best_box_j != tf.constant(value=1, dtype=TYPE_FLOAT))
            ]
        )

        # --------------------------------------------------------------------------------
        # Identity function IObj_i tells if an object exists in the cell as a responsible cell.
        # [Original Paper]
        # where Iobj_i denotes if object appears in cell i and Iobj_i_j denotes that
        # the jth bounding box predictor in cell i is “responsible” for that prediction.
        # Note that the loss function only penalizes classification error if an object
        # is present in that grid cell (hence the conditional class probability discussed).
        # --------------------------------------------------------------------------------
        identity_i: tf.Tensor = y_true[..., YOLO_LABEL_INDEX_CP:YOLO_LABEL_INDEX_CP+1]
        assert identity_i.shape == (self.N, self.S, self.S, 1), \
            f"expected shape {(self.N, self.S, self.S, 1)} got {identity_i.shape}."

        # --------------------------------------------------------------------------------
        # Predicted localization (x, y, w, h) from the best bounding box per grid cell.
        # This corresponds to Iobj_i_j as picking up the best box j for each cell.
        # if best box j == 0, then YOLO_PREDICTION_INDEX_X1:YOLO_PREDICTION_INDEX_H1+1 as
        # the (x, y, w, h) for the predicted localization. If j == 1, the other.
        # --------------------------------------------------------------------------------
        boxes_predicted: tf.Tensor = identity_i * (
            (TYPE_FLOAT(1.0) - best_box_j) * y_pred[..., YOLO_PREDICTION_INDEX_X1:YOLO_PREDICTION_INDEX_H1+1] +
            best_box_j * y_pred[..., YOLO_PREDICTION_INDEX_X2:YOLO_PREDICTION_INDEX_H2+1]
        )

        # --------------------------------------------------------------------------------
        # True localization (x, y, w, h) from per grid cell.
        # --------------------------------------------------------------------------------
        boxes_truth: tf.Tensor = identity_i * y_true[YOLO_LABEL_INDEX_X:YOLO_LABEL_INDEX_H+1]
        assert YOLO_PREDICTION_INDEX_H1-YOLO_PREDICTION_INDEX_X1 == 4
        assert YOLO_LABEL_INDEX_H-YOLO_LABEL_INDEX_X == 4

        # --------------------------------------------------------------------------------
        # Take square root of w, h
        # [Original yolo v1 paper]
        # Sum-squared error also equally weights errors in large boxes and small boxes.
        # Our error metric should reflect that small deviations in large boxes matter
        # less than in small boxes. To partially address this we predict the square root
        # of the bounding box width and height instead of the width and height directly.
        # --------------------------------------------------------------------------------
        # (x/0, y/1, w/2, h/3)
        w_h_predicted = boxes_predicted[..., 2:4]
        w_h_predicted.assign(
            # --------------------------------------------------------------------------------
            # abs() as predicted values can be negative. EPSILON to avoid the derivative of
            # sqrt(x), which is 0.5 * 1/sqrt(x), from getting infinitive when x == 0.
            # --------------------------------------------------------------------------------
            tf.math.sign(w_h_predicted) * tf.math.sqrt(tf.math.abs(w_h_predicted) + EPSILON)
        )
        boxes_truth[..., 2:4].assign(
            tf.math.sqrt(tf.math.abs(boxes_truth[..., 2:4]) + EPSILON)
        )

        return tf.math.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))
