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
    """YOLO v1 objective (loss) layer
    [References]
    https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_losses
    If you need a loss function that takes in parameters beside T and y_pred,
    you can subclass the tf.keras.losses.Loss class and implement the two methods:
    1. __init__(self): accept parameters to pass during the call of the loss function
    2. call(self, T, y_pred): compute the model's loss

    https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    > To be implemented by subclasses:
    >     call(): Contains the logic for loss calculation using T, y_pred.
    """
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
        super().__init__(reduction=tf.keras.losses.Reduction, **kwargs)
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

    def get_config(self) -> dict:
        """Returns the config dictionary for a Loss instance.
        https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss#get_config
        Return serializable layer configuration from which the instance can be reinstantiated.
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
            self,
            T: Union[np.ndarray, tf.Tensor],
            Y: Union[np.ndarray, tf.Tensor]
    ) -> tf.Tensor:
        """Loss forward process
        See ../image/yolo_loss_function.png and the original v1 paper.
        [Steps]
        1. Take the max IoU per cell (from between b-boxes and the truth at each cell).
        2. Get Iobj_i_j per each cell i.
           IObj_i is 1 if the cell i is responsible (pc in truth==1) for an object, or 0.
           IObj_i_j is 1 if IObj_i is 1 and bbox j has the max IoU at the cell, or 0.
        3. Take sqrt(w) and sqrt(h) to reduce the impact from the object size.

        [References]
        https://www.tensorflow.org/api_docs/python/tf/keras/metrics/mean_squared_error

        Args:
            T: ground truth
            Y: prediction of shape (N, S, S, (C+B*P)) where N is batch size

        Returns: loss
        """
        # The model output shape for one input image should be
        # (S=7 * S=7 * (C+B*P)=30) or (S, S, (C+B*P)).
        assert isinstance(Y, (np.ndarray, tf.Tensor, tf.Variable))
        assert Y.shape[-1] in (
            (self.S * self.S * (self.C + self.B * self.P)),
            (self.C + self.B * self.P)
        )

        # The label shape for one input image should be
        # (S=7 * S=7 * (C+P)=25) or (S, S, (C+P)).
        assert isinstance(T, (np.ndarray, tf.Tensor))
        assert T.shape[-1] in (
            (self.S * self.S * (self.C + self.P)),
            (self.C + self.P)
        )

        # --------------------------------------------------------------------------------
        # Convert to tf.Variable to be able to update it.
        # Y into (N, S, S, (C+B*P)) of YOLO v1 model output.
        # T into (N, S, S, (C+P)
        # --------------------------------------------------------------------------------
        Y: tf.Variable = tf.Variable(Y).reshape((-1, self.S, self.S, self.C + self.B * self.P))
        T: tf.Tensor = tf.Variable(T).reshape((-1, self.S, self.S, self.C + self.P))
        assert Y.shape[0] == T.shape[0]
        self.N: tf.Tensor = Y.shape[0]
        _logger.debug("%s: batch size [%s]", self.N)

        # --------------------------------------------------------------------------------
        # IoU per predicted bounding box and target bbox
        # --------------------------------------------------------------------------------
        iou_b1 = intersection_over_union(
            Y[..., YOLO_PREDICTION_INDEX_X1:YOLO_PREDICTION_INDEX_H1+1],   # (x,y,w,h) for the first bbox
            T[..., YOLO_LABEL_INDEX_X:YOLO_LABEL_INDEX_H+1]                # (x,y,w,h) for label
        )
        iou_b2 = intersection_over_union(
            Y[..., YOLO_PREDICTION_INDEX_X2:YOLO_PREDICTION_INDEX_H2+1],
            T[..., YOLO_LABEL_INDEX_X:YOLO_LABEL_INDEX_H+1]                # (x,y,w,h) for label
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
        Iobj_i: tf.Tensor = T[..., YOLO_LABEL_INDEX_CP:YOLO_LABEL_INDEX_CP+1]
        assert Iobj_i.shape == (self.N, self.S, self.S, 1), \
            f"expected shape {(self.N, self.S, self.S, 1)} got {Iobj_i.shape}."

        # --------------------------------------------------------------------------------
        # Predicted localization (x, y, w, h) from the best bounding box j per grid cell.
        # This corresponds to Iobj_i_j as picking up the best box j at each cell.
        # if best box j == 0, then YOLO_PREDICTION_INDEX_X1:YOLO_PREDICTION_INDEX_H1+1 as
        # the (x, y, w, h) for the predicted localization. If j == 1, the other.
        # --------------------------------------------------------------------------------
        boxes_predicted: tf.Tensor = Iobj_i * (
            (TYPE_FLOAT(1.0) - best_box_j) * Y[..., YOLO_PREDICTION_INDEX_X1:YOLO_PREDICTION_INDEX_H1+1] +
            best_box_j * Y[..., YOLO_PREDICTION_INDEX_X2:YOLO_PREDICTION_INDEX_H2+1]
        )

        # --------------------------------------------------------------------------------
        # True localization (x, y, w, h) per grid cell.
        # --------------------------------------------------------------------------------
        boxes_truth: tf.Tensor = Iobj_i * T[YOLO_LABEL_INDEX_X:YOLO_LABEL_INDEX_H+1]
        assert YOLO_PREDICTION_INDEX_H1-YOLO_PREDICTION_INDEX_X1 == 4
        assert YOLO_LABEL_INDEX_H-YOLO_LABEL_INDEX_X == 4

        # --------------------------------------------------------------------------------
        # Take square root of w, h.
        # https://datascience.stackexchange.com/questions/118674
        # https://youtu.be/n9_XyCGr-MI?list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&t=2804
        # --------------------------------------------------------------------------------
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
            # abs(x) as predicted values can be negative. EPSILON to avoid the derivative of
            # sqrt(x), which is 0.5 * 1/sqrt(x), from getting infinitive when x == 0.
            # sign(x) to restore the sign lost due to abs(x).
            # --------------------------------------------------------------------------------
            tf.math.sign(w_h_predicted) * tf.math.sqrt(tf.math.abs(w_h_predicted) + EPSILON)
        )
        # (x/0, y/1, w/2, h/3)
        w_h_truth = boxes_truth[..., 2:4]
        w_h_truth.assign(tf.math.sqrt(w_h_truth))

        # --------------------------------------------------------------------------------
        # Localisation loss via MSE sum(T - Y)^2 / N
        # mean(
        #   (x_pred-x_true)^2 +
        #   (x_pred-x_true)^2 +
        #   (sqrt(w_pred)-sqrt(w_true))^2 +
        #   (sqrt(h_pred)-sqrt(h_true))^2
        # )
        # --------------------------------------------------------------------------------
        localization_loss: tf.Tensor = tf.math.reduce_mean(tf.square(
            boxes_truth - boxes_predicted
        ))

        # --------------------------------------------------------------------------------
        # Classification loss
        # --------------------------------------------------------------------------------

