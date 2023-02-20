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
            S: int = 7,
            B: int = 2,
            C: int = 20,
            P: int = 5,
            **kwargs
    ):
        """
        Initialization of the Loss layer
        Args:
            S: size of grid
            B: number of bounding box per grid cell
            C: number of classes
            P: number of predictions per bounding box (x, y, w, h, c)
        """
        super().__init__(reduction=tf.keras.losses.Reduction)
        self.N: int = -1    # Number of bagtch records
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
            y_pred: prediction

        Returns: loss
        """
        # Make sure the final output of YOLO v1 is (N
        y_pred: tf.Tensor = y_pred.reshape((-1, self.S, self.S, self.C + self.B * self.P))
        self.N = y_pred.shape[0]

        return tf.math.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))
