"""
YOLO v1 loss function module based on:
1. https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py
2. https://github.com/a-g-moore/YOLO/blob/master/loss.py
"""
import logging
from typing import (
    Union,
    Optional,
)

import numpy as np
import tensorflow as tf
from keras.losses import (
    Loss,
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
    YOLO_PREDICTION_INDEX_H1,
    YOLO_PREDICTION_INDEX_CP2,
    YOLO_PREDICTION_INDEX_X2,
    YOLO_PREDICTION_INDEX_H2,
    YOLO_LABEL_INDEX_CP,
    YOLO_LABEL_INDEX_X,
    YOLO_LABEL_INDEX_H,
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
        self.batch_size: int = -1   # batch size
        self.N: int = -1    # Total cells in the batch (S * S * batch_size)
        self.S: int = S     # Number of division per axis (YOLO divides an image into SxS grids)
        self.B: int = B     # Number of bounding boxes per cell
        self.C: int = C     # Number of classes to detect
        self.P: int = P     # Prediction (cp, x, y, w, h) per bounding box

        # --------------------------------------------------------------------------------
        # lambda parameters to prioritise the localization vs classification
        # --------------------------------------------------------------------------------
        # [YOLO v1 paper]
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

        # --------------------------------------------------------------------------------
        # Identity function IObj_i tells if an object exists in a cell.
        # [Original Paper]
        # where Iobj_i denotes if object appears in cell i and Iobj_i_j denotes that
        # the jth bounding box predictor in cell i is “responsible” for that prediction.
        # Note that the loss function only penalizes classification error if an object
        # is present in that grid cell (hence the conditional class probability discussed).
        # --------------------------------------------------------------------------------
        self.Iobj_i: Optional[tf.Tensor] = None

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

    def IObj_i_j(
            self,
            bounding_boxes: tf.Tensor,
            best_box_indices:tf.Tensor
    ) -> tf.Tensor:
        """Take the responsible bounding box from each cell where an object exists in the cell.

        [YOLO v1 paper]
        Iobj_i_j denotes that the jth bounding box in cell i is “responsible” for that prediction.

        Args:
            bounding_boxes:
                Bounding boxes from all the cells in shape (N, B, D) where (B, D)
                is the B bounding boxes from a cell. D depends on the content.
                When (w,h) is passed, then D==2.

            best_box_indices:
                list of index to the best bounding box of a cell in shape (N,)

        Returns: responsible bounding boxes
        """
        # --------------------------------------------------------------------------------
        # From the bounding boxes of each cell, take the box identified by the beset box index.
        # MatMul X:(N, B, D) with OneHotEncoding:(N, B) extracts the rows as (N, D).
        # --------------------------------------------------------------------------------
        best_boxes: tf.Tensor = tf.einsum(
            "nbd,nb->nd",
            tf.reshape(tensor=bounding_boxes, shape=(self.N, self.B, -1)),
            tf.one_hot(
                indices=tf.reshape(tensor=best_box_indices, shape=(-1)),
                depth=self.B,
                dtype=bounding_boxes.dtype
            )
        )
        # --------------------------------------------------------------------------------
        # Multiply the best boxes:(N, D) with IObj_i:(N, 1) masks non-responsible boxes.
        # --------------------------------------------------------------------------------
        responsible_boxes: tf.Tensor = self.Iobj_i * best_boxes
        return responsible_boxes

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
            Y: prediction of shape (N, S, S, (C+B*P)) where N is batch size.

            [Original Paper]
            Each grid cell predicts B bounding boxes and confidence scores for
            those boxes. These confidence scores reflect how confident the model
            is that the box contains an object and also how accurate it thinks
            the box is that it predicts.

            Formally we define confidence as Pr(Object) * IOU between truth and pred .
            If no object exists in that cell, the confidence scores should be zero.
            Otherwise we want the confidence score to equal the intersection over
            union (IOU) between the predicted box and the ground truth.

            Each bounding box consists of 5 predictions: x, y, w, h, and confidence.
            The (x; y) coordinates represent the center of the box relative to the
            bounds of the grid cell. The width and height are predicted relative to
            the whole image. Finally the confidence prediction represents the IOU
            between the predicted box and any ground truth box.

        Returns: loss
        """
        _name: str = "call()"
        # --------------------------------------------------------------------------------
        # Sanity checks
        # --------------------------------------------------------------------------------
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
        # Reshape Y into N consecutive predictions in shape (N, (C+B*P)) =(N, 30).
        # Reshape T into N consecutive truth in shape (N, (C+P))=(N, 25).
        # All we need are predictions and labels of each cell, hence no need to retain
        # (S x S) geometry of the grids.
        # --------------------------------------------------------------------------------
        Y: tf.Tensor = tf.reshape(tensor=Y, shape=(-1, self.C + self.B * self.P))
        T: tf.Tensor = tf.reshape(tensor=T, shape=(-1, self.C + self.P))
        assert Y.shape[0] == T.shape[0], \
            f"got different number of predictions:[{Y.shape[0]}] and labels:[{T.shape[0]}]"

        self.N: int = int(Y.shape[0])
        self.batch_size = self.N / (self.S * self.S)
        _logger.debug(
            "%s: batch size:[%s] total cells to process:[%s]", self.batch_size, self.N
        )

        self.Iobj_i = T[..., YOLO_LABEL_INDEX_CP:YOLO_LABEL_INDEX_CP+1]
        assert self.Iobj_i.shape == (self.N, 1), \
            f"expected shape {(self.N, 1)} got {self.Iobj_i.shape}."

        # --------------------------------------------------------------------------------
        # Classification loss
        # [YOLO v1 paper]
        # Note that the loss function only penalizes classification error if an object is
        # present in that cell (hence the conditional class probability discussed earlier).
        # --------------------------------------------------------------------------------
        classification_loss: tf.Tensor = tf.math.reduce_mean(tf.math.square(
            self.Iobj.i * Y[..., :self.C],   # class predictions
            self.Iobj.i * T[..., :self.C]    # class truth
        ))

        # --------------------------------------------------------------------------------
        # Localization (c, x, y, w, h)
        # --------------------------------------------------------------------------------
        localization_predicted: tf.Tensor = tf.reshape(
            tensor=Y[..., self.C:],         # Take B*(c,x,y,w,h)
            shape=(-1, self.B, self.P)
        )
        localization_truth: tf.Tensor = tf.reshape(
            tensor=T[..., self.C:],
            shape=(-1, self.P)
        )

        # --------------------------------------------------------------------------------
        # IoU between predicted bounding boxes and the target bbox at a cell.
        # --------------------------------------------------------------------------------
        IOU: tf.Tensor = tf.concat(
            [
                intersection_over_union(
                    localization_predicted[..., j, 1:],     # shape:(N,1,4)
                    localization_truth[..., 1:]             # shape:(N,4)
                )
                for j in range(self.B)
            ],
            axis=-1
        )
        assert IOU.shape == (self.N, self.B)

        # --------------------------------------------------------------------------------
        # Max IOU per grid cell (axis=-1)
        # best_box_j tells which bbox j (0 or 1) is the best box for a cell.
        #
        # [YOLO v1 paper]
        # YOLO predicts multiple bounding boxes per grid cell. At training time we only
        # want one bounding box predictor to be responsible for each object.
        # We assign one predictor to be “responsible” for predicting an object based on
        # which prediction has the highest current IOU with the ground truth.
        # This leads to specialization between the bounding box predictors.
        # Each predictor gets better at predicting certain sizes, aspect ratios, or
        # classes of object, improving overall recall.
        # --------------------------------------------------------------------------------
        max_IOU: tf.Tensor = tf.math.reduce_max(input_tensor=IOU, axis=-1, keepdims=True)
        assert max_IOU.shape == (self.N, 1), \
            f"expected shape {(self.N, 1)} got {IOU.shape}."

        best_box_j: tf.Tensor = tf.reshape(    # argmax drops the last dimension
            tensor=tf.math.argmax(input=IOU, axis=-1, output_type=TYPE_FLOAT),
            shape=(self.N, 1)
        )
        del IOU
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
        # Predicted localization (cp, x, y, w, h) from the best bounding box j per grid cell.
        # This corresponds to Iobj_i_j as picking up the best box j at each cell.
        # if best box j == 0, then YOLO_PREDICTION_INDEX_X1:YOLO_PREDICTION_INDEX_H1+1 as
        # the (x, y, w, h) for the predicted localization. If j == 1, the other.
        #
        # [YOLO v1 paper]
        # It also only penalizes bounding box coordinate error if that predictor is
        # “responsible” for the ground truth box (i.e. has the highest IOU of any
        # predictor in that grid cell).
        # --------------------------------------------------------------------------------
        best_boxes: tf.Tensor = self.IObj_i_j(
            bounding_boxes=localization_predicted,
            best_box_indices=best_box_j
        )
        assert best_boxes.shape == (self.N, self.P), \
            f"expected shape {(self.N, self.P)}, got {best_boxes.shape}"

        # --------------------------------------------------------------------------------
        # Localization_x_y
        # --------------------------------------------------------------------------------
        localization_x_y: tf.Tensor = best_boxes[..., 1:3]  # (x,y) from (cp,x,y,w,h)
        localization_x_y_truth: tf.Tensor = localization_truth[..., 1:3]

        # --------------------------------------------------------------------------------
        # Localization_w_h as square root of w, h.
        # https://datascience.stackexchange.com/questions/118674
        # https://youtu.be/n9_XyCGr-MI?list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&t=2804
        # --------------------------------------------------------------------------------
        # [Original yolo v1 paper]
        # Sum-squared error also equally weights errors in large boxes and small boxes.
        # Our error metric should reflect that small deviations in large boxes matter
        # less than in small boxes. To partially address this we predict the square root
        # of the bounding box width and height instead of the width and height directly.
        # --------------------------------------------------------------------------------
        _w_h: tf.Tensor = best_boxes[..., 3:5]  # (x,y) from (cp,x,y,w,h)
        localization_w_h: tf.Tensor = \
            tf.math.sign(_w_h) * tf.math.sqrt(tf.math.abs(_w_h) + EPSILON)

        _w_h_truth: tf.Tensor = localization_truth[..., 3:5]
        localization_w_h_truth: tf.Tensor = \
            tf.math.sign(_w_h_truth) * tf.math.sqrt(tf.math.abs(_w_h_truth) + EPSILON)

        # --------------------------------------------------------------------------------
        # Localisation loss
        # --------------------------------------------------------------------------------
        localization_loss: tf.Tensor = self.lambda_coord * tf.add_n(
            tf.math.reduce_mean(tf.square(localization_x_y_truth - localization_x_y)),
            tf.math.reduce_mean(tf.square(localization_w_h_truth - localization_w_h)),
        )

        # --------------------------------------------------------------------------------
        # Confidence loss with object
        # [YOLO v1 paper]
        # These confidence scores reflect how confident the model is that the box contains
        # an object and also how accurate it thinks the box is that it predicts.
        # Formally we define confidence as Pr(Object) IOU(truth,pred) . If no object exists
        # in that cell, the confidence scores should be zero.
        # Otherwise we want the confidence score to equal the intersection over union (IOU)
        # between the predicted box and the ground truth.
        #
        # https://stats.stackexchange.com/q/559122
        # the ground-truth value 𝐶𝑖 is computed during training (the IOU).
        #
        # https://github.com/aladdinpersson/Machine-Learning-Collection/pull/44/commits
        # object_loss = self.mse(
        #     torch.flatten(exists_box * target[..., 20:21]),
        #     # To calculate confidence score in paper, I think it should multiply iou value.
        #     torch.flatten(exists_box * target[..., 20:21] * iou_maxes),
        # )
        # --------------------------------------------------------------------------------
        confidence: tf.Tensor = best_boxes[0:1]     # cp from (cp,x,y,w,h)
        confidence_truth: tf.Tensor = self.Iobj_i * max_IOU
        assert confidence.shape == confidence_truth.shape == (self.N, 1)
        confidence_loss: tf.Tensor = tf.math.reduce_mean(tf.square(
            confidence_truth - confidence
        ))

        # --------------------------------------------------------------------------------
        # Confidence loss with no object
        # [YOLO v1 paper]
        # Also, in every image many grid cells do not contain any object.
        # This pushes the “confidence” scores of those cells towards zero, often
        # overpowering the gradient from cells that do contain objects.
        # This can lead to model instability, causing training to diverge early on.
        # To remedy this, we increase the loss from bounding box coordinate predictions
        # and decrease the loss from confidence predictions for boxes that don’t contain
        # objects. We use two parameters, coord and noobj to accomplish this.
        # We set coord = 5 and noobj = :5.
        #
        # no_obj_confidence_truth = Inoobj_i * cp_i = 0 always, because:
        # When Inoobj_i = 1 with no object, then cp_i is 0, hence Inoobj_i * cp_i -> 0.
        # When Inoobj_i = 0 with an object, then again Inoobj_i * cp_i -> 0.
        # --------------------------------------------------------------------------------
        Inoobj_i: tf.Tensor = tf.constant(1.0, dtype=TYPE_FLOAT) - self.Iobj_i
        no_obj_confidences: tf.Tensor = Inoobj_i * localization_predicted[..., 0:1]
        no_obj_confidence_loss: tf.Tensor = \
            self.lambda_noobj * \
            tf.math.reduce_mean(tf.math.square(no_obj_confidences))

        # --------------------------------------------------------------------------------
        # Total loss
        # tf.add_n be more efficient than reduce_sum because it sums the tensors directly.
        # --------------------------------------------------------------------------------
        loss: tf.Tensor = tf.math.add_n([
            localization_loss,
            confidence_loss,
            no_obj_confidence_loss,
            classification_loss
        ])
        return loss
