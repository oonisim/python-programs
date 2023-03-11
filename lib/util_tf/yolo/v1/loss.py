"""
YOLO v1 loss function module based on:
1. https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/loss.py
2. https://github.com/a-g-moore/YOLO/blob/master/loss.py

[Terminology]
input image: 448x448 color image which is divide into an S x S grid.
grid: S x S divisions of an input image
cell: a cell in the grid
responsible cell:
    [YOLO v1 paper]
    If the center of an object falls into a grid cell, that grid cell
    is responsible for detecting that object.
responsible bounding box:
    [YOLO v1 paper]
    YOLO predicts multiple bounding boxes per grid cell.
    At training time we only want one bounding box predictor to be responsible
    for each object. We assign one predictor to be “responsible” for predicting
    an object based on which prediction has the highest current IOU with the
    ground truth.
bbox: bounding box
localization: predicted bounding box (cp, x, y, w, h)
    [YOLO v1 paper]
    Each bounding box consists of 5 predictions: x, y, w, h, and confidence.
    The (x; y) coordinates represent the center of the box relative to the
    bounds of the grid cell. The width and height are predicted relative to
    the whole image.
cp: confidence score = (Pr(Object) * IOU_truth_pred) = IOU
    'p' to distinguish from C/c for Classification
    Pr(Object) will be 0:non-exist or 1:exist, hence cp is expected to be IOU
    as stated in the paper.

    [YOLO v1 paper]
    Formally we define confidence as (Pr(Object) * IOU_truth_pred) . If no object
    exists in that cell, the confidence scores should be zero.
    Otherwise, we want the confidence score to equal the intersection over union
    (IOU) between the predicted box and the ground truth.
x, y:
    center of a bounding box from the left/top corner of a grid, normalized
    between [0, 1] relative to the grid cell size, but can be larger than 1
    if the center is outside the cell.
w, h:
    width of a bounding box normalized between [0, 1] relative to the image
    width and height, but can be larger than 1 when the bounding box is larger
    than the image itself.
Ci/C(i):
    ground truth classification probability for class i in each cell.
C_hat(i):
    predicted conditional classification probability for class i.
    [YOLO v1 paper]
    Each grid cell also predicts C conditional class probabilities, Pr(Classij|Object).
    These probabilities are conditioned on the grid cell containing an object.
    We only predict one set of class probabilities per grid cell, regardless of
    the number of boxes B.

    At test time we multiply the conditional class probabilities and the individual
    box confidence predictions,
    Pr(Class_|Object) * (Pr(Object) * IOU_truth_pred) = Pr(Class_i) * IOU_truth_pred (1)
    which gives us class-specific confidence scores for each box. These scores
    encode both the probability of that class appearing in the box and how well
    the predicted box fits the object.
Iobj_i: 0 or 1 to tell if object appears in cell i
Iobj_j: if the j-th bounding box predictor in cell i is “responsible” bbox.
IOU: Intersection Over Union

S: number of divisions per side. S=7 in YOLO v1.
B: number of bounding boxes to be predicted per cell. B=2 in YOLO v1.
P: size of a prediction of a bounding box = len(cp, x, y, w, h) = 5.
C:
    number of classes to classify an object into. C=20 in YOLO v1.
    YOLO v1 uses PASCAL VOC 2007 and 2012 datasets that have 20 classes.

    [YOLO v1 paper]
    We train the network for about 135 epochs on the training and validation
    data sets from PASCAL VOC 2007 and 2012. When testing on 2012 we also
    include the VOC 2007 test data for training.

non-max-suppression:
    A mechanism to identify one cell that identifies an object to avoid multiple
    cells from detecting the same object. At each cell, C_hat(i) is multiplied
    with the confidence score cp of *every* bounding box, not just with the best
    bounding box with the max IOU at each cell. This generates S*S*B class-specific
    confidence scores for all the bounding boxes.

    Then a box that has the highest class confidence score for an object will be
    identified as the box for the object. This is encoded in the formula in the paper.

    Pr(Class_|Object) * (Pr(Object) * IOU_truth_pred) = Pr(Class_i) * IOU_truth_pred (1)

    See:
    https://medium.com/diaryofawannapreneur/yolo-you-only-look-once-for-object-detection-explained-6f80ea7aaa1e

    [YOLO v1 paper]
    Figure 1: The YOLO Detection System.
    Processing images with YOLO is simple and straightforward. Our system (1) resizes
    the input image to 448 x 448, (2) runs a single convolutional network on the
    image, and (3) thresholds the resulting detections by the model’s confidence.
    ...

    At test time we multiply the conditional class probabilities and the individual
    box confidence predictions,
    Pr(Class_|Object) * (Pr(Object) * IOU_truth_pred) = Pr(Class_i) * IOU_truth_pred (1)
    which gives us class-specific confidence scores for each box. These scores
    encode both the probability of that class appearing in the box and how well
    the predicted box fits the object.
    ...

    Often it is clear which grid cell an object falls in to and the network only
    predicts one box for each object. However, some large objects or objects near
    the border of multiple cells can be well localized by multiple cells.
    Non-maximal suppression can be used to fix these multiple detections.

[References]
* PASCAL VOC (Visual Object Classes) - http://host.robots.ox.ac.uk/pascal/VOC/
* PASCAL VOC 2007 - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ (information and link to data)
* PASCAL VOC 2007 examples - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/examples/index.html
* PASCAL VOC 2007 Development Kit - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/index.html
  (Details about the dataset)
---
Objects of the twenty classes listed above are annotated in the ground truth.
    class:
        the object class e.g. `car' or `bicycle'
    bounding box:
        an axis-aligned rectangle specifying the extent of the object visible in the image.
    view:
        `frontal', `rear', `left' or `right'.
        The views are subjectively marked to indicate the view of the `bulk' of the object.
        Some objects have no view specified.
    `truncated':
        an object marked as `truncated' indicates that the bounding box specified for
        the object does not correspond to the full extent of the object e.g. an image
        of a person from the waist up, or a view of a car extending outside the image.
    `difficult':
        an object marked as `difficult' indicates that the object is considered difficult
        to recognize, for example an object which is clearly visible but unidentifiable
        without substantial use of context. Objects marked as difficult are currently
        ignored in the evaluation of the challenge.
---
* TensorFlow Data Set - https://www.tensorflow.org/datasets/catalog/voc
> This dataset contains the data from the PASCAL Visual Object Classes Challenge,
> corresponding to the Classification and Detection competitions.
> https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/voc.py
---
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
---
"""
# pylint: disable=too-many-statements
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

from util_constant import (
    TYPE_FLOAT,
    TYPE_INT,
)
from constant import (
    DEBUG_LEVEL,
    DUMP,
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
# Layer
# --------------------------------------------------------------------------------
class YOLOLoss(Loss):
    """YOLO v1 objective (loss) layer
    [References]
    https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_losses
    If you need a loss function that takes in parameters beside T and y_pred,
    you can subclass the tf.keras.losses.Loss class and implement the two methods:
    1. __init__(self): accept parameters to pass during the call of the loss function
    2. call(self, y_true, y_pred): compute the model's loss

    https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    > To be implemented by subclasses:
    >     call(): Contains the logic for loss calculation using y_true, y_pred.
    """
    def __init__(
            self,
            S: int = YOLO_GRID_SIZE,                # pylint: disable=invalid-name
            B: int = YOLO_V1_PREDICTION_NUM_BBOX,      # pylint: disable=invalid-name
            C: int = YOLO_V1_PREDICTION_NUM_CLASSES,   # pylint: disable=invalid-name
            P: int = YOLO_V1_PREDICTION_NUM_PRED,      # pylint: disable=invalid-name
            **kwargs
    ):
        """
        Initialization of the Loss instance
        Args:
            S: Number of division per side (YOLO divides an image into SxS grids)
            B: number of bounding boxes per grid cell
            C: number of classes to detect
            P: size of predictions len(cp, x, y, w, h) per bounding box
        """
        super().__init__(**kwargs)
        self.batch_size: TYPE_FLOAT = TYPE_FLOAT(-1)
        # pylint: disable=invalid-name
        self.N: int = -1    # Total cells in the batch (S * S * batch_size) pylint: disable=invalid-name
        self.S: int = S     # pylint: disable=invalid-name
        self.B: int = B     # pylint: disable=invalid-name
        self.C: int = C     # pylint: disable=invalid-name
        self.P: int = P     # pylint: disable=invalid-name

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
        # Identity function Iobj_i tells if an object exists in a cell.
        # [Original Paper]
        # where Iobj_i denotes if object appears in cell i and Iobj_j denotes that
        # the jth bounding box predictor in cell i is “responsible” for that prediction.
        # Note that the loss function only penalizes classification error if an object
        # is present in that grid cell (hence the conditional class probability discussed).
        # --------------------------------------------------------------------------------
        self.Iobj_i: Optional[tf.Tensor] = None     # pylint: disable=invalid-name
        self.Inoobj_i: Optional[tf.Tensor] = None   # pylint: disable=invalid-name

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

    def loss_fn(self, y_true: tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
        """Sum squared loss function
        [YOLO v1 paper]
        We use sum-squared error because it is easy to optimize

        [Note]
        tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        does normalization at axis=-1, but YOLO v1 sums.

        Args:
            y_true: ground truth
            y_pred: prediction

        Returns: batch size normalized loss
        """
        return tf.math.reduce_sum(tf.square(y_true - y_pred)) / self.batch_size

    def Iobj_j(     # pylint: disable=invalid-name
            self,
            bounding_boxes: tf.Tensor,
            best_box_indices: tf.Tensor
    ) -> tf.Tensor:
        """
        Identify the responsible bounding boxes and get predictions from them.

        Iobj_j is supposed to be a binary function to return 0 or 1. However,
        the effective result of Iobj_j is to get (x, y) or (w, h) or (pc) from
        the predictions, hence repurpose it to return them.

        [YOLO v1 paper]
        Iobj_j denotes that the jth bounding box in cell i is “responsible”
        for that prediction.

        Args:
            bounding_boxes:
                Bounding boxes from all the cells in shape (N, B, D) where (B, D)
                is the B bounding boxes from a cell. D depends on the content.
                When (w,h) is passed, then D==2.

            best_box_indices:
                list of index to the best bounding box of a cell in shape (N,)

        Returns: predictions from the responsible bounding boxes
        """
        # --------------------------------------------------------------------------------
        # From the bounding boxes of each cell, take the box identified by the beset box index.
        # MatMul X:(N, B, D) with OneHotEncoding:(N, B) extracts the rows as (N, D).
        # --------------------------------------------------------------------------------
        responsible_boxes: tf.Tensor = tf.einsum(
            "nbd,nb->nd",
            tf.reshape(tensor=bounding_boxes, shape=(self.N, self.B, -1)),
            tf.one_hot(
                indices=tf.reshape(tensor=best_box_indices, shape=(-1)),
                depth=self.B,
                dtype=bounding_boxes.dtype
            )
        )
        return responsible_boxes

    def call(
            self,
            y_true: Union[np.ndarray, tf.Tensor],
            y_pred: Union[np.ndarray, tf.Tensor]
    ) -> tf.Tensor:
        """YOLO loss function calculation
        See ../image/yolo_loss_function.png and the original v1 paper.
        Follow the defined Σ formula strictly. If summation is 1..B, then sum along 1..B.

        [Steps]
        1. Take the max IoU per cell (from between b-boxes and the truth at each cell).
        2. Get Iobj_i_j per each cell i.
           Iobj_i is 1 if the cell i is responsible (cp in truth==1) for an object, or 0.
           Iobj_i_j is 1 if Iobj_i is 1 and bbox j has the max IoU at the cell, or 0.
        3. Take sqrt(w) and sqrt(h) to reduce the impact from the object size.
        4. Calculate localization loss.
        5. Calculate confidence loss.
        6. Calculate classification loss.
        7. Sum the losses

        [References]
        https://www.tensorflow.org/api_docs/python/tf/keras/metrics/mean_squared_error

        Args:
            y_true: ground truth
            y_pred: prediction of shape (N, S, S, (C+B*P)) where N is batch size.

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
        # The model output shape for single input image should be
        # (S=7 * S=7 * (C+B*P)=30) or (S, S, (C+B*P)).
        assert isinstance(y_pred, (np.ndarray, tf.Tensor))
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
        # Reshape y_pred into N consecutive predictions in shape (N, (C+B*P)).
        # Reshape y_true into N consecutive labels in shape (N, (C+P)).
        # All we need are the predictions and label at each cell, hence no need to retain
        # (S x S) util_tf.geometry of the grids.
        # --------------------------------------------------------------------------------
        # pylint: disable=invalid-name
        Y: tf.Tensor = tf.reshape(tensor=y_pred, shape=(-1, self.C + self.B * self.P))
        T: tf.Tensor = tf.reshape(tensor=y_true, shape=(-1, self.C + self.P))
        assert Y.shape[0] == T.shape[0], \
            f"got different number of predictions:[{Y.shape[0]}] and labels:[{T.shape[0]}]"

        self.N: int = int(Y.shape[0])
        self.batch_size = TYPE_FLOAT(self.N / (self.S * self.S))
        _logger.debug(
            "%s: batch size:[%s] total cells:[%s]", _name, self.batch_size, self.N
        )

        self.Iobj_i = T[..., YOLO_V1_LABEL_INDEX_CP:YOLO_V1_LABEL_INDEX_CP+1]
        self.Inoobj_i = ONE - self.Iobj_i
        assert self.Iobj_i.shape == (self.N, 1), \
            f"expected shape {(self.N, 1)} got {self.Iobj_i.shape}."
        DUMP and _logger.debug("%s: self.Iobj_i:[%s]", _name, self.Iobj_i)

        # --------------------------------------------------------------------------------
        # Classification loss
        # [YOLO v1 paper]
        # Note that the loss function only penalizes classification error if an object is
        # present in that cell (hence the conditional class probability discussed earlier).
        # --------------------------------------------------------------------------------
        classification_loss: tf.Tensor = self.loss_fn(
            y_true=self.Iobj_i * T[..., :self.C],
            y_pred=self.Iobj_i * Y[..., :self.C],
        )
        _logger.debug("%s: classification_loss[%s]", _name, classification_loss)

        # --------------------------------------------------------------------------------
        # Bounding box predictions (c, x, y, w, h)
        # --------------------------------------------------------------------------------
        box_pred: tf.Tensor = tf.reshape(
            tensor=Y[..., self.C:],         # Take B*(c,x,y,w,h)
            shape=(-1, self.B, self.P)
        )
        box_true: tf.Tensor = tf.reshape(
            tensor=T[..., self.C:],
            shape=(-1, self.P)
        )
        DUMP and _logger.debug(
            "%s: box_pred shape:%s\n[%s]", _name, box_pred.shape, box_pred
        )

        # --------------------------------------------------------------------------------
        # IoU between predicted bounding boxes and the ground truth at a cell.
        # --------------------------------------------------------------------------------
        IOU: tf.Tensor = tf.concat(            # pylint: disable=invalid-name
            values=[
                intersection_over_union(
                    box_pred[..., j, 1:5],     # shape:(N,4)
                    box_true[..., 1:5]         # shape:(N,4)
                )
                for j in range(self.B)
            ],
            axis=-1,
            name="IOU"
        )
        assert IOU.shape == (self.N, self.B)

        # --------------------------------------------------------------------------------
        # Max IOU per grid cell (axis=-1)
        # best_box_j tells which bbox j has the max IOU.
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
        # pylint: disable=invalid-name
        max_IOU: tf.Tensor = tf.math.reduce_max(input_tensor=IOU, axis=-1, keepdims=True)
        assert max_IOU.shape == (self.N, 1), \
            f"expected max IOU shape {(self.N, 1)}, got {IOU.shape}."
        DUMP and _logger.debug("%s: max_IOU[%s]", _name, max_IOU)

        best_box_j: tf.Tensor = tf.reshape(    # argmax drops the last dimension
            tensor=tf.math.argmax(input=IOU, axis=-1, output_type=TYPE_INT),
            shape=(self.N, 1)
        )
        DUMP and _logger.debug("%s: best_box_j:%s", _name, best_box_j)
        del IOU

        # --------------------------------------------------------------------------------
        # Bbox prediction (cp, x, y, w, h) from the responsible bounding box j per cell.
        # This corresponds to Iobj_j as picking up the best box j at each cell.
        # if best box j == 0, then YOLO_V1_PREDICTION_INDEX_X1:YOLO_V1_PREDICTION_INDEX_H1+1 as
        # the (x, y, w, h) for the predicted localization. If j == 1, the other.
        #
        # [YOLO v1 paper]
        # It also only penalizes bounding box coordinate error if that predictor is
        # “responsible” for the ground truth box (i.e. has the highest IOU of any
        # predictor in that grid cell).
        # --------------------------------------------------------------------------------
        best_boxes: tf.Tensor = self.Iobj_j(
            bounding_boxes=box_pred,
            best_box_indices=best_box_j
        )
        assert best_boxes.shape == (self.N, self.P), \
            f"expected shape {(self.N, self.P)}, got {best_boxes.shape}"
        DUMP and _logger.debug("%s: best_boxes[%s]", _name, best_boxes)

        # --------------------------------------------------------------------------------
        # Localization loss (x, y)
        # --------------------------------------------------------------------------------
        x_y_loss = self.lambda_coord * self.loss_fn(
            y_true=self.Iobj_i * best_boxes[..., 1:3],  # (x,y) from (cp,x,y,w,h)
            y_pred=self.Iobj_i * box_true[..., 1:3]
        )
        _logger.debug("%s: x_y_loss[%s]", _name, x_y_loss)

        # --------------------------------------------------------------------------------
        # Localization loss (sqrt(w), sqrt(h))
        # https://datascience.stackexchange.com/questions/118674
        # https://youtu.be/n9_XyCGr-MI?list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&t=2804
        # --------------------------------------------------------------------------------
        # Prevent 1/sqrt(x=0) in the gradient during the back propagation.
        # Gradient sqrt(x) is 0.5*sqrt(x). sqrt(abs(x)+eps) to avoid infinity by sqrt(x=0).
        # sign(x) to restore the original sign which is lost via abs(x).
        # --------------------------------------------------------------------------------
        # [Original yolo v1 paper]
        # Sum-squared error also equally weights errors in large boxes and small boxes.
        # Our error metric should reflect that small deviations in large boxes matter
        # less than in small boxes. To partially address this we predict the square root
        # of the bounding box width and height instead of the width and height directly.
        # --------------------------------------------------------------------------------
        _w_h_pred: tf.Tensor = best_boxes[..., 3:5]  # (x,y) from (cp,x,y,w,h)
        sqrt_w_h_pred: tf.Tensor = \
            tf.math.sign(_w_h_pred) * tf.math.sqrt(tf.math.abs(_w_h_pred) + EPSILON)

        _w_h_true: tf.Tensor = box_true[..., 3:5]
        sqrt__w_h_true: tf.Tensor = \
            tf.math.sign(_w_h_true) * tf.math.sqrt(tf.math.abs(_w_h_true) + EPSILON)

        w_h_loss = self.lambda_coord * self.loss_fn(
            y_true=self.Iobj_i * sqrt__w_h_true,
            y_pred=self.Iobj_i * sqrt_w_h_pred
        )
        _logger.debug("%s: w_h_loss[%s]", _name, w_h_loss)

        # --------------------------------------------------------------------------------
        # Confidence loss with an object in a cell
        # [YOLO v1 paper]
        # These confidence scores reflect how confident the model is that the box contains
        # an object and also how accurate it thinks the box is that it predicts.
        # Formally we define confidence as Pr(Object) IOU(truth,pred) . If no object exists
        # in that cell, the confidence scores should be zero.
        # Otherwise we want the confidence score to equal the intersection over union (IOU)
        # between the predicted box and the ground truth.
        #
        # https://stats.stackexchange.com/q/559122
        # the ground-truth value 𝐶𝑖 is computed during training (IOU).
        #
        # https://github.com/aladdinpersson/Machine-Learning-Collection/pull/44/commits
        # object_loss = self.mse(
        #     torch.flatten(exists_box * target[..., 20:21]),
        #     # To calculate confidence score in paper, I think it should multiply iou value.
        #     torch.flatten(exists_box * target[..., 20:21] * iou_maxes),
        # )
        # --------------------------------------------------------------------------------
        confidence_pred: tf.Tensor = best_boxes[..., 0:1]     # cp from (cp,x,y,w,h)
        confidence_true: tf.Tensor = max_IOU
        assert confidence_pred.shape == confidence_true.shape == (self.N, 1), \
            f"expected confidence shape:{(self.N, 1)}, got " \
            f"confidence_pred:{confidence_pred} confidence_truth:{confidence_true.shape}"
        confidence_loss: tf.Tensor = self.loss_fn(
            y_true=self.Iobj_i * confidence_true,
            y_pred=self.Iobj_i * confidence_pred
        )
        _logger.debug("%s: confidence_loss[%s]", _name, confidence_loss)

        # --------------------------------------------------------------------------------
        # Confidence loss with no object
        # --------------------------------------------------------------------------------
        # [YOLO v1 paper]
        # Also, in every image many grid cells do not contain any object.
        # This pushes the “confidence” scores of those cells towards zero, often
        # overpowering the gradient from cells that do contain objects.
        # This can lead to model instability, causing training to diverge early on.
        # To remedy this, we increase the loss from bounding box coordinate predictions
        # and decrease the loss from confidence predictions for boxes that don’t contain
        # objects. We use two parameters, coord and noobj to accomplish this.
        # We set coord = 5 and noobj = :5.
        # --------------------------------------------------------------------------------
        # Each cell has B number of cp in (cp,x,y,w,h).
        # Calculate C_hat(i) by taking the sum of B number of cp per cell on axis=1
        # as per the loss function formula Σ (C(i)-C_hat(i))^2 along (1..B) reducing
        # box_pred[..., 0] of shape (N, B) into shape (N, 1) with keepdims=True.
        no_obj_confidences_pred: tf.Tensor = \
            self.Inoobj_i * tf.math.reduce_sum(box_pred[..., 0], axis=-1, keepdims=True)
        assert no_obj_confidences_pred.shape == (self.N, 1)

        # No subtraction of no_obj_confidence_true.
        # no_obj_confidence_true
        # = Inoobj_i * box_true[..., 0]
        # = Inoobj_i * Iobj_i
        # = (1 - Iobj_i) * Iobj_i
        # = Iobj_i - Iobj_i^2
        # = Iobj_i - Iobj_i     # Iobj_i^2 == Iobj_i because it is either 1 or 0
        # = 0
        # or ...
        # no_obj_confidence_true = Inoobj_i * cp_i = 0 always, because:
        # When Inoobj_i = 1 with no object, then cp_i is 0, hence Inoobj_i * cp_i -> 0.
        # When Inoobj_i = 0 with an object, then again Inoobj_i * cp_i -> 0.
        # no_obj_confidence_true = self.Inoobj_i * self.Iobj_i
        no_obj_confidence_true: tf.Tensor = ZERO
        no_obj_confidence_loss: tf.Tensor = self.lambda_noobj * self.loss_fn(
            y_true=no_obj_confidence_true,
            y_pred=self.Inoobj_i * no_obj_confidences_pred
        )
        _logger.debug("%s: no_obj_confidence_loss[%s]", _name, no_obj_confidence_loss)

        # --------------------------------------------------------------------------------
        # Total loss
        # tf.add_n be more efficient than reduce_sum because it sums the tensors directly.
        # --------------------------------------------------------------------------------
        loss: tf.Tensor = tf.math.add_n([
            x_y_loss,
            w_h_loss,
            confidence_loss,
            no_obj_confidence_loss,
            classification_loss
        ])
        _logger.debug("%s: loss[%s]", _name, loss)
        return loss


def main():
    """Simple test run"""
    loss: Loss = YOLOLoss()

    S: int = YOLO_GRID_SIZE                  # pylint: disable=invalid-name
    C: int = YOLO_V1_PREDICTION_NUM_CLASSES     # pylint: disable=invalid-name
    B: int = YOLO_V1_PREDICTION_NUM_BBOX        # pylint: disable=invalid-name
    P: int = YOLO_V1_PREDICTION_NUM_PRED        # pylint: disable=invalid-name
    y_pred: tf.Tensor = tf.constant(np.ones(shape=(1, S, S, C+B*P)), dtype=TYPE_FLOAT)
    y_true: tf.Tensor = tf.constant(np.zeros(shape=(1, S, S, C+P)), dtype=TYPE_FLOAT)
    loss: tf.Tensor = loss(y_pred=y_pred, y_true=y_true)
    print(loss)


if __name__ == "__main__":
    main()
