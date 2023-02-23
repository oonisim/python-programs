import sys
sys.path.append("/Users/oonisim/home/repository/git/oonisim/python-programs/lib")
sys.path.append("/Users/oonisim/home/repository/git/oonisim/python-programs/tensorflow/cnn/yolo/v1/src")

import logging
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras


from constant import (
    DEBUG_LEVEL,
    TYPE_FLOAT,
    TYPE_INT,
    EPSILON,
    YOLO_GRID_SIZE,
    YOLO_PREDICTION_NUM_CLASSES,
    YOLO_PREDICTION_NUM_BBOX,
    YOLO_PREDICTION_NUM_PRED,
)
from loss import (
    YOLOLoss
)
from util_logging import (
    get_logger
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__, level=DEBUG_LEVEL)


def intersection_over_union(box1, box2):
    box1x1 = box1[..., 0] - box1[..., 2] / 2
    box1y1 = box1[..., 1] - box1[..., 3] / 2
    box1x2 = box1[..., 0] + box1[..., 2] / 2
    box1y2 = box1[..., 1] + box1[..., 3] / 2
    box2x1 = box2[..., 0] - box2[..., 2] / 2
    box2y1 = box2[..., 1] - box2[..., 3] / 2
    box2x2 = box2[..., 0] + box2[..., 2] / 2
    box2y2 = box2[..., 1] + box2[..., 3] / 2

    box1area = torch.abs((box1x1 - box1x2) * (box1y1 - box1y2))
    box2area = torch.abs((box2x1 - box2x2) * (box2y1 - box2y2))

    x1 = torch.max(box1x1, box2x1)
    y1 = torch.max(box1y1, box2y1)
    x2 = torch.min(box1x2, box2x2)
    y2 = torch.min(box1y2, box2y2)

    intersection_area = torch.clamp(x2 - x1, min = 0) * torch.clamp(y2 - y1, min = 0)

    iou = intersection_area / (box1area + box2area - intersection_area + 1e-6)
    return iou


class TorchYoloLoss(nn.Module):
    def __init__(self, num_classes=20, num_boxes = 2):
        super(TorchYoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        _name: str = "forward()"
        batch_size = target.size()[0]

        predictions = predictions.reshape(-1, 7, 7, self.num_classes + 5 * self.num_boxes)
        class_predictions = predictions[..., :self.num_classes]

        class_target = target[..., :self.num_classes]
        indicator_i = target[..., self.num_classes].unsqueeze(3)
        _logger.debug("%s indicator_i: shape %s\n%s", _name, indicator_i.shape, indicator_i)

        # class loss
        class_loss = self.mse(
            indicator_i * class_predictions,
            indicator_i * class_target
        )
        _logger.debug("%s: class_loss[%s]", _name, class_loss)

        box_predictions = predictions[..., self.num_classes:].reshape(-1, 7, 7, self.num_boxes, 5)
        # print(f"box_predictions: shape:{box_predictions.shape}\n{box_predictions}")
        _logger.debug(
            "%s: box_predictions shape:%s\n[%s]",
            _name, box_predictions.shape, box_predictions
        )

        box_target = target[..., self.num_classes:]
        box_target = torch.cat((box_target, box_target), dim=3).reshape(-1, 7, 7, self.num_boxes, 5)

        iou = torch.cat(
            [
                intersection_over_union(
                    box_predictions[..., i, 1:],
                    box_target[..., i, 1:]
                ).unsqueeze(3).unsqueeze(0)
                for i in range(self.num_boxes)
            ],
            dim = 0
        )
        # print(f"iou: shape:{iou.shape}\n{iou}")

        best_iou, best_box = torch.max(iou, dim = 0)
        # print(f"best_box: shape:{best_box.shape}\n{best_box}")
        _logger.debug("%s: best_box[%s]", _name, best_box)

        first_box_mask = torch.cat((torch.ones_like(indicator_i), torch.zeros_like(indicator_i)), dim=3)
        second_box_mask = torch.cat((torch.zeros_like(indicator_i), torch.ones_like(indicator_i)), dim=3)
        # print(f"first_box_mask: shape:{first_box_mask.shape}\n{first_box_mask}")
        # print(f"second_box_mask: shape:{second_box_mask.shape}\n{second_box_mask}")

        indicator_ij = (indicator_i * ((1-best_box) * first_box_mask + best_box * second_box_mask))
        # print(f"indicator_ij: shape:{indicator_ij.shape}\n{indicator_ij}")
        indicator_ij = indicator_ij.unsqueeze(4)
        # print(f"indicator_ij: shape:{indicator_ij.shape}\n{indicator_ij}")

        box_target[..., 0] = torch.cat((best_iou, best_iou), dim=3)
        box_target = indicator_ij * box_target

        # localization loss
        xy_loss = self.lambda_coord * self.mse(
            indicator_ij * box_predictions[..., 1:3],
            indicator_ij * box_target[..., 1:3]
        )
        _logger.debug("%s: localization_xy_loss[%s]", _name, xy_loss)

        wh_loss = self.lambda_coord * self.mse(
            indicator_ij * torch.sign(box_predictions[..., 3:5]) * torch.sqrt(torch.abs(box_predictions[..., 3:5]) + 1e-6),
            indicator_ij * torch.sign(box_target[..., 3:5]) * torch.sqrt(torch.abs(box_target[..., 3:5]) + 1e-6)
        )
        _logger.debug("%s: localization_wh_loss[%s]", _name, wh_loss)

        # object loss
        object_loss = self.mse(
            indicator_ij * box_predictions[..., 0:1],
            indicator_ij * box_target[..., 0:1]
        )
        _logger.debug("%s: confidence_loss[%s]", _name, object_loss)

        # no object loss
        no_object_loss = self.lambda_noobj * self.mse(
            (1-indicator_ij) * box_predictions[..., 0:1],
            (1-indicator_ij) * box_target[..., 0:1]
        )
        _logger.debug(
            "%s: (1-indicator_ij) * box_predictions[..., 0:1] \n%s",
            _name, (1-indicator_ij) * box_predictions[..., 0:1]
        )
        _logger.debug("%s: no_obj_confidence_loss[%s]", _name, no_object_loss)

        return (xy_loss + wh_loss + object_loss + no_object_loss + class_loss) / float(batch_size)


def test_compare_with_torch_01():
    N: int = 1
    S: int = YOLO_GRID_SIZE
    B: int = YOLO_PREDICTION_NUM_BBOX
    C: int = YOLO_PREDICTION_NUM_CLASSES
    P: int = YOLO_PREDICTION_NUM_PRED

    # --------------------------------------------------------------------------------
    # 0/1 only
    # --------------------------------------------------------------------------------
    ones: np.ndarray = np.ones(shape=(1, S, S, C+B*P), dtype=TYPE_FLOAT)
    zeros: np.ndarray = np.zeros(shape=(1, S, S, C+P), dtype=TYPE_FLOAT)

    # Loss from Torch
    torch_loss_instance = TorchYoloLoss()
    loss_from_torch: np.ndarray = torch_loss_instance.forward(
        predictions=torch.Tensor(ones),
        target=torch.Tensor(zeros)
    ).numpy()
    # Loss from TF
    tf_loss_instance = YOLOLoss()
    loss_from_tf: tf.Tensor = tf_loss_instance(
        y_pred=tf.constant(ones),
        y_true=tf.constant(zeros)
    ).numpy()

    assert np.allclose(a=loss_from_torch, b=loss_from_tf, atol=TYPE_FLOAT(1e-5)), \
        f"loss_from_torch:{loss_from_torch} loss_from_tf:{loss_from_tf}"


def test_compare_with_torch_rand():
    N: int = 1
    S: int = YOLO_GRID_SIZE
    B: int = YOLO_PREDICTION_NUM_BBOX
    C: int = YOLO_PREDICTION_NUM_CLASSES
    P: int = YOLO_PREDICTION_NUM_PRED

    # --------------------------------------------------------------------------------
    # randn
    # --------------------------------------------------------------------------------
    pred: np.ndarray = np.random.randn(1, S, S, C+B*P).astype(TYPE_FLOAT)
    true: np.ndarray = np.random.randn(1, S, S, C+P).astype(TYPE_FLOAT)

    y_pred_torch = torch.tensor(pred)
    y_true_torch = torch.tensor(true)
    y_pred_tf: tf.Tensor = tf.constant(pred)
    y_true_tf: tf.Tensor = tf.constant(true)

    # Loss from Torch
    torch_loss_instance = TorchYoloLoss()
    loss_from_torch: np.ndarray = torch_loss_instance.forward(
        predictions=y_pred_torch,
        target=y_true_torch
    ).numpy()

    # Loss from TF
    print("-" * 80)
    print("TF")
    print("-" * 80)
    tf_loss_instance = YOLOLoss()
    loss_from_tf: tf.Tensor = tf_loss_instance(
        y_true=y_true_tf,
        y_pred=y_pred_tf
    ).numpy()

    assert np.allclose(a=loss_from_torch, b=loss_from_tf, atol=TYPE_FLOAT(1e-5)), \
        f"loss_from_torch:{loss_from_torch} loss_from_tf:{loss_from_tf}"


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    test_compare_with_torch_rand()
