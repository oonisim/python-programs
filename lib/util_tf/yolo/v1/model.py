"""
Implementation of Yolo (v1) architecture with slight modification with added BatchNorm.
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/model.py
"""
# pylint: disable=too-many-statements
import logging
from typing import (
    Tuple,
    List,
)

from tensorflow import keras
from keras import (
    Model,
)
from keras.layers import (
    Layer,
)
from keras.optimizers import (
    SGD
)
from keras.losses import (
    SparseCategoricalCrossentropy
)

from constant import (
    DEBUG_LEVEL,
    TYPE_FLOAT,
    YOLO_V1_IMAGE_WIDTH,
    YOLO_V1_IMAGE_HEIGHT,
    YOLO_V1_IMAGE_CHANNELS,
    YOLO_GRID_SIZE,
    YOLO_V1_PREDICTION_NUM_CLASSES,
    YOLO_V1_PREDICTION_NUM_BBOX,
    YOLO_V1_PREDICTION_NUM_PRED,
    YOLO_V1_LEAKY_RELU_SLOPE,
    YOLO_V1_MOMENTUM,
    YOLO_V1_DECAY,
    YOLO_V1_LR,
)
from util_logging import (
    get_logger,
)
from util_tf.nn import (
    LAYER_NAME_NORM,
    LAYER_NAME_CONV2D,
    LAYER_NAME_ACTIVATION,
    LAYER_NAME_MAXPOOL2D,
    LAYER_NAME_DENSE,
    LAYER_NAME_FLAT,
    LAYER_NAME_BN,
    LAYER_NAME_DROP,
    LAYER_NAME_RESHAPE,
    build_nn_model,
    train,
    get_early_stopping_callback,
    get_tensorboard_callback,
)
from loss import (
    YOLOLoss
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__, level=DEBUG_LEVEL)


# --------------------------------------------------------------------------------
# YOLO v1 model architecture
#
# [YOLO v1 paper]
# We train the network for about 135 epochs on the training and validation data sets
# from PASCAL VOC 2007 and 2012. When testing on 2012 we also include the VOC 2007
# test data for training.
# Throughout training, we use a batch size of 64, a momentum of 0:9 and a decay of 0:0005.
# Our learning rate schedule is as follows: For the first epochs we slowly raise the
# learning rate from 10^-3 (1e-3) to 10^-2 (1e-2). If we start at a high learning rate
# our model often diverges due to unstable gradients. We continue training with 10^-2
# for 75 epochs, then 10^-3 for 30 epochs, and finally 10^4 (1e-4) for 30 epochs.
#
# To avoid overfitting we use dropout and extensive data augmentation. A dropout layer
# with rate = .5 after the first connected layer prevents co-adaptation between layers.
# --------------------------------------------------------------------------------
S: int = YOLO_GRID_SIZE                 # pylint: disable=invalid-name
B: int = YOLO_V1_PREDICTION_NUM_BBOX       # pylint: disable=invalid-name
C: int = YOLO_V1_PREDICTION_NUM_CLASSES    # pylint: disable=invalid-name
P: int = YOLO_V1_PREDICTION_NUM_PRED       # pylint: disable=invalid-name

input_shape: Tuple[int, int, int] = (
    YOLO_V1_IMAGE_WIDTH, YOLO_V1_IMAGE_HEIGHT, YOLO_V1_IMAGE_CHANNELS
)
layers_config = {
    # --------------------------------------------------------------------------------
    # 1st
    # --------------------------------------------------------------------------------
    "norm": {
        "kind": LAYER_NAME_NORM, "input_shape": input_shape, "axis": -1
    },
    "conv01": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (7, 7), "filters": 64, "strides": (2, 2), "padding": "same"
    },
    "act01": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "maxpool01": {
        "kind": LAYER_NAME_MAXPOOL2D, "pool_size": (2, 2), "strides": (2, 2), "padding": "valid"
    },
    # --------------------------------------------------------------------------------
    # 2nd
    # --------------------------------------------------------------------------------
    "conv02": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters": 192, "strides": (1, 1), "padding": "same"
    },
    "act02": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "maxpool02": {
        "kind": LAYER_NAME_MAXPOOL2D, "pool_size": (2, 2), "strides": (2, 2), "padding": "valid"
    },
    # --------------------------------------------------------------------------------
    # 3rd
    # --------------------------------------------------------------------------------
    "conv03_1": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters": 192, "strides": (1, 1), "padding": "same"
    },
    "act03_1": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv03_2": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters": 256, "strides": (1, 1), "padding": "same"
    },
    "act03_2": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv03_3": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters": 256, "strides": (1, 1), "padding": "same"
    },
    "act03_3": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv03_4": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":512, "strides": (1, 1), "padding": "same"
    },
    "act03_4": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "maxpool03": {
        "kind": LAYER_NAME_MAXPOOL2D, "pool_size": (2, 2), "strides": (2, 2), "padding": "valid"
    },
    # --------------------------------------------------------------------------------
    # 4th
    # --------------------------------------------------------------------------------
    # Repeat 1
    "conv04_1_1": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters": 256, "strides": (1, 1), "padding": "same"
    },
    "act04_1_1": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv04_1_2": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters": 512, "strides": (1, 1), "padding": "same"
    },
    "act04_1_2": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    # Repeat 2
    "conv04_2_1": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters": 256, "strides": (1, 1), "padding": "same"
    },
    "act04_2_1": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv04_2_2": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":512, "strides": (1, 1), "padding": "same"
    },
    "act04_2_2": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    # Repeat 3
    "conv04_3_1": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters": 256, "strides": (1, 1), "padding": "same"
    },
    "act04_3_1": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv04_3_2": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":512, "strides": (1, 1), "padding": "same"
    },
    "act04_3_2": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    # Repeat 4
    "conv04_4_1": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters": 256, "strides": (1, 1), "padding": "same"
    },
    "act04_4_1": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv04_4_2": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters": 512, "strides": (1, 1), "padding": "same"
    },
    "act04_4_2": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    # rest
    "conv04_5": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters": 512, "strides": (1, 1), "padding": "same"
    },
    "act04_5": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv04_6": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters": 1024, "strides": (1, 1), "padding": "same"
    },
    "act04_6": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "maxpool04": {
        "kind": LAYER_NAME_MAXPOOL2D, "pool_size": (2, 2), "strides": (2, 2), "padding": "valid"
    },
    # --------------------------------------------------------------------------------
    # 5th
    # --------------------------------------------------------------------------------
    # Repeat 1
    "conv05_1_1": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters":512, "strides": (1, 1), "padding": "same"
    },
    "act05_1_1": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv05_1_2": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":1025, "strides": (1, 1), "padding": "same"
    },
    "act05_1_2": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    # Repeat 2
    "conv05_2_1": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (1, 1), "filters":512, "strides": (1, 1), "padding": "same"
    },
    "act05_2_1": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv05_2_2": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":1024, "strides": (1, 1), "padding": "same"
    },
    "act05_2_2": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    # rest
    "conv05_3": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":1024, "strides": (1, 1), "padding": "same"
    },
    "act05_3": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv05_4": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":1024, "strides": (2, 2), "padding": "same"
    },
    "act05_4": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    # --------------------------------------------------------------------------------
    # 6th
    # --------------------------------------------------------------------------------
    "conv06_1": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":1024, "strides": (1, 1), "padding": "same"
    },
    "act06_1": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    "conv06_2": {
        "kind": LAYER_NAME_CONV2D, "kernel_size": (3, 3), "filters":1024, "strides": (1, 1), "padding": "same"
    },
    "act06_2": {
        "kind": LAYER_NAME_ACTIVATION, "activation": "leaky_relu", "slope": YOLO_V1_LEAKY_RELU_SLOPE
    },
    # --------------------------------------------------------------------------------
    # Fully Connected
    # --------------------------------------------------------------------------------
    "flat": {
        "kind": LAYER_NAME_FLAT, "data_format": "channels_last"
    },
    "full01": {
        "kind": LAYER_NAME_DENSE, "units": 4096, "activation": "relu", "l2": 1e-2
    },
    "drop01": {
        "kind": LAYER_NAME_DROP, "rate": TYPE_FLOAT(0.5),
    },
    # To be able to reshape into (S, S, (C + B * P))
    "full02": {
        "kind": LAYER_NAME_DENSE, "units": (S * S * (C + B * P)), "activation": "relu", "l2": 1e-2
    },
    # --------------------------------------------------------------------------------
    # Rehape into (S, S, (C + B * P))
    # --------------------------------------------------------------------------------
    "reshape": {
        "kind": LAYER_NAME_RESHAPE, "target_shape": (S, S, (C + B * P))
    }
}


# --------------------------------------------------------------------------------
# Model
# TODO: Normalization layer (standardization)
# --------------------------------------------------------------------------------
class YOLOModel(Model):
    def __init__(self, *args, **kwargs):
        super(YOLOModel, self).__init__(*args, **kwargs)
        self.model: Model = build_nn_model(model_name="yolo_v1", input_shape=input_shape, layers_config=layers_config)
        self.optimizer: keras.optimizers.Optimizer = SGD(
            learning_rate=YOLO_V1_LR,
            momentum=YOLO_V1_MOMENTUM,
            decay=YOLO_V1_DECAY
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss=YOLOLoss(),
            metrics=[SparseCategoricalCrossentropy()]
        )
        self.model.summary()

    @property
    def layers(self) -> List[Layer]:
        """Provide model layers as a list"""
        return self.model.layers

    @property
    def input(self):
        return self.model.input

    def get_layer(self, name=None, index=None):
        """Provide Keras Model get_layer I/F"""
        return self.model.get_layer(name=name, index=index)

    def get_model(self) -> Model:
        """Provide the Model instance"""
        return self.model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

