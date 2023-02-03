"""
Implementation of Yolo (v1) architecture with slight modification with added BatchNorm.
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/model.py
"""
from keras.layers import (
    Conv2D,
    ZeroPadding2D,
    UpSampling2D,
    Concatenate,
    MaxPooling2D,
    BatchNormalization,
    LeakyReLU,
)
from keras.models import (
    Model,
)

from constant import (
    YOLO_LEAKY_RELU_SLOPE,
)


class CNNBlock(Model):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(
            filters=out_channels,
            data_format="channels_last",
            padding="same"
        )
        self.batch_norm = BatchNormalization(axis=-1)
        self.leaky_relu = LeakyReLU(alpha=YOLO_LEAKY_RELU_SLOPE)

    def call(self, inputs, training=None, mask=None):
        return self.leaky_relu(self.batch_norm(self.conv(inputs)))
