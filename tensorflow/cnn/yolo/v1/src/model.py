"""
Implementation of Yolo (v1) architecture with slight modification with added BatchNorm.
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/model.py
"""
from typing import (
    Tuple,
)
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


class YOLOCNNBlock(Model):
    """Building block of (Convolution -> BN -> Activation)
    """
    def __init__(
            self,
            kernel_size: Tuple[int, int],
            strides: Tuple[int, int],
            padding: str,
            num_output_channels: int,
            **kwargs
    ):
        """Initialization
        Args:
            kernel_size: (height, width) of the 2D convolution window.
            strides: sizes of stride (height, width) along the height and width directions.
            padding: "valid" or "same" (case-insensitive). "valid" means no padding.
            num_output_channels:
                number of edge detection filters in the convolution layer, which is
                number of the output channels
        """
        super().__init__(**kwargs)
        self.conv = Conv2D(
            filters=num_output_channels,
            data_format="channels_last",
            padding="same"
        )
        self.batch_norm = BatchNormalization(axis=-1)
        self.activation = LeakyReLU(alpha=YOLO_LEAKY_RELU_SLOPE)

    def call(self, inputs, training=None, mask=None):
        return self.activation(self.batch_norm(self.conv(inputs)))
