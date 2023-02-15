"""
CNN utility module
"""
from typing import (
    Tuple,
)
from keras.layers import (
    Layer,
    Conv2D,
    ZeroPadding2D,
    UpSampling2D,
    Concatenate,
    MaxPooling2D,
    BatchNormalization,
    Activation,
)
from keras.activations import (
    relu
)
from keras.models import (
    Model,
)


# --------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------
class CNNBlock(Model):
    """Building block of (Convolution -> BN -> Activation)
    """
    def __init__(
            self,
            kernel_size: Tuple[int, int],
            strides: Tuple[int, int],
            padding: str,
            activation_layer: Layer,
            num_output_channels: int,
            **kwargs
    ):
        """Initialization
        Args:
            kernel_size: (height, width) of the 2D convolution window.
            strides: sizes of stride (height, width) along the height and width directions.
            padding: "valid" or "same" (case-insensitive). "valid" means no padding.
            activation_layer: Activation Layer instance
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
        self.activation = activation_layer

    def call(self, inputs, training=None, mask=None):
        return self.activation(self.batch_norm(self.conv(inputs)))
