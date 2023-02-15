"""
CNN utility module
"""
from typing import (
    Tuple,
    Callable,
    Optional,
)

import tensorflow as tf
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
# Utility
# --------------------------------------------------------------------------------
def get_relu_activation_function(
        alpha: float = 0.0,
        max_value: Optional[float] = None,
        threshold: float = 0.0
) -> Callable:
    """
    https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu
    Args:
        alpha: the slope for values lower than the threshold.
        max_value: The largest value the function will return
        threshold:
            value of the activation function below which values
            will be damped or set to zero. Normally 0 (x=0)
    """
    def f(x) -> tf.Tensor:
        return tf.keras.activations.relu(
            x, alpha=alpha, max_value=max_value, threshold=threshold
        )

    return f


# --------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------
class Conv2DBlock(Layer):
    """Keras custom layer to build a block of (Convolution -> BN -> Activation)
    See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer for the Layer signatures.
    See https://keras.io/api/layers/convolution_layers/convolution2d/ for Keras Conv2D layer.

    [References for Keras custom layer]
    Hands on ML2 Ch 12 Custom Layers
    https://www.tensorflow.org/tutorials/customization/custom_layers#implementing_custom_layers
    https://stackoverflow.com/a/58799021/4281353
    """
    def __init__(
            self,
            filters: int,
            kernel_size: Tuple[int, int],
            strides: Tuple[int, int],
            padding: str,
            activation_layer: Layer,
            data_format: Optional[str] = "channels_last",
            **kwargs
    ):
        """Initialization
        Args:
            filters:
                number of edge detection filters in the convolution layer, which is
                number of the output channels
            kernel_size: (height, width) of the 2D convolution window.
            strides: sizes of stride (height, width) along the height and width directions.
            padding: "valid" or "same" (case-insensitive). "valid" means no padding.
            data_format: A string, one of channels_last (default) or channels_first.
            activation_layer: Activation Layer instance
            # input_shape: shape of the input if this is the first layer to take input, otherwise None
        """
        super().__init__(**kwargs)

        self.filters: int = filters
        self.kernel_size: Tuple[int, int] = kernel_size
        self.strides: Tuple[int, int] = strides
        self.padding: str = padding
        self.data_format: str = data_format
        self.batch_norm = BatchNormalization(axis=-1)
        self.activation = activation_layer
        self.conv: Optional[Layer] = None

    def build(self, input_shape):
        """build the layer state
        Args:
            input_shape: TensorShape, or list of instances of TensorShape
        """
        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
            data_format=self.data_format,
            input_shape=input_shape
        )

        # Tell Keras the layer is built
        super().build(input_shape=input_shape)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape=input_shape)

    def get_config(self) -> dict:
        """
        Return serializable layer configuration from which the layer can be reinstantiated.
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_config
        """
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format,
            'batch_norm': self.batch_norm,
            'activation': self.activation,
            'conf': self.conv
        })
        return config

    def call(self, inputs, *args, **kwargs):
        """Layer forward process
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#call
        """
        return self.activation(self.batch_norm(self.conv(inputs)))
