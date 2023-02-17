"""
CNN utility module
"""
import json
import logging
import os.path
from typing import (
    List,
    Dict,
    Tuple,
    Callable,
    Optional,
    Union,
)

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import (
    Layer,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Flatten,
    Activation,
)
from keras.models import (
    Model
)
from keras import (
    regularizers
)
from keras.optimizers import (
    Adam
)
from keras.callbacks import (
    Callback,
    History,
)

from util_constant import (
    TYPE_FLOAT,
)
from util_logging import (
    get_logger,
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)

# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
LAYER_NAME_NORM = "Norm"
LAYER_NAME_CONV2D: str = "Conv2D"
LAYER_NAME_MAXPOOL2D: str = "MaxPool2D"
LAYER_NAME_DENSE: str = "Dense"
LAYER_NAME_FLAT: str = "Flatten"
LAYER_NAME_BN: str = "BatchNorm"
LAYER_ARGV_CHANNELS_LAST: str = "channels_last"
LAYER_ARGV_PADDING: str = "padding"

# --------------------------------------------------------------------------------
# Information
# --------------------------------------------------------------------------------
_logger.info("TensorFlow version: %s", tf.__version__)
_logger.info("Eager execution is: %s", tf.executing_eagerly())
_logger.info("Keras version: %s", tf.keras.__version__)


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
    def func(tensor) -> tf.Tensor:
        return tf.keras.activations.relu(
            tensor, alpha=alpha, max_value=max_value, threshold=threshold
        )

    return func


# --------------------------------------------------------------------------------
# Keras Layer
# --------------------------------------------------------------------------------
class Conv2DBlock(Layer):
    """Keras custom layer to build a block of (Convolution -> BN -> Activation)
    Using Activation as a layer instead of using activation arg of the Keras Conv2D layer.

    See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer for the Layer signatures.
    See https://keras.io/api/layers/convolution_layers/convolution2d/ for Keras Conv2D layer.

    [References for Keras custom layer]
    Hands on ML2 Ch 12 Custom Layers
    https://www.tensorflow.org/tutorials/customization/custom_layers#implementing_custom_layers
    https://stackoverflow.com/a/58799021/4281353
    """
    def __init__(
            self,
            filters: int = 32,
            kernel_size: Tuple[int, int] = (3, 3),
            strides: Tuple[int, int] = (1, 1),
            padding: str = "same",
            activation_layer: Layer = None,
            data_format: Optional[str] = "channels_last",
            **kwargs
    ):
        """Initialization
        Args:
            filters: number of edge detection filters in the convolution layer
            kernel_size: (height, width) of the 2D convolution window.
            strides: sizes of stride (height, width) along the height and width directions.
            padding: "valid" or "same" (case-insensitive). "valid" means no padding.
            data_format: A string, one of channels_last (default) or channels_first.
            activation_layer: Activation layer instance to use for the output of Conv2D

            # input_shape is to be passed as part of kwargs as one of the standard Keras layer arg.
            # input_shape: shape of the input if this is the first layer to take input, otherwise None
        """
        super().__init__(**kwargs)

        self.filters: int = filters
        self.kernel_size: Tuple[int, int] = kernel_size
        self.strides: Tuple[int, int] = strides
        self.padding: str = padding
        self.data_format: str = data_format
        self.batch_norm = BatchNormalization(axis=-1)
        self.conv: Optional[Layer] = None

        # --------------------------------------------------------------------------------
        # If Activation layer is not provisioned, instantiate internally.
        # --------------------------------------------------------------------------------
        if activation_layer is None:
            activation_function: Callable = get_relu_activation_function(
                alpha=0.1, max_value=None, threshold=0.0
            )
            self.activation = Activation(activation=activation_function)
        else:
            self.activation = activation_layer

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

    def call(self, inputs, *args, **kwargs):    # pylint: disable=unused-argument
        """Layer forward process
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#call
        """
        # return self.activation(self.batch_norm(self.conv(inputs)))
        return self.batch_norm(self.activation(self.conv(inputs)))


def build_layers(config: Dict[str, dict]) -> List[Layer]:
    """Build layers based on the configuration dictionary of layers.
    Args:
        config: layer configuration dictionary
    Returns: List of layer instances generated

    The order of the layers is as per the order in the dictionary.

    [Example]
    example_config_of_layers = {
        "conv01": {  # Name of the layer. "conv01" is set to name arg of the Layer
            "kind": LAYER_NAME_CONV2D,     # Layer class Conv2D
            "kernel_size":(3,3),           # Conv2D argument
            "filters":64,
            "strides":1,
            "padding": "same"
        },
        "maxpool01": {
            "kind": LAYER_NAME_MAXPOOL2D, "pool_size":(2,2), "strides":(2,2), "padding": "valid"
        },
        "conv02": {
            "kind": LAYER_NAME_CONV2D, "kernel_size":(3,3), "filters":64, "strides":1, "padding": "same"
        },
        "flat": {
            "kind": LAYER_NAME_FLAT, "data_format": LAYER_ARG_CHANNELS_LAST
        },
        "full": {
            "kind": LAYER_NAME_DENSE,
            "units": 100,
            "activation": "relu",
            "l2": 1e-2                    # L2 kernel regularizer lambda value
        },
        "bn": {
            "kind": LAYER_NAME_BN
        },
        "label": {
            "kind": LAYER_NAME_DENSE, "units": 10, "activation": "softmax"
        }
    }
    """
    _name: str = "build_layers()"
    layers: List[Layer] = []
    sequence = (i for i in range(len(config)))

    for name, value in config.items():
        index = next(sequence)
        kind = value["kind"]
        _logger.debug("%s: creating layer[%s] with %s", _name, index, value)

        # --------------------------------------------------------------------------------
        # Convolution
        # --------------------------------------------------------------------------------
        if kind == LAYER_NAME_CONV2D:
            conv: Layer = Conv2DBlock(
                name=name,
                filters=value.get("filters", 32),
                kernel_size=value.get("kernel_size", 2),
                strides=value.get("strides", 1),
                padding=value.get("padding", "same"),
                data_format=value.get("data_format", "channels_last")
                # activation="relu"
            )
            layers.append(conv)

        # --------------------------------------------------------------------------------
        # Max Pooling
        # --------------------------------------------------------------------------------
        elif kind == LAYER_NAME_MAXPOOL2D:
            pool: Layer = MaxPooling2D(
                name=name,
                pool_size=value.get("pool_size", 2),
                strides=value.get("strides", 1),
                padding=value.get("padding", "same"),
                data_format=value.get("data_format", "channels_last")
            )
            layers.append(pool)

        # --------------------------------------------------------------------------------
        # Batch Norm
        # --------------------------------------------------------------------------------
        elif kind == LAYER_NAME_BN:
            norm: Layer = BatchNormalization(
                name=name,
            )
            layers.append(norm)

        # --------------------------------------------------------------------------------
        # Fatten
        # --------------------------------------------------------------------------------
        elif kind == LAYER_NAME_FLAT:
            flat: Layer = Flatten(
                name=name,
                data_format=value.get("data_format", "channels_last")
            )
            layers.append(flat)

        # --------------------------------------------------------------------------------
        # Fully Connected
        # --------------------------------------------------------------------------------
        elif kind == LAYER_NAME_DENSE:
            full: Layer = Dense(
                name=name,
                units=value.get("units", 100),
                # L2 kernel regularizer
                # https://stats.stackexchange.com/a/383326/105137
                # https://keras.io/api/layers/regularizers/#l2-class
                kernel_regularizer=regularizers.l2(l2=value.get("l2", 1e-2)),
                activation=value.get("activation", "relu")
            )
            layers.append(full)
        # --------------------------------------------------------------------------------
        # Invalid
        # --------------------------------------------------------------------------------
        else:
            msg = f"invalid layer name [{kind}]"
            _logger.error("%s: layer[%s] failed due to %s.", _name, index, msg)
            _logger.error("%s: config=\n%s", _name, json.dumps(config, indent=4))
            raise RuntimeError(msg)

    return layers


# --------------------------------------------------------------------------------
# Keras Model
# --------------------------------------------------------------------------------
def build_nn_model(
        model_name: str,
        input_shape,
        layers_config: dict,
        normalization: Optional[Layer] = None,
        learning_rate: TYPE_FLOAT = TYPE_FLOAT(1e-3),
) -> Model:
    """
    Args:
        model_name: name of the model
        input_shape: shape of the input
        normalization: tf.Keras.layers.Normalization if it has been used for normalizing
        learning_rate: learning rate
        layers_config: dictionary of layer configurations

    Returns: Model
    """
    # --------------------------------------------------------------------------------
    # Input Layers
    # --------------------------------------------------------------------------------
    inputs = tf.keras.Input(
        shape=input_shape,
        batch_size=None,
        name="inputs",
        dtype=TYPE_FLOAT,
        sparse=False
    )
    tensor: tf.Tensor = inputs

    # --------------------------------------------------------------------------------
    # Normalization Layer
    # --------------------------------------------------------------------------------
    if normalization is not None:
        assert isinstance(normalization, Layer)
        tensor = normalization(tensor)

    # --------------------------------------------------------------------------------
    # CNN Layers
    # --------------------------------------------------------------------------------
    layers: List[Layer] = build_layers(layers_config)
    for layer in layers:
        tensor = layer(tensor)

    outputs: tf.Tensor = tensor

    # --------------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------------
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    model.summary()

    return model


def train(
        model: Model,
        x: Union[np.ndarray, tf.Tensor],
        y: Union[np.ndarray, tf.Tensor],
        batch_size: int = 32,
        epochs: int = 1,
        validation_split: float = 0.2,
        use_multiprocessing: bool = True,
        workers: int = 4,
        verbosity: int = 0,
        callbacks: List[Callback] = None
) -> History:
    """Model training runner
    https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

    Args:
        model: Keras Model to train
        x: input data
        y: label data
        batch_size: batch size
        epochs: number of epochs to run the training
        validation_split: ratio of x data to allocate for validation during training
        use_multiprocessing: flat for multiprocessing
        workers: number of workers to use for multiprocessing
        verbosity: verbosity level 'auto', 0, 1, or 2.
        callbacks: List of keras.callbacks.Callback instances.

    Returns:
        History (see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
        History.history is a record of training loss values and metrics values at epochs
    """
    history = model.fit(
        x,
        y,
        shuffle=True,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        verbose=verbosity,
        callbacks=callbacks
    )
    return history


# --------------------------------------------------------------------------------
# Keras Callbacks
# --------------------------------------------------------------------------------
def get_tensorboard_callback(
    log_dir: str,
    write_graph: bool = True,
    write_images: bool = True,
    histogram_freq: int = 1,
    embeddings_freq: int = 1,
) -> Callback:
    """
    https://keras.io/api/callbacks/tensorboard/

    Args:
        log_dir: directory to write logs
        write_graph: whether to visualize the graph in TensorBoard
        write_images: whether to write model weights
        histogram_freq:  frequency in epochs to compute weight histograms for the layers. 0 for not to compute.
        embeddings_freq: frequency in epochs at which embedding layers will be visualized
        # update_freq: disabled. Do not use
    Returns: Callback
    """
    callback: Callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=write_graph,
        write_images=write_images,
        histogram_freq=histogram_freq,
        embeddings_freq=embeddings_freq,
    )
    return callback


def get_early_stopping_callback(
        monitor='val_loss',
        mode: str = "min",
        min_delta: int = 0,
        patience: int = 5,
        verbose: int = 1,
        start_from_epoch: int = 0,
        restore_best_weights: bool = True
) -> Callback:
    """Get Early Stopping callback instance
    By default, monitor the validation loss to decrease (mode=min) as it quantifies
    the deviation from the grand truth and indicates over-fitting if it starts to
    increase.

    https://keras.io/api/callbacks/early_stopping/
    Args:
        monitor: metrics to monitor e.g. val_loss (stackoverflow.com/questions/75479364/)
        mode: {"auto", "min", "max"}. min: early stop when metric stopped decreasing
        min_delta: Minimum change in metric to qualify as an improvement,
        patience:  Number of epochs with no improvement after which to early stop
        verbose: 0 or 1. 0 is silent, 1 displays messages when the callback takes an action.
        restore_best_weights: restore weights from the epoch with the best metrics value
    Returns: Callback
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        restore_best_weights=restore_best_weights,
        mode=mode,
        verbose=verbose
    )


# --------------------------------------------------------------------------------
# Model Evaluation
# --------------------------------------------------------------------------------
def _validate_precisions_labels(
        predictions: Union[np.ndarray, tf.Tensor],
        labels: Union[List, np.ndarray, tf.Tensor],
):
    """Validate if predictions and labels.

    model.predict(x) provides the probabilities of each label per prediction.
    If num_classes=3, model.predict(x) for single x gives e.g [0.112, 0.873, 0.005]
    for each class. For N number of data in X, model.predict(X) gives (N, num_classes).
    The argument predictions is expected to have the shape (N, num_classes).

    Args:
        predictions: predictions from which to generate the confusion matrix.
        labels: ground truth

    Raises: AssertionError
    """
    assert isinstance(predictions, (np.ndarray, tf.Tensor)), \
        f"expected np.ndarray or tf.Tensor, got {type(type(predictions))}"
    assert predictions.ndim == 2 and predictions.shape[0] == labels.shape[0], \
        f"number of elements in prediction[{len(predictions)}] " \
        f"does not match that in labels[{len(labels)}]."


def get_confusion_matrix(
        predictions: Union[np.ndarray, tf.Tensor],
        labels: Union[List, np.ndarray, tf.Tensor],
        num_classes: int
) -> tf.Tensor:
    """Generate a confusion matrix from predictions.
    See https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    Args:
        predictions: predictions from which to generate the confusion matrix.
        labels: ground truth
        num_classes: number of classes

    Return: Confusion Matrix
    """
    _validate_precisions_labels(predictions=predictions, labels=labels)
    assert predictions.shape[1] == num_classes, \
        f"expected shape (N, {num_classes}), got {predictions.shape}"

    confusion_matrix: tf.Tensor = tf.math.confusion_matrix(
        labels=labels,
        predictions=tf.math.argmax(predictions, axis=-1),
        num_classes=num_classes
    )
    return confusion_matrix


def get_precision(
        predictions: Union[np.ndarray, tf.Tensor],
        labels: Union[List, np.ndarray, tf.Tensor],
) -> float:
    """Calculate precision = TP/(TP+FP)
    Args:
        predictions: predictions from which to generate the confusion matrix.
        labels: ground truth
    Returns: precision as float
    """
    _validate_precisions_labels(predictions=predictions, labels=labels)

    precision: tf.keras.metrics.Metric = tf.keras.metrics.Precision()
    precision.update_state(y_true=labels, y_pred=tf.math.argmax(predictions, axis=-1))
    return float(precision.result().numpy())


def get_recall(
        predictions: Union[np.ndarray, tf.Tensor],
        labels: Union[List, np.ndarray, tf.Tensor],
) -> float:
    """Calculate recall = TP/(TP+FN)
    Args:
        predictions: predictions from which to generate the confusion matrix.
        labels: ground truth
    Returns: recall as float
    """
    _validate_precisions_labels(predictions=predictions, labels=labels)

    recall: tf.keras.metrics.Metric = tf.keras.metrics.Recall()
    recall.update_state(y_true=labels, y_pred=tf.math.argmax(predictions, axis=-1))
    return float(recall.result().numpy())


def get_accuracy(
        predictions: Union[np.ndarray, tf.Tensor],
        labels: Union[List, np.ndarray, tf.Tensor],
) -> float:
    """Calculate accuracy = (TP+TN)/(TP+FP+TN+FN)
    Args:
        predictions: predictions from which to generate the confusion matrix.
        labels: ground truth
    Returns: accuracy as float
    """
    _validate_precisions_labels(predictions=predictions, labels=labels)

    accuracy: tf.keras.metrics.Metric = tf.keras.metrics.Accuracy()
    accuracy.update_state(y_true=labels, y_pred=tf.math.argmax(predictions, axis=-1))
    return float(accuracy.result().numpy())


def get_auc(
        predictions: Union[np.ndarray, tf.Tensor],
        labels: Union[List, np.ndarray, tf.Tensor],
) -> float:
    """Calculate ROC/AUC
    https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC
    Args:
        predictions: predictions from which to generate the confusion matrix.
        labels: ground truth
    Returns: ROC/AUC as float
    """
    _validate_precisions_labels(predictions=predictions, labels=labels)

    auc: tf.keras.metrics.Metric = tf.keras.metrics.Accuracy()
    auc.update_state(y_true=labels, y_pred=tf.math.argmax(predictions, axis=-1))
    return float(auc.result().numpy())
