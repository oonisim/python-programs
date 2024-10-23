"""
Module to log to Tensorboard
"""
from typing import (
    List,
    Optional,
    Union
)
from itertools import (
    product
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix
)
import torch
from torch import nn
from torch.utils.tensorboard import (
    SummaryWriter,
)


def write_histograms_conv2d(
        writer: SummaryWriter,
        layer: nn.Conv2d,
        layer_number: int,
        step: int,
) -> None:
    """Write Conv2D layer weight and bias (when Conv2D bias=True) histgram.
    Args:
        writer: TensorBoard summary writer
        layer: Conv2D module
        layer_number:
        step: Step number in the execution cycle.
    """
    assert isinstance(layer, nn.Conv2d), f"expected Conv2D, got {type(layer)}."

    # weights of all the kernels (filters) in the Conv2D layer.
    num_kernels: int = layer.weight.shape[0]
    weights: torch.Tensor = layer.weight
    biases: Optional[torch.Tensor]
    if layer.bias is not None:
        biases: torch.Tensor = layer.bias
        assert biases.shape[0] == num_kernels, \
            f"expected [{num_kernels}] number of biases, got [{biases.shape[0]}]."
    else:
        biases = None

    for _k in range(num_kernels):
        weight: torch.Tensor = weights[_k].flatten()
        writer.add_histogram(
            tag=f"layer_{layer_number}/Conv2D_kernel_{_k}_weight",
            values=weight, global_step=step, bins='tensorflow'
        )
        if biases is not None:
            bias: torch.Tensor = biases[_k].flatten()
            tag = f"layer_{layer_number}/Conv2D_kernel_{_k}_bias"
            writer.add_histogram(tag=tag, values=bias, global_step=step, bins='tensorflow')


def write_histograms_linear(
        writer: SummaryWriter,
        layer: nn.Linear,
        layer_number: int,
        step: int,
) -> None:
    """Write Linear layer weight and bias (when Conv2D bias=True) histgram.
    Args:
        writer: TensorBoard summary writer
        layer: Linear module
        layer_number:
        step: Step number in the execution cycle.
    """
    assert isinstance(layer, nn.Linear), f"expected Linear, got {type(layer)}."
    writer.add_scalar(
        tag=f"layer_{layer_number}/Linear_weight_variance",
        scalar_value=torch.var(input=layer.weight), global_step=step, new_style=True
    )
    writer.add_histogram(
        tag=f"layer_{layer_number}/Linear_weight",
        values=layer.weight.flatten(), global_step=step, bins='tensorflow'
    )

    if layer.bias is not None:
        bias: torch.Tensor = layer.bias.flatten()
        writer.add_histogram(
            tag=f"layer_{layer_number}/Linear_bias",
            values=bias, global_step=step, bins='tensorflow'
        )


def tensorboard_write_histogram(
        writer: SummaryWriter,
        model: nn.Module,
        step: int,
) -> None:
    """Write model parameter histgram to Tensorboard.
    Args:
        model:
        writer: Tensorboard Summary Writer
        step: Step number in the execution cycle.
    """
    assert isinstance(model, nn.Module) and getattr(model, "layers", None), \
        f"expected a model with layers, got {type(model)}."

    for layer_number in range(len(model.layers)):
        layer: nn.Module = model.layers[layer_number]
        if isinstance(layer, nn.Linear):
            write_histograms_linear(writer=writer, layer=layer, step=step, layer_number=layer_number)
        elif isinstance(layer, nn.Conv2d):
            write_histograms_conv2d(writer=writer, layer=layer, step=step, layer_number=layer_number)


def tensorboard_write_graph(
        writer: SummaryWriter,
        model: nn.Module,
        x: torch.Tensor
) -> None:
    """Write model graph to Tensorboard.
    Args:
        model:
        writer: Tensorboard Summary Writer
        x: input to the model
    """
    assert isinstance(model, nn.Module) and getattr(model, "layers", None), \
        f"expected a model with layers, got {type(model)}."
    writer.add_graph(model, input_to_model=x, verbose=False)


def tensorboard_write_scalar(
        writer: SummaryWriter,
        tag: str,
        value: Union[float, str],
        step: int,
        walltime: Optional[float] = None
) -> None:
    """Write metric value to Tensorboard.
    Args:
        writer: Tensorboard Summary Writer
        tag: identifier of the scalar data in the board
        value: scalar value to write
        step: Step number in the execution cycle.
        walltime: Optional override default walltime (time.time()) with seconds after epoch
    """
    writer.add_scalar(
        tag=tag, scalar_value=value, global_step=step, new_style=True, walltime=walltime
    )


def tensorboard_write_image(
    writer: SummaryWriter,
    tag: str,
    image: Union[np.ndarray, torch.Tensor],
    step: Optional[int],
    dataformats: str = 'CWH',
    walltime: Optional[float] = None
) -> None:
    """Write image to Tensorboard.
    Args:
        writer: Tensorboard Summary Writer
        image: image of shape (c, h, w) or (h, w, c) or (h, w) or (w, h) defaults to (3,h,w)
               where c is channels, w is width, h is height of the image.
               The shape must match with the dataformats string.
        tag: identifier of the scalar data in the board
        step: Step number in the execution cycle.
        walltime: Optional override default walltime (time.time()) with seconds after epoch
        dataformats: image data format specification e.g, CHW, HWC, HW, WH, etc
    """
    writer.add_image(
        tag=tag, img_tensor=image, global_step=step, walltime=walltime, dataformats=dataformats
    )


def plot_confusion_matrix(
        matrix: np.ndarray,
        class_names: Union[List[str], np.ndarray]
):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      matrix (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(
        matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis],
        decimals=2
    )

    # Use white text if squares are dark; otherwise black.
    threshold = matrix.max() / 2.
    for i, j in product(range(matrix.shape[0]), range(matrix.shape[1])):
        color = "white" if matrix[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def tensorboard_write_confusion_matrix(
        writer: SummaryWriter,
        tag: str,
        predictions: torch.Tensor,
        truth: torch.Tensor,
        class_names: Union[List[str], np.ndarray],
        step: Optional[int] = None,
        walltime: Optional[float] = None
) -> None:
    """Write confusion matrix as an image.
    Args:

    """
    matrix = confusion_matrix(y_true=truth, y_pred=predictions)

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(matrix, class_names=class_names)

    # Log the confusion matrix as an image summary.
    writer.add_figure(
        tag=tag,
        figure=figure,
        global_step=step,
        close=True,
        walltime=walltime
    )
