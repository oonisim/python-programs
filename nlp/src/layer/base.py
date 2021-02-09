import numpy as np


class Layer:
    @property
    def name(self) -> str:
        """A unique name to identify each layer"""
        return self._name

    def forward(self, X):
        """ Calculate the layer output.
        Note that "forward" is the process of calculating the layer output which
        is to be brought forward as the input to the post layer(s). To actually
        "forward" the  output to the post layer is an implementation matter.

        Args:
            X: Batch input data from the previous layer.
        Returns:
            Y: Layer output
        """
        pass

    def backward(self):
        """Calculate the gradient dL/dX
        Note that "backward" is the process of calculating the gradient(s) which is
        to be brought backward to the previous layer. To actually "backward" the
        gradient to the previous layer is an implementation matter.
        """
        pass

    def __init__(self, name: str):
        self._name: str = name
