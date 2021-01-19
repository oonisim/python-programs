import numpy as np


class Layer:
    @property
    def name(self) -> str:
        """Layer ID"""
        return self._name

    def forward(self, X):
        """Forward propagate the affine layer output Y = X@W
        Args:
            X: Batch input data from the input layer.
        Returns:
            Normalized loss of those returned from the post layers.
        """
        pass

    def backward(self):
        pass

    def __init__(self, name: str):
        self._name: str = name
