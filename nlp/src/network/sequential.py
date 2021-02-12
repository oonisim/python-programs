"""Sequential neural network"""
from typing import (
    List,
    Dict,
    Final,
    Union
)
import compose
import numpy as np
from base import Base


class Sequential(Base):
    def __init__(self, name: str, layers: list):
        """
        Args:
            name: Network ID name
            layers:
                Sequential layers for the prediction, hence excluding the loss layer.
                Ordered from the input to the prediction at the end.
        """
        super().__init__(name)
        self._layers = layers
        self._M = self._num_classes = layers[-1].

    @property
    def layers(self) -> List:
        """Neural network layers"""
        return self._layers
