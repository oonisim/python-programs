"""Base for neural network implementations"""
from typing import (
    List,
    Dict,
    Final,
    Union
)
import compose
import numpy as np


class Base:
    """Base class for neural network classes"""
    def __init__(self, name: str):
        """
        Args:
            name: Network ID name
        """
        self._name = name
        self._loss: float = -np.inf

    @property
    def name(self) -> str:
        """Network ID name"""
        return self._name

    @property
    def loss(self) -> float:
        """Network loss value"""
        return self._loss

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict(self, X: np.ndarray) -> Union[List[float], np.ndarray]:
        pass

