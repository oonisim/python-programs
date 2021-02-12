"""Neural network layer base
A layer is a stateful function f that calculates Y=f(X) based on the current state.
For instance, weight matrix W for Matmul layer gets constantly updated with the
gradient descent, hence the layer is not a pure function.

output(X):
    Calculate the layer output Y=f(X).

gradient(dY):
    Calculate the gradient dL/dX = dL/dY * dY/dX (dY is alias of dL/dY).
    The layer does not know the loss function L=g(Y), hence the gradient dL/dY
    needs to be given from the post layer(s).

numerical_gradient(L, h=1e-5):
    Calculate the gradient dL/dX numerically as (L(X+h) - L(X-h)) / 2h with
    the loss function L(X)=g(f(X)). The layer knows f but gives L as g(f)
    because we may know simplified formula for g(f).
"""
from typing import (
    Optional,
    Callable,
    NoReturn
)
import numpy as np


class Layer:
    def __init__(self, name: str, num_nodes: int):
        """
        Args:
            name: layer ID name
            num_nodes: number of nodes in a layer
        """
        assert name
        self._name: str = name

        assert num_nodes > 0
        self._num_nodes = num_nodes

        self._loss: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A unique name to identify each layer"""
        return self._name

    @property
    def num_nodes(self) -> int:
        """Number of nodes in a layer"""
        return self._num_nodes

    @property
    def loss(self) -> Callable[[np.ndarray], np.ndarray]:
        """Loss function L=g(f(X))"""
        assert self._loss, "Loss function L has not been initialized."
        return self._loss

    @loss.setter
    def loss(self, f: Callable[[np.ndarray], np.ndarray]) -> NoReturn:
        self._loss = f

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def output(self, *args) -> np.ndarray:
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

    def forward(self, X: np.ndarray) -> None:
        """Forward the layer output to the post layers
        Args:
            X: layer input
        """
        pass

    def gradient(self, *args):
        """Calculate the gradient dL/dX, the impact on L by the input delta X.
        Note that "backward" is the process of calculating the gradient(s) which is
        to be brought backward to the previous layer. To actually "backward" the
        gradient to the previous layer is an implementation matter.
        """
        pass

    def backward(self) -> np.ndarray:
        """Calculate and back-propagate the gradient dL/dX"""
        pass

    def numerical_gradient(self, h: float = 1e-05):
        """Calculate numerical gradient"""
        pass
