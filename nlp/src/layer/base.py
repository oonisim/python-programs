"""Base of neural network (NN) layer
A layer is a process (f, g, gn [, u, state]) that has three interfaces f, g, gn.
Optionally it can have a state and an interface u to update the state at cycle t.

[Restriction]
To avoid DAG and simplify this specific NN implementation, NN is sequential.
That is, each layer has only one layer as its successor. The layer i output fi
is the input to the fi+1 of the next layer i+1.

When there are n layers (i:0,1,...n-1) in NN, the objective loss function L is
L = (fn-1 o ... fi o ... f0) as a sequential composition of functions fi where
i: 0, ... n-1. A function composition g o f = g(f(arg))

[Considerations]
Loss function L:
    Each layer i has its loss function Li = (fn-1 o ... fi+1) with i < n-1 that
    calculates L as Li(Zi) where Zi = fi(arg).

    L = Li(fi) i < n-1   (last layer Ln-1 does not have Ln)

    Note that the Li composition does NOT include fi as there can be multiple
    args to calculate fi e.g. X and W for Matmul and we need numerical gradients
    as Li( (X+/-h)@W.T ) and Li( X@(W+/-h) ) respectively for X and W.

    Li may be written as L when a specific layer 'i' is difficult to identify.
    Be clear if L is the NN loss function L or a layer loss function Li.

    ```
    Li = reduce(lambda f,g: lambda x: g(f(x), [ for layer.f in layers[i+1:n] ])
    or
    # compose apply the last element first.
    from compose import compose
    Li = compose(*[ for layer.f in layers[::-1][i+1:n] ]) with 0 =< i < n=1
    ```

[Design]
f(arg):
    f is a function that calculates the layer output Z=f(arg) where arg is the
    output of the previous layer.

g(dL/dZ):
    g is a function that calculates a gradient dL/dX = g(dL/dZ) = dL/dZ * dZ/dX
    and back-propagate it to the previous layers. dL/dZ is given because the
    layer i does not know fi at layers i+1, ... n-1 hence cannot calculate it.

gn(Li, h):
    gn is a function that calculate numerical gradients, given L and h, as
    gn=( L(arg+h) - L(arg-h) ) / 2h

    arg: any argument that computes the output of the layer Z.
         Although fi can take a single arg, the actual computation may need
         multiple arguments e.g. X and W for matmul. Then gn calculates for
         X and W, and returns two numerical gradients.
    Li: the loss function of the layer Li=(fn-1 o ... fi+1) with i < n-1
        loss=Li(Z)=L(f(arg)), NOT L(arg).
    h: a small number e.g. 1e-5

state S:
    A state in a layer is a combination of variables at a specific cycle t.

u(dL/dZ):
    u is a function that calculates dL/dS = u(dL/dZ) = dL/dZ * dZ/dS to update
    the state S in a layer with gradient descent as S=optimizer(S, dL/dS).

[Example]
    Matmul at cycle t with a input X(t), its state is [ W(t) ], NOT [ X(t), W(t)].
    X(t) is a constant for the layer and the layer will not update it in itself.
    Such a constant is pre-set to a layer before calculations at each cycle.

    f calculates the output Z = f(X). g calculates the gradient dL/dX = g(dL/dZ).

    u calculate the gradient dL/dW=u(dL/dZ) and updates W as W=optimizer(W,dL/dW).
"""
from typing import (
    Optional,
    Union,
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

        self._loss: Union[Callable[[np.ndarray], np.ndarray], None] = None

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
        """Loss function L=fn-1 o fn-2 o ... o fi"""
        assert self._loss, "Loss function L has not been initialized."
        return self._loss

    @loss.setter
    def loss(self, Li: Callable[[np.ndarray], np.ndarray]) -> NoReturn:
        assert Li
        self._loss = Li

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: np.ndarray) -> np.ndarray:
        """ Calculate the layer output f(arg).
        Note that "output" is the process of calculating the layer output which
        is to be brought forward as the input to a post layer.

        Args:
            X: input to the layer
        Returns:
            Layer output
        """
        pass

    def forward(self) -> NoReturn:
        """Forward the layer output to the post layers
        Args: None
        Returns: None
        """
        pass

    def gradient(self, dZ: np.ndarray):
        """Calculate the gradient dL/dX=g(dL/dZ), the impact on L by the input dX.
        Note that "gradient" is the process of calculating the gradient to back
        propagate to the previous layer.

        Args:
            dZ: dL/dZ, impact on L by the layer output Z
        Returns:
            dL/dX, impact on L by the layer input X
        """
        pass

    def backward(self) -> np.ndarray:
        """Calculate and back-propagate the gradient dL/dX"""
        pass

    def gradient_numerical(
            self, Li: Callable[[np.ndarray], np.ndarray], h: float = 1e-05
    ) -> np.ndarray:
        """Calculate numerical gradient
        Args:
            Li: Loss fouction
        """
        pass

    def update(self, dZ: np.ndarray):
        """Calculate the gradient dL/dS and update S
        Args:
            dZ: dL/dZ
        Returns:
            dL/dS
        """
        pass
