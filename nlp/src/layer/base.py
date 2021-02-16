"""Base of neural network (NN) layer
A layer is a process (f, g, gn [, u, state]) that has three interfaces f, g, gn.
Optionally it can have a state and an interface u to update the state at cycle t.

[Restriction]
To avoid DAG and simplify this specific NN implementation, NN is sequential.
That is, each layer has only one layer as its successor. The layer i output fi
is the input to the fi+1 of the next layer i+1.

When there are n layers (i:0,1,...n-1) in NN, the objective objective function
L is L = (fn-1 o ... fi o ... f0) as a sequential composition of functions fi
where i: 0, ... n-1. A function composition g o f = g(f(arg))

[Considerations]
Objective function L:
    Each layer i has its objective function Li = (fn-1 o ... fi+1) with i < n-1
    that calculates L as Li(Yi) from the layer output Yi=fi(arg).

    L = Li(fi) i < n-1   (last layer Ln-1 does not have Ln)

    Note that the Li composition does NOT include fi as there can be multiple
    args to calculate fi e.g. X and W for Matmul and we need numerical gradients
    as Li( (X+/-h)@W.T ) and Li( X@(W+/-h) ) respectively for X and W.

    Li may be written as L when a specific layer 'i' is difficult to identify.
    Be clear if L is the NN objective function L or a layer objective function Li.

    ```
    Li = reduce(lambda f,g: lambda x: g(f(x), [ for layer.f in layers[i+1:n] ])
    or
    # compose apply the last element first.
    from compose import compose
    Li = compose(*[ for layer.f in layers[::-1][i+1:n] ]) with 0 =< i < n=1
    ```

[Design]
f(arg):
    f is a function that calculates the layer output Y=f(arg) where arg is the
    output of the previous layer.

g(dL/dY):
    g is a function that calculates a gradient dL/dX = g(dL/dY) = dL/dY * dY/dX
    and back-propagate it to the previous layers. dL/dY is given because the
    layer i does not know f at layers i+1, ... n-1 hence cannot calculate it.

gn(Li, h):
    gn is a function that calculate numerical gradients, given L and h, as
    gn=[ Li(fi(arg+h)) = Li(fi(arg-h)) ] / 2h

    arg: any argument that computes the output Y of the layer.
         Although fi can take a single arg, the actual computation may need
         multiple arguments e.g. X and W for matmul. Then gn calculates for
         X and W, and returns two numerical gradients.
    Li: the objective function of the layer Li=(fn-1 o ... fi+1) with i < n-1
        objective=Li(Y)=L(f(arg)), NOT L(arg).
    h: a small number e.g. 1e-5

state S:
    A state in a layer is a combination of variables at a specific cycle t.

u(dL/dY):
    u is a function that calculates dL/dS = u(dL/dY) = dL/dY * dY/dS and updates
    the state S in the layer with gradient descent as S=optimizer(S, dL/dS).
    There may be multiple dS to calculate and u returns all dL/dS.

    Why not calculating all gradients in g? -> Separation of concerns.
        f focuses on X, g focuses on its gradient dL/dX, u focuses on dL/dS.
        It can be efficient to calculate all in g, but loses transparency (
        clear declaration of intentions) and ambiguates what it focuses on.

[Example]
    Matmul at cycle t with a input X(t), its state is [ W(t) ], NOT [ X(t), W(t)].
    X(t) is a constant for the layer and the layer will not update it in itself.
    Such constants can be pre-set to each layer before calculations at each cycle.
    L is regarded as L(W, X, T) where X, T are constants during a cycle t.

    f calculates the output Y = f(X). g calculates the gradient dL/dX = g(dL/dY).
    u calculate the gradient dL/dW=u(dL/dY) and updates W as W=optimizer(W,dL/dW).

[Python]
Python relative import is defective. Use sys.path + absolute import ONLY.
Or spend hours on "attempted relative import beyond top-level package"
"""
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Callable,
    NoReturn,
    Final
)
import logging
import numpy as np
from common.functions import (
    numerical_gradient
)


class Layer:
    """Neural network layer base class"""
    # ================================================================================
    # Class initialization
    # ================================================================================

    # ================================================================================
    # Instance initialization
    # ================================================================================
    def __init__(self, name: str, num_nodes: int, log_level: int = logging.ERROR):
        """
        Args:
            name: layer ID name
            num_nodes: number of nodes in a layer
        """
        assert name
        self._name: str = name

        # number of nodes in the layer
        assert num_nodes > 0
        self._num_nodes = num_nodes
        self._M: int = num_nodes

        # X: batch input of shape(N, D)
        self._X: np.ndarray = np.empty((0, num_nodes), dtype=float)
        self._N: int = -1

        # Labels of shape(N, M) for OHE or (N,) for index.
        self._T: np.ndarray = np.empty((), dtype=int)

        # Objective Li function for the layer. L = Li o fi
        self._objective: Union[Callable[[np.ndarray], np.ndarray], None] = None

        self._logger = logging.getLogger(name)
        self._logger.setLevel(log_level)

    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """A unique name to identify a layer"""
        return self._name

    @property
    def num_nodes(self) -> int:
        """Number of nodes in a layer"""
        return self._num_nodes

    @property
    def X(self) -> np.ndarray:
        """Latest batch input to the layer"""
        assert self._X and self._X.size, "X is not initialized"
        return self._X

    @X.setter
    def X(self, X: np.ndarray):
        assert X
        self._X = X
        self._N = X.shape[0]

    @property
    def N(self) -> int:
        """Batch size"""
        assert self._N > 0, "N is not initialized"
        return self._N

    @property
    def T(self) -> np.ndarray:
        """Label in OHE or index format"""
        assert self._T and self.T.size > 0, "T is not initialized"
        return self._T

    @T.setter
    def T(self, T: np.ndarray):
        assert T and T.shape[0] == self.N, \
            f"The batch size of T should be {self.N} but {T.shape[0]}"
        self._T = T.astype(int)

    @property
    def objective(self) -> Callable[[np.ndarray], np.ndarray]:
        """Objective function L=fn-1 o fn-2 o ... o fi"""
        assert self._objective, "Objective function L has not been initialized."
        return self._objective

    @objective.setter
    def objective(self, Li: Callable[[np.ndarray], np.ndarray]) -> NoReturn:
        assert Li
        self._objective = Li

    @property
    def logger(self) -> logging.Logger:
        """Instance logger"""
        assert self._logger, "logger is not initialized"
        return self._logger

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: np.ndarray) -> np.ndarray:
        """Calculate the output f(arg) to forward as the input to the post layer.
        Args:
            X: input to the layer
        Returns:
            Y: Layer output
        """

    def forward(self, X: np.ndarray) -> NoReturn:
        """Forward the layer output to the post layers
        Args:
            X: input to the layer
        Returns:
            Y: layer output
        """

    def gradient(self, dY: np.ndarray) -> np.ndarray:
        """Calculate the gradient dL/dX=g(dL/dY), the impact on L by the input dX
        to back propagate to the previous layer.

        Args:
            dY: dL/dY, impact on L by the layer output Y
        Returns:
            dL/dX: impact on L by the layer input X
        """

    def backward(self) -> np.ndarray:
        """Calculate and back-propagate the gradient dL/dX"""

    def gradient_numerical(
            self, h: float = 1e-05
    ) -> np.ndarray:
        """Calculate numerical gradients
        Args:
            h: small number for delta to calculate the numerical gradient
        Returns:
            dX: [L(f(X+h) - L(f(X-h)] / 2h
        """
        def L(X: np.ndarray): return self.objective(self.function(X))
        dX = numerical_gradient(L, self.X)

        return dX

    def update(self, dY: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """Calculate the gradient dL/dS and update S
        Args:
            dY: dL/dY
        Returns:
            dL/dS: Gradient(s) on state S. There may be multiple dL/dS.
        """
