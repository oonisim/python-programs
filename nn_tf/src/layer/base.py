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
1. Use 'to' to clarify the goal
2. Be specific what a function outputs as its observable behavior and why.

state S = List[s]:
    A state in a layer is a combination of variables [s] at a specific cycle t.

Y=f(arg):
    f is a function to calculates the layer output Y=f(arg) where arg is the
    output of the previous layer.

G=g(dL/dY):
    g is a function to calculates gradients. dL/dY is given because the layer i
    does not know f at layers i+1, ... n-1 hence cannot calculate it.
    Later 'u' runs gradient descents to update S.

    As its observable behaviour, returns the gradient dL/dX = dL/dY * dY/dX
    to back-propagate to the previous layers.

    The internal behavior of calculating dL/dS=(dL/dY * dY/dS) is not observable,
    but g focuses on gradients and u focuses on updating state S as the
    separation of concerns.


List[GN]=gn(Li, h):
    gn is a function to calculate numerical gradients, given L and h, as
    gn=[ Li(fi(arg+h)) = Li(fi(arg-h)) ] / 2h

    arg: any argument that computes the output Y of the layer.
         Although fi can take a single arg, the actual computation may need
         multiple arguments e.g. X and W for matmul. Then gn calculates for
         X and W, and returns two numerical gradients.
    Li: the objective function of the layer Li=(fn-1 o ... fi+1) with i < n-1
        objective=Li(Y)=L(f(arg)), NOT L(arg).
    h: a small number e.g. 1e-5

    As its observable behavior of what it has done, returns a list of
    numerical gradients as [GN] as there can be multiple GN.

List[dL/ds]=u(dL/dY):
    u is a function that calculates dL/dS = u(dL/dY) = dL/dY * dY/dS and updates
    the state S in the layer with gradient descent as S=optimizer(S, dL/dS).
    There may be multiple dS to calculate and u handles all.

    As its observable behaviour of what is has done, returns the dL/dS
    as a list of gradients for the state variables as List[dL/ds].

    As u is to 'update' S, the observable behaviour can be returning updated S.
    However, to be able to validate the dL/dS with the numerical gradient [GN],
    externally, returns dL/dS. It would be possible to return updated S and
    validate dL/dS with GN internally with another function 'v' to validate.

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
    Tuple,
    Callable,
    Any,
    NoReturn,
    Final
)
import logging
import numpy as np
from common.constants import (
    TYPE_FLOAT,
    TYPE_LABEL,
    GRADIENT_SATURATION_THRESHOLD,
)
from common.function import (
    numerical_jacobian,
)
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS
)


class Layer:
    """Neural network layer base class"""
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def build(parameters: Dict):
        """Build a matmul layer based on the specification
        parameters: {
            "name": "name
            "num_nodes": 8
        }
        """
        assert False, "Must override"

    # ================================================================================
    # Instance
    # ================================================================================

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
    def M(self) -> int:
        """Number of nodes in a layer"""
        return self.num_nodes

    @property
    def D(self) -> int:
        """Number of feature of a node in a layer"""
        assert self._D > 0
        return self._D

    @property
    def X(self) -> np.ndarray:
        """Latest batch input to the layer"""
        assert isinstance(self._X, np.ndarray) and self._X.dtype == TYPE_FLOAT \
            and self._X.size > 0, "X is not initialized or invalid"
        return self._X

    @X.setter
    def X(self, X: Union[float, np.ndarray]):
        """Set layer input X
        1. Convert into 2D array if X is scalar or X.ndim < 2.
        2. Allocate _dX storage.
        3. DO NOT set/update _D as it can be set with the weight shape.
        """
        assert X is not None and \
               ((isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT) or isinstance(X, float))

        if np.all(np.abs(X) > 1.0):
            self.logger.warning("Input data X has not been standardized.")

        if isinstance(X, float):
            self._X = np.array(X)
        elif X.ndim == 1:
            self._X = np.array(X).reshape(1, -1)
        else:
            self._X = X

        assert self.X.size > 0
        self._N = self.X.shape[0]

        # Allocate the storage for np.func(out=dX).
        if self._dX.shape != self.X.shape:
            self._dX = np.empty(self.X.shape, dtype=TYPE_FLOAT)

        # DO NOT
        # self._D = X.shape[1]

    @property
    def N(self) -> int:
        """Batch size"""
        assert self._N >= 0, "N is not initialized"
        return self._N

    @property
    def dX(self) -> np.ndarray:
        """Gradient dL/dX"""
        assert isinstance(self._dX, np.ndarray) and self._dX.size > 0, \
            "dX is not initialized"

        if np.all(np.abs(self._dX) < GRADIENT_SATURATION_THRESHOLD):
            self.logger.warning(
                "Gradient dL/dX \n[%s] may have been saturated." % self._dX
            )

        return self._dX

    @property
    def T(self) -> np.ndarray:
        """Label in OHE or index format"""
        assert self._T is not None and self._T.size > 0, "T is not initialized"
        return self._T

    @T.setter
    def T(self, T: Union[np.ndarray, int]):
        assert T is not None and (
                (isinstance(T, np.ndarray) and np.issubdtype(T.dtype, np.integer)) or
                (isinstance(T, int))
            )

        self._T = np.array(T, dtype=TYPE_LABEL) if isinstance(T, int) else T.astype(int)
        # T can be set after X, hence not possible to verify.
        # assert T.shape[0] == self.N, \
        #     f"Set X first and the batch size of T should be {self.N} but {T.shape[0]}"

    @property
    def Y(self) -> np.ndarray:
        """Latest layer output
        No need to allocate a storage for dY as it is allocated by the post layer.
        """
        assert \
            (isinstance(self._Y, np.ndarray) and self._Y.dtype == TYPE_FLOAT) and \
            self._Y.size > 0, \
            "Y %s of type %s is not initialized or invalid." % \
            (self._Y, type(self._Y))
        return self._Y

    @property
    def dY(self) -> np.ndarray:
        """Latest gradient dL/dY (impact on L by dY) given from the post layer(s)"""
        assert \
            isinstance(self._dY, np.ndarray) and \
            self._dY.dtype == TYPE_FLOAT \
            and self._dY.size > 0, "dY is not initialized or invalid"

        if np.all(np.abs(self._dY) < GRADIENT_SATURATION_THRESHOLD):
            self.logger.warning(
                "Gradient dY/dX \n[%s] may have been saturated." % self._dY[5::]
            )
        return self._dY

    @property
    def S(self) -> \
            Union[
                List[Union[TYPE_FLOAT, np.ndarray]],
                List[
                    List[Union[float, np.ndarray]]
                ]
            ]:
        """State of the layer
        For a layer which has no state, it is an empty list.
        Hence cannot validate if initialized or not.
        """
        assert isinstance(self._S, list), \
            "Gradients dL/S of the network not initialized."
        return self._S

    @property
    def dS(self) -> \
            Union[
                List[Union[TYPE_FLOAT, np.ndarray]],
                List[
                    List[Union[float, np.ndarray]]
                ]
            ]:
        """Gradients dL/dS that have been used to update S in each layer
        For a layer which has no state, it is an empty list.
        Hence cannot validate if initialized or not.
        """
        assert isinstance(self._S, list), \
            "Gradients dL/dS of the network not initialized."
        return self._dS

    @property
    def objective(self) -> Callable[[np.ndarray], np.ndarray]:
        """Objective function L=fn-1 o fn-2 o ... o fi
        Layer i has its objective function Li = (fn-1 o ... o fi+1) for i < n-1
        that calculates L=Li(Yi) from its output Yi=fi(Xi). L = Li(fi) i < n-1.

        Li   = (fn-1 o ... o fi+1)
        Li-1 = (fn-1 o ... o fi+1 o fi) = Li(fi) = Li o fi
        L = Li-1(fi-1(Xi-1)) = Li(fi(Yi-1)) where Yi-1=fi-1(Xi-1)
        #
        Set Li to each inference layer i in the reverse order.
        Ln-2 for the last inference layer n-2 is fn-1 of the objective layer.
        Ln-2 = fn-1 by the definition Li = (fn-1 o ... o fi+1).
        """
        assert callable(self._objective), "Objective function L has not been initialized."
        return self._objective

    @objective.setter
    def objective(self, Li: Callable[[np.ndarray], np.ndarray]) -> NoReturn:
        assert Li is not None and callable(Li)
        self._objective = Li

    @property
    def logger(self) -> logging.Logger:
        """Instance logger"""
        assert isinstance(self._logger, logging.Logger), "logger is not initialized"
        return self._logger

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            num_nodes: int,
            posteriors: Optional[List] = None,
            log_level: int = logging.WARNING
    ):
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

        # --------------------------------------------------------------------------------
        # X: batch input of shape(N, D)
        # Gradient dL/dX of has the same shape(N,D) with X because L is scalar.
        # --------------------------------------------------------------------------------
        self._X: np.ndarray = np.empty(0, dtype=TYPE_FLOAT)
        self._N: int = -1
        self._dX: np.ndarray = np.empty(0, dtype=TYPE_FLOAT)
        self._D: int = -1

        # Labels of shape(N, M) for OHE or (N,) for index.
        self._T: np.ndarray = np.empty(0, dtype=TYPE_LABEL)

        # --------------------------------------------------------------------------------
        # layer output Y of shape(N, M)
        # Gradient dL/dY has the same shape (N, M) with Y because the L is scalar.
        # dL/dY is the sum of all the impact on L by dY.
        # --------------------------------------------------------------------------------
        self._Y: np.ndarray = np.empty(0, dtype=TYPE_FLOAT)
        self._dY: np.ndarray = np.empty(0, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # State of the layer
        # --------------------------------------------------------------------------------
        self._S: Union[
            List[Union[TYPE_FLOAT, np.ndarray]],
            List[
                List[Union[float, np.ndarray]]
            ]
        ] = []   # Gradients dL/dS of layers
        self._dS: Union[
            List[Union[TYPE_FLOAT, np.ndarray]],
            List[
                List[Union[float, np.ndarray]]
            ]
        ] = []   # Gradients dL/dS of layers

        # --------------------------------------------------------------------------------
        # Layers to which forward the matmul output
        # --------------------------------------------------------------------------------
        self._posteriors: List[Layer] = posteriors
        self._num_posteriors: int = len(posteriors) if posteriors else -1

        # --------------------------------------------------------------------------------
        # Objective Li function for the layer. L = Li o fi
        # --------------------------------------------------------------------------------
        self._objective: Callable[[np.ndarray], np.ndarray] = None

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging._levelToName[log_level])

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate the output f(arg) to forward as the input to the post layer.
        Args:
            X: input to the layer
        Returns:
            Y: Layer output
        """
        assert isinstance(X, float) or (isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT)
        self.logger.warning(
            "Layer base method %s not overridden but called.",
        )

        # In case for the layer is a repeater, pass X through as the default behavior.
        X = np.array(X).reshape((1, -1)) if isinstance(X, float) else X
        return X

    def forward(self, X: np.ndarray) -> Union[np.ndarray, float]:
        """Calculate and forward-propagate the matmul output Y to post layers if exist.
        Args:
            X: input to the layer
        Returns:
            Y: layer output
        """
        assert isinstance(self._posteriors, list) and len(self._posteriors) > 0, \
            "forward(): No post layer exists."

        def _forward(Y: np.ndarray, layer: Layer) -> None:
            """Forward the matmul output Y to a post layer
            Args:
                Y: Standardization output
                layer: Layer where to propagate Y.
            Returns:
                Z: Return value from the post layer.
            """
            layer.forward(Y)

        Y: np.ndarray = self.function(X)
        list(map(_forward, Y, self._posteriors))
        return Y

    def gradient(self, dY: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate the gradient dL/dX, the impact on L by the input dX
        to back propagate to the previous layer, and other gradients on S
        dL/dS = dL/dY * dY/dS.

        Args:
            dY: dL/dY, impact on L by the layer output Y
        Returns:
            dL/dX: impact on L by the layer input X
        """
        assert isinstance(dY, float) or (isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT)

        # In case the layer is a repeater or no gradient, pass dY through.
        self.logger.warning(
            "Layer base method %s not overridden but called",
        )
        assert isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT
        return dY

    # def backward(self) -> Union[np.ndarray, float]:
    #     """Calculate and back-propagate the gradient dL/dX"""
    #     self.logger.warning(
    #         "Layer base method %s not overridden but called by %s.",
    #         inspect.stack()[0][3], inspect.stack()[1][3]
    #     )
    #     assert False, "Need to override"
    #     return np.array(-np.inf)
    def backward(self) -> Union[np.ndarray, float]:
        """Calculate the gradient dL/dX to back-propagate
        """
        assert isinstance(self._posteriors, list) and len(self._posteriors) > 0, \
            "backward() called when no post layer exist."

        def _backward(layer: Layer) -> np.ndarray:
            """Get gradient dL/dY from a post layer
            Args:
                layer: a post layer
            Returns:
                dL/dY: the impact on L by the layer output dY
            """
            # --------------------------------------------------------------------------------
            # Back propagation from the post layer(s)
            # dL/dY has the same shape with Y:shape(N, M) as L and dL are scalar.
            # --------------------------------------------------------------------------------
            dY: np.ndarray = layer.backward()
            assert np.array_equal(dY.shape, (self.N, self.M)), \
                f"dY.shape needs {(self.N, self.M)} but ({dY.shape}))"

            return dY

        # --------------------------------------------------------------------------------
        # Gradient dL/dY, the total impact on L by dY, from post layer(s) if exist.
        # np.add.reduce() is faster than np.sum() as sum() calls it internally.
        # --------------------------------------------------------------------------------
        dY = np.add.reduce(map(_backward, self._posteriors))
        return self.gradient(dY)

    def gradient_numerical(
            self, h: float = 1e-5
    ) -> List[Union[float, np.ndarray]]:
        """Calculate numerical gradients
        Args:
            h: small number for delta to calculate the numerical gradient
        Returns:
            dX: [L(f(X+h) - L(f(X-h)] / 2h
        """
        # self.logger.warning(
        #    "Layer base method gradient_numerical not overridden but called by %s."
        # )

        # L = Li(f(arg))
        def L(X: np.ndarray):
            return self.objective(self.function(X))

        dX = numerical_jacobian(L, self.X)
        return [dX]

    def update(self) -> List[Union[float, np.ndarray]]:
        """Calculate the gradient dL/dS and update S with gradient descent.
        Returns:
            dL/dS:
                Gradient(s) on state S. There may be multiple dL/dS.

                Back-propagation to the previous layer is not included as
                it is not part of the state S of the layer.
        """
        self.logger.warning(
            "Layer base method %s not overridden but called."
        )
        # Return 0 as the default for dL/dS to mark no change in case there is none
        # to update in a layer.
        return self.dS

    def predict(self, X):
        """Calculate the score for prediction
        Args:
            X: Input
        Returns:
            Y: score
        """
        return self.function(X)
