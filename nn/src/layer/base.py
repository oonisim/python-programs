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
Note:
    Li is a composition of post layer functions. As each layer can have its state,
    only layer output Yi=fi(Xi) should go into the next layer function fi+1.

    If fi+1 is invoked from the layer i by other than fi, it would cause
    unexpected state transfers in the network. Hence invoking the function of the
    post layers from e.g. numerical gradient gn() must NOT happen.
    Set a stateless Li to the objective function Li for the layer i.

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
import logging
from traceback import print_stack
from typing import (
    Any,
    Optional,
    Union,
    List,
    Dict,
    Callable,
    NoReturn
)

import numpy as np

from common.constant import (
    TYPE_FLOAT,
    TYPE_LABEL,
    TYPE_TENSOR,
    GRADIENT_SATURATION_THRESHOLD,
)
from common.function import (
    numerical_jacobian,
)
import function.fileio as fileio
# import function.nn.base as nn
import function.nn.tf as nn


class Layer(nn.Function):
    """Neural network layer base class"""
    # ================================================================================
    # Class
    # ================================================================================
    @classmethod
    def class_id(cls):
        """Identify the class
        Avoid using Python implementation specific __qualname__

        Returns:
            Class identifier which may not be unique as Python does not have
            a way to get fully qualified hierarchical name
        """
        return cls.__qualname__

    @staticmethod
    def build(parameters: Dict):
        """Build a matmul layer based on the specification
        parameters: {
            "name": "name
            "num_nodes": 8
        }
        """
        raise NotImplementedError("Must implement")

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
    def X(self) -> TYPE_TENSOR:
        """Latest batch input to the layer"""
        if not (
            (self.is_tensor(self._X) and self.tensor_size(self._X) > 0) or
            self.is_float_scalar(self._X)
        ):
            # print_stack()     # pytest comes here and show lots of lines...
            raise AssertionError("X is not initialized or invalid. type(X)={} X=\n{}\n".format(
                type(self._X), self._X
            ))

        return self._X

    @X.setter
    def X(self, X: TYPE_TENSOR):
        """Set layer input X
        2. Allocate _dX storage.
        3. DO NOT set/update _D as it can be set with the weight shape.
        """
        # X can be string e.g EventIndexing
        # assert X is not None and self.is_tensor(X) and self.is_finite(X), \
        assert X is not None and self.is_tensor(X), f"Invalid X {X}"
        if self.tensor_dtype(X) == TYPE_FLOAT:
            assert self.is_finite(X)

        if self.is_float_tensor(X):
            assert np.all(np.isfinite(X)), f"{X}"
            if np.all(np.abs(X) > TYPE_FLOAT(1.0)):
                self.logger.warning("Input data X has not been standardized.")

        self._X = X
        self._N = self.tensor_shape(X)[0]

        # Allocate the storage for np.func(out=dX).
        if not self.is_same_shape(self._dX, self._X):
            # self._dX = np.empty(self._X.shape, dtype=self._X.dtype)
            self._dX = np.empty(self.tensor_shape(X), dtype=TYPE_FLOAT)

        # DO NOT
        # self._D = X.shape[1]

    @property
    def N(self) -> int:
        """Batch size"""
        assert self._N >= 0, "N is not initialized"
        return self._N

    @property
    def dX(self) -> TYPE_TENSOR:
        """Gradient dL/dX"""
        assert isinstance(self._dX, np.ndarray) and self._dX.size > 0, \
            "dX is not initialized"

        assert self.is_finite(self._dX), "Nan/inf detected \n%s\n" % self._dX
        if np.all(np.abs(self._dX) < GRADIENT_SATURATION_THRESHOLD):
            self.logger.warning(
                "Gradient dL/dX \n[%s] may have been saturated." % self._dX
            )

        return self._dX

    @property
    def T(self) -> TYPE_TENSOR:
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
    def Y(self) -> TYPE_TENSOR:
        """Latest layer output
        No need to allocate a storage for dY as it is allocated by the post layer.
        """
        # Y can be TYPE_INT tensor for e.g. EventIndexing
        # assert \
        #     (self.is_float_tensor(self._Y) and self.tensor_size(self._Y) > 0) or \
        #     self.is_float_scalar(self._Y), \
        #     "Y %s of type %s is not initialized or invalid." % \
        #     (self._Y, type(self._Y))
        assert \
            (self.is_float_tensor(self._Y) and self.tensor_size(self._Y) > 0) or \
            (self.is_integer_tensor(self._Y) and self.tensor_size(self._Y) > 0) or \
            self.is_float_scalar(self._Y), \
            "Y %s of type %s is not initialized or invalid." % \
            (self._Y, type(self._Y))
        if self.tensor_dtype(self._Y) == TYPE_FLOAT:
            assert self.is_finite(self._Y), "nan/inf detected\n%s\n" % self._Y
        return self._Y

    @property
    def dY(self) -> TYPE_TENSOR:
        """Latest gradient dL/dY (impact on L by dY) given from the post layer(s)"""
        assert \
            self.is_float_tensor(self._dY) and self.tensor_size(self._dY) > 0, \
            "dY is not initialized or invalid"

        if self.tensor_dtype(self._dY) == TYPE_FLOAT:
            assert self.is_finite(self._dY), "nan/inf detected\n%s\n" % self._dY

        if np.all(np.abs(self._dY) < GRADIENT_SATURATION_THRESHOLD):
            self.logger.warning(
                "Gradient dY/dX \n[%s] may have been saturated." % self._dY[5::]
            )
        return self._dY

    @dY.setter
    def dY(self, dY):
        assert self.is_same_shape(dY, self.Y), \
            "dL/dY shape needs %s but %s" \
            % (self.tensor_shape(self.Y), self.tensor_shape(dY))
        self._dY = dY

    @property
    def S(self) -> Union[List, Dict]:
        """State of the layer
        For a layer which has no state, it is an empty list.
        Hence cannot validate if initialized or not.
        """
        assert isinstance(self._S, list), \
            "Gradients dL/S of the network not initialized."
        return self._S

    @property
    def dS(self) -> Union[List, Dict]:
        """Gradients dL/dS that have been used to update S in each layer
        For a layer which has no state, it is an empty list.
        Hence cannot validate if initialized or not.
        """
        assert isinstance(self._dS, list), \
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
        super().__init__(name=name, log_level=log_level)

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
        self._S: List = []
        self._dS: List = []

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
    def function(self, X: TYPE_TENSOR) -> TYPE_TENSOR:
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
        self._X = self.to_tensor(X)
        self._Y = self.X
        # In case for the layer is a repeater, pass X through as the default behavior.
        return self.Y

    def forward(self, X: TYPE_TENSOR) -> TYPE_TENSOR:
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

    def gradient(
            self, dY: Union[TYPE_TENSOR, TYPE_FLOAT]
    ) -> Union[TYPE_TENSOR, TYPE_FLOAT]:
        """Calculate the gradient dL/dX, the impact on L by the input dX
        to back propagate to the previous layer, and other gradients on S
        dL/dS = dL/dY * dY/dS.

        Args:
            dY: dL/dY, impact on L by the layer output Y
        Returns:
            dL/dX: impact on L by the layer input X
        """
        # assert isinstance(dY, float) or (isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT)
        dY = self.assure_tensor(dY)
        assert self.tensor_shape(self.Y) == self.tensor_shape(dY)

        # In case the layer is a repeater or no gradient, pass dY through.
        self.logger.warning(
            "Layer base method %s not overridden but called",
        )
        # The shape of dX must match X. Simply back-prop dY which would have a
        # different shape of Y=f(X) is incorrect. We do not know how to restore
        # the shape of dX from dY, hence we cannot handle it here.
        # However, there are layers e.g. event indexing which has no gradient
        # to back-prop. Hence returns:
        # - dL/dY if it has the same shape of X
        # - dL/dY * Ones(shape=X.shape) if dL/dY can be broadcast to X
        # - X

        if self.is_same_shape(dY, self.X):
            pass
        elif self.is_broadcastable(dY, self.X):
            dY = self.multiply(dY, self.X)
        else:
            dY = self.X

        self._dY = dY
        return dY

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
            self, h: TYPE_FLOAT = 1e-5
    ) -> List[Union[TYPE_TENSOR, TYPE_FLOAT]]:
        """Calculate numerical gradients
        Args:
            h: small number for delta to calculate the numerical gradient
        Returns:
            dX: [L(f(X+h) - L(f(X-h)] / 2h
        """
        # L = Li(f(arg))
        def L(X: np.ndarray):
            # pylint: disable=not-callable
            return self.objective(self.function(X))

        dX = numerical_jacobian(L, self.X)
        return [dX]

    def update(self) -> List[Union[TYPE_TENSOR, TYPE_FLOAT]]:
        """Calculate the gradient dL/dS and update S with gradient descent.
        Returns:
            dL/dS:
                List of gradient(s) on state S. There may be multiple dL/dS.

                Back-propagation to the previous layer is not included as
                it is not part of the state S of the layer.

            Why not returning S instead of dS?
            Because state S of of a layer includes all the memory state, both mutable
            and immutable, which the layer needs to functions. An embedding layer
            instance requires a dictionary which may have million events (e.g. words).

            Why not returning the subset of S which have been updated?
            TODO: Answer
        """
        self.logger.warning(
            "Layer base method %s not overridden but called."
        )
        # Return 0 as the default for dL/dS to mark no change in case there is none
        # to update in a layer.
        return self.dS

    def predict(self, X: Union[TYPE_TENSOR, TYPE_FLOAT]):
        """Calculate the score for prediction
        Args:
            X: Input
        Returns:
            Y: score
        """
        return self.function(X)

    def save(self, path: str):
        """Save the layer state
        Args:
            path: path to save the state
        """
        fileio.Function.serialize(path, self.S)

    def load(self, path: str) -> Any:
        """Load the layer state
        The responsibility to restore the layer state is that of the child.
        Consideration:
            Need to be clear if update a reference to the state object OR
            update the object memory area itself.

            If switching the reference to the new state object, there would be
            references to the old objects which could cause unexpected results.

            Hence, if state object memory can be directly updated, do so.
            If not, **delete** the object so that the references to the old
            object will cause an error and fix the code not to hold a reference
            but get the reference via the property method every time.

        Decision"
            1. load() method is responsible only for variables, not for
               class instances or functions. save() method should not save
               such objects.
            2. Delete the old object otherwise orphaned before resetting
               to a new one in the child classes.

        NOTE:
            self.S is a list of references to the layer state objects.
            Hence setting/updating self._S has no effect. You need to update
            the actual layer state objects themselves to restore the states.

        Args:
            path: path to save the state
        Returns:
            state: loaded state
        """
        state = fileio.Function.deserialize(path)
        # DO NOT limit the implementation. Let the actual class handles it.
        # assert isinstance(state, list) and len(state) > 0
        return state

