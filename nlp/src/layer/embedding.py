"""Embedding layer implementation
"""
import copy
import logging
from typing import (
    Optional,
    Union,
    List,
    Dict
)

import numpy as np

import optimizer as optimiser
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR
)
from common.function import (
    numerical_jacobian,
)
from layer.base import Layer
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS
)
from layer._utility_builder_non_layer import (
    build_optimizer_from_layer_parameters,
    build_weights_from_layer_parameters
)


class Embedding(Layer):
    """Embedding Layer class
    Batch X: shape(N, R)
    --------------------
    X is N rows of (target, context) pairs.
    - R is the context window size where context includes the target.
      (Not using D as D is the features of event vector or word vector).
    - N is the number of context windows in a sentence = (H-R+1).
    - H is the sequence length.
    - E is the target event size.

    The first E columns in X are the labels.

    The idea is that a target or an event is defined by the context
    in which it occurs. The context including the target is a context window.
    Embedding is to train a model to infer target T from a context.

    For "I am a cat who has" where H=6. With R = 5 and E = 1,
    context windows, and (target, context) pair are:
    1. (I am a cat)       -> (target=a,   context=i,  am, cat, who)
    2. (am a cat who has) -> (target=cat, context=am, a,  who, has)

    Train the model which infers 'a' from a context (i, am, cat, who) and 'cat'
    from (am, a,  who, has).

    Stacking multiple sequences
    --------------------
    X:(N, R) has small N for a short sequence, e.g. N=1 for 'I am a cat deeley'
    as it only has 1 event-context pair (a, I am cat deeley). Hence multiple
    X can be stacked along axis 0.

    TODO: Decide where to handle the stacking

    Weights W: shape(M, D)
    --------------------
    W is event embedding where each row represents an event (word).
    - M is the number of vocabulary that depends on the vocabulary_size of the
    EventIndexing instance.
    - D is the number of features in the event vector.

    W is a concept vector space in which each concept represented by a event
    (word) is encoded with a vector of the finite length D. word2vec is proven
    to be able to capture 'concept' e.g. king=man+(queen-woman). This is the
    proof that the idea "event is defined by its context" is correct.

    Gradient dL/dW: shape(M, D)
    --------------------
    Has the same shape of W.
   """
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return Embedding.specification(
            name="embedding001",
            num_nodes=3,
            num_features=2,     # without bias
        )

    @staticmethod
    def specification(
            name: str,
            num_nodes: int,
            num_features: int,
            weights_initialization_scheme: str = "uniform",
            weights_optimizer_specification: dict = None
    ):
        """Generate Embedding specification
        Args:
            name: layer name
            num_nodes: number of nodes (outputs) in the layer
            num_features: number of features in the layer input (without bias)
            weights_initialization_scheme: weight initialization scheme e.g. he
            weights_optimizer_specification:
                optimizer specification. Default to  SGD
        """
        return {
            _SCHEME: Embedding.__qualname__,
            _PARAMETERS: {
                _NAME: name,
                _NUM_NODES: num_nodes,
                _NUM_FEATURES: num_features,  # NOT including bias
                _WEIGHTS: {
                    _SCHEME: weights_initialization_scheme
                },
                _OPTIMIZER: weights_optimizer_specification
                if weights_optimizer_specification is not None
                else optimiser.SGD.specification_template()
            }
        }

    @staticmethod
    def build(parameters: Dict):
        """Build a Embedding layer based on the parameters
        parameters: {
            "num_nodes": 8,
            "num_features": 2,  # NOT including bias
            "weight": <weight_spec>,
            "optimizer": <optimizer_spec>
        }
        """
        parameters = copy.deepcopy(parameters)
        assert (
            isinstance(parameters, dict) and
            (_NAME in parameters and len(parameters[_NAME]) > 0) and
            (_NUM_NODES in parameters and parameters[_NUM_NODES] > 0) and
            (_NUM_FEATURES in parameters and parameters[_NUM_FEATURES] > 0) and
            _WEIGHTS in parameters
        ), "Embedding.build(): missing mandatory elements %s in the parameters\n%s" \
           % ((_NAME, _NUM_NODES, _NUM_FEATURES, _WEIGHTS), parameters)

        name = parameters[_NAME]
        num_nodes = parameters[_NUM_NODES]
        num_features = parameters[_NUM_FEATURES]

        # Weights
        W = build_weights_from_layer_parameters(parameters)

        # Optimizer
        _optimizer = build_optimizer_from_layer_parameters(parameters)

        embedding = Embedding(
            name=name,
            num_nodes=num_nodes,
            W=W,
            optimizer=_optimizer,
            log_level=parameters["log_level"] if "log_level" in parameters else logging.ERROR
        )

        return embedding

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def vocabulary_size(self) -> TYPE_INT:
        """vocabulary size. Same with M"""
        return self.M

    @property
    def window_size(self) -> TYPE_INT:
        """vocabulary size. Same with R"""
        return self.R

    @property
    def target_size(self) -> TYPE_INT:
        """Target event length. Same with E"""
        return self.E

    @property
    def W(self) -> TYPE_TENSOR:
        """Layer weight vectors W"""
        return self._W

    @property
    def dW(self) -> np.ndarray:
        """Layer weight gradients dW"""
        assert self._dW.size > 0, "dW is not initialized"
        return self._dW

    @property
    def X(self) -> TYPE_TENSOR:
        """Latest batch input to the layer"""
        return super().X

    @X.setter
    def X(self, X: TYPE_TENSOR):
        """Set X"""
        super(Embedding, type(self)).X.fset(self, X)
        assert self.X.shape[1] == self.D, \
            "X shape needs (%s, %s) but %s" % (self.N, self.D, self.X.shape)

    @property
    def S(self) -> List[Union[TYPE_FLOAT, np.ndarray]]:
        """State of the layer"""
        self._S = [self.W]
        return self._S

    @property
    def optimizer(self) -> optimiser.Optimizer:
        """Optimizer instance for gradient descent
        """
        return self._optimizer

    @property
    def lr(self) -> Union[float, np.ndarray]:
        """Learning rate of the gradient descent"""
        return self.optimizer.lr

    @lr.setter
    def lr(self, lr):
        """Set Learning rate"""
        self.optimizer.lr = lr

    @property
    def l2(self) -> Union[float, np.ndarray]:
        """L2 regularization hyper parameter"""
        return self.optimizer.l2

    @l2.setter
    def l2(self, l2):
        """Set L2 regularization"""
        self.optimizer.l2 = l2

    def __init__(
            self,
            name: str,
            num_nodes: int,
            W: np.ndarray,
            posteriors: Optional[List[Layer]] = None,
            optimizer: optimiser.Optimizer = optimiser.SGD(),
            log_level: int = logging.ERROR
    ):
        """Initialize a Embedding layer that has 'num_nodes' nodes
        Input X:(N,D) is a batch. D is number of features NOT including bias
        Weight W:(M, D+1) is the layer weight including bias weight.
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            W: Weight of shape(M=num_nodes, D+1). A row is a weight vector of a node.
            posteriors: Post layers to which forward the Embedding layer output
            optimizer: Gradient descent implementation e.g SGD, Adam.
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        # --------------------------------------------------------------------------------
        # W: weight matrix of shape(M,D) where M=num_nodes
        # Gradient dL/dW has the same shape shape(M, D) with W because L is scalar.
        #
        # Not use WT because W keeps updated every cycle, hence need to update WT as well.
        # Hence not much performance gain and risk of introducing bugs.
        # self._WT: np.ndarray = W.T          # transpose of W
        # --------------------------------------------------------------------------------
        assert W.shape[0] == num_nodes, \
            f"W shape needs to be ({num_nodes}, D) but {W.shape}."
        self._D = W.shape[1]                    # number of features in x including bias
        self._W: np.ndarray = copy.deepcopy(W)  # node weight vectors
        self._dW: np.ndarray = np.empty(0, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # State of the layer
        # --------------------------------------------------------------------------------
        self._S = [self.W]

        self.logger.debug(
            "Embedding[%s] W.shape is [%s], number of nodes is [%s]",
            name, W.shape, num_nodes
        )
        # --------------------------------------------------------------------------------
        # Optimizer for gradient descent
        # Z(n+1) = optimiser.update((Z(n), dL/dZ(n)+regularization)
        # --------------------------------------------------------------------------------
        assert isinstance(optimizer, optimiser.Optimizer)
        self._optimizer: optimiser.Optimizer = optimizer

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def function(self, X: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate the layer output Y = X@W.T
        Args:
            X: Batch input data from the input layer without the bias.
        Returns:
            Y: Layer value of X@W.T
        """
        assert isinstance(X, np.ndarray) and X.dtype == TYPE_FLOAT, \
            f"Only np array of type {TYPE_FLOAT} is accepted"

        name = "function"
        # --------------------------------------------------------------------------------
        # Y = (W@X + b) could be efficient
        # --------------------------------------------------------------------------------
        if X.ndim <= 1:
            X = np.array(X).reshape(1, -1)

        self.X = np.c_[
            np.ones(X.shape[0], dtype=TYPE_FLOAT),  # Add bias
            X
        ]
        assert self.X.shape == (self.N, self.D)

        self.logger.debug(
            "layer[%s].%s: X.shape %s W.shape %s",
            self.name, name, self.X.shape, self.W.shape
        )
        assert self.W.shape == (self.M, self.D), \
            f"W shape needs {(self.M, self.D)} but ({self.W.shape})"

        # --------------------------------------------------------------------------------
        # Allocate array storage for np.func(out=) for Y but not dY.
        # Y:(N,M) = [ X:(N,D) @ W.T:(D,M) ]
        # gradient() need to validate the dY shape is (N,M)
        # --------------------------------------------------------------------------------
        if self._Y.size <= 0 or self.Y.shape[0] != self.N:
            self._Y = np.empty((self.N, self.M), dtype=TYPE_FLOAT)
            # --------------------------------------------------------------------------------
            # DO NOT allocate memory area for the gradient that has already been calculated.
            # dL/dY is calculated at the post layer, hence it has the buffer allocated already.
            # --------------------------------------------------------------------------------
            # self._dY = np.empty((self.N, self.M), dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # TODO:
        # Because transpose(T) is run everytime Embedding is invoked, using the transposed W
        # would save the calculation time. This is probably the reason why cs231n uses
        # in column order format.
        # --------------------------------------------------------------------------------
        np.Embedding(self.X, self.W.T, out=self._Y)
        assert np.all(np.isfinite(self.Y)), f"{self.Y}"
        return self.Y

    def gradient(self, dY: Union[np.ndarray, float] = 1.0) -> Union[np.ndarray, float]:
        """Calculate the gradients dL/dX and dL/dW.
        Args:
            dY: Gradient dL/dY, the total impact on L by dY.
        Returns:
            dX: dL/dX of shape (N, D-1) without the bias
        """
        name = "gradient"
        assert isinstance(dY, float) or (isinstance(dY, np.ndarray) and dY.dtype == TYPE_FLOAT)

        dY = np.array(dY).reshape((1, -1)) if isinstance(dY, float) or dY.ndim < 2 else dY
        assert dY.shape == self.Y.shape, \
            "dL/dY shape needs %s but %s" % (self.Y.shape, dY.shape)

        self.logger.debug("layer[%s].%s: dY.shape %s", self.name, name, dY.shape)
        self._dY = dY

        # --------------------------------------------------------------------------------
        # dL/dW of shape (M,D):  [ X.T:(D, N)  @ dL/dY:(N,M) ].T
        # --------------------------------------------------------------------------------
        dW = np.Embedding(self.X.T, self.dY).T
        assert dW.shape == (self.M, self.D), \
            f"Gradient dL/dW shape needs {(self.M, self.D)} but ({dW.shape}))"

        self._dW = dW
        assert np.all(np.isfinite(self.dW)), f"{self.dW}"

        # --------------------------------------------------------------------------------
        # dL/dX of shape (N,D):  [ dL/dY:(N,M) @ W:(M,D)) ]
        # --------------------------------------------------------------------------------
        np.Embedding(self.dY, self.W, out=self._dX)
        assert self.dX.shape == (self.N, self.D), \
            "dL/dX shape needs (%s, %s) but %s" % (self.N, self.D, self.dX.shape)

        assert np.all(np.isfinite(self.dX)), f"{self.dX}"
        return self.dX[
            ::,
            1::     # Omit bias column 0
        ]

    def gradient_numerical(
            self, h: Optional[TYPE_FLOAT] = None
    ) -> List[Union[float, np.ndarray]]:
        """Calculate numerical gradients
        Args:
            h: small number for delta to calculate the numerical gradient
        Returns:
            [dX, dW]: Numerical gradients for X and W without bias
            dX is dL/dX of shape (N, D-1) without the bias to match the original input
            dW is dL/dW of shape (M, D) including the bias weight w0.
        """
        name = "gradient_numerical"
        self.logger.debug("layer[%s].%s", self.name, name)
        L = self.objective
        WT = self.W.T

        def objective_X(x: np.ndarray):
            return L(x @ WT)

        def objective_W(w: np.ndarray):
            return L(self.X @ w.T)

        dX = numerical_jacobian(objective_X, self.X, delta=h)
        dX = dX[
            ::,
            1::     # Omit the bias
        ]
        dW = numerical_jacobian(objective_W, self.W, delta=h)
        return [dX, dW]

    def _gradient_descent(self, W, dW, out=None) -> Union[np.ndarray, float]:
        """Gradient descent
        Directly update matrices to avoid the temporary copies
        """
        return self.optimizer.update(W, dW, out=out)

    def update(self) -> List[Union[float, np.ndarray]]:
        """
        Responsibility: Update layer state with gradient descent.

        1. Calculate dL/dW = (dL/dY * dY/dW).
        2. Update W with the optimizer.

        dL/dW.T:(D,M) = [ X.T:(D, N) @ dL/dY:(N,M) ].
        dL/dW:  (M,D):  [ X.T:(D, N) @ dL/dY:(N,M) ].T.

        Returns:
            [dL/dW]: List of dL/dW.
            dW is dL/dW of shape (M, D) including the bias weight w0.

        Note:
            update() is to update the state of the layer S. Hence not
            include dL/dX which is not part of the layer state.
       """
        self._gradient_descent(self.W, self.dW, out=self._W)
        self._dS = [self.dW]

        return self.dS

    def load(self, path: str):
        """Load and restore the layer state
        Consideration:
            Need to be clear if update a reference to the state object OR
            update the object memory area itself.

            If switching the reference to the new state object, there would be
            references to the old objects which could cause unexpected results.

            Hence, if state object memory can be directly updated, do so.
            If not, **delete** the object so that the references to the old
            object will cause an error and fix the code not to hold a reference
            but get the reference via the property method every time.

        TODO:
            Consider if resetting other properties (dW, X, etc) are required.

        Args:
            path: state file path
        """
        state = super().load(path)
        if self.W.shape == state[0].shape:
            np.copyto(self._W, state[0])
        else:
            del self._W
            self._W = state[0]
