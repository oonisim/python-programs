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
    TYPE_TENSOR,
    EVENT_VECTOR_SIZE,
    EVENT_META_ENTITIES
)
from common.function import (
    numerical_jacobian,
)
# TODO: Update to layer.base once RC is done
from layer.base import Layer
from layer.preprocessing import (
    EventIndexing
)
from layer.constants import (
    _WEIGHTS,
    _NAME,
    _SCHEME,
    _OPTIMIZER,
    _NUM_NODES,
    _NUM_FEATURES,
    _PARAMETERS,
    MAX_NEGATIVE_SAMPLE_SIZE,
)
from layer._utility_builder_non_layer import (
    build_optimizer_from_layer_parameters,
    build_weights_from_layer_parameters
)


class Embedding(Layer):
    """Embedding Layer class
    Batch X: shape(N, E+C)
    --------------------
    X is N rows of (event, context) pairs.
    - N is the number of context windows in a sentence = (H-C+1).
    - E is the event size.
    - C is the context size.
    - H is the sequence length.

    The first E columns in X are the labels telling what is the event(s) to be
    inferred from the context. Context Windows = (event, context) and its size
    windows_size = E+C

    The idea is that an event is defined by the context in which it occurs.
    Embedding is to train a model to infer target from a context.

    For "I am a cat who has" where H=6. With C = 5 and E = 1,
    context windows, and (target, context) pair are:
    1. (I am a cat)       -> (target=a,   context=i,  am, cat, who)
    2. (am a cat who has) -> (target=cat, context=am, a,  who, has)

    Train the model which infers 'a' from a context (i, am, cat, who) and 'cat'
    from (am, a,  who, has).

    Stacking multiple sequences
    --------------------
    X:(N, C) has small N for a short sequence, e.g. N=1 for 'I am a cat deeley'
    as it only has 1 event-context pair (a, I am cat deeley). Hence multiple
    X can be stacked along axis 0.

    TODO: Decide where to handle the stacking

    Weights W: shape(V, D)
    --------------------
    W is event vector space where each row represents an event (e.g. word).
    - V is the number of vocabulary that depends on the vocabulary_size of the
    EventIndexing instance.
    - D is the number of features in a event vector.

    W is a concept vector space in which each concept represented by a event
    is encoded with a vector of the finite length D. word2vec is proven to be
    able to capture 'concept' e.g. king=man+(queen-woman). This is the proof
    that the idea "event is defined by its context" is correct.
    """
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        raise NotImplementedError("TBD")

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
        raise NotImplementedError("TBD")

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
        raise NotImplementedError("TBD")

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def E(self) -> TYPE_INT:
        """Target event size"""
        return self._target_size

    @property
    def C(self) -> TYPE_INT:
        """Context size"""
        return self._context_size

    @property
    def window_size(self) -> TYPE_INT:
        """Context window size (E+C)"""
        return self.E + self.C

    @property
    def negative_sample_size(self) -> TYPE_INT:
        """Negative sample size"""
        return self._negative_sample_size

    @property
    def dictionary(self) -> EventIndexing:
        """Event dictionary"""
        assert \
            isinstance(self._dictionary, EventIndexing) and \
            self._dictionary.vocabulary_size > len(EVENT_META_ENTITIES), \
            "Invalid vocabulary size"
        return self._dictionary

    @property
    def V(self) -> TYPE_INT:
        """vocabulary size. Same"""
        return self.dictionary.vocabulary_size

    @property
    def W(self) -> TYPE_TENSOR:
        """Layer weight vectors W"""
        assert self._W is not None and self.tensor_size(self._W) > 0
        return self._W

    @property
    def dW(self) -> TYPE_TENSOR:
        """Layer weight gradients dW"""
        assert self.tensor_size(self._dW.size) > 0, "dW is not initialized"
        return self._dW

    @property
    def X(self) -> TYPE_TENSOR:
        """Latest batch input to the layer"""
        return super().X

    @X.setter
    def X(self, X: TYPE_TENSOR):
        """Batch input (event, context) pairs.
        """
        assert X.shape[1] == self.window_size, \
            "Xi needs shape (E+C,) but %s" % X.shape[1]
        super(Embedding, type(self)).X.fset(self, X)

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
            num_nodes: TYPE_INT = 1,
            target_size: TYPE_INT = 1,
            context_size: TYPE_INT = 4,
            negative_sample_size: TYPE_INT = 10,
            event_vector_size: TYPE_INT = EVENT_VECTOR_SIZE,
            dictionary: EventIndexing = None,
            W: Optional[TYPE_TENSOR] = None,
            optimizer: optimiser.Optimizer = optimiser.SGD(),
            posteriors: Optional[List[Layer]] = None,
            log_level: int = logging.ERROR
    ):
        """Initialize a Embedding layer that has 'num_nodes' nodes
        Input X:(N,E+C) is a batch.
        Weight W:(M, D) is the event vector space of size V.
        Args:
            name: Layer identity name
            num_nodes: Number of output M
            target_size: size of the target event of the (target, context) pair.
            context_size: size of the context of the (target, context) pair.
            negative_sample_size: size of the negative samples
            event_vector_size: size of the event vector
            dictionary: Dictionary to consult event probabilities and event samples.
            W: Weight of shape(V=vocabulary_size, D).
            posteriors: Post layers to which forward the Embedding layer output
            optimizer: Gradient descent implementation e.g SGD, Adam.
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        assert num_nodes == 1, "Number of output should be 1 for logistic classification"
        assert target_size > 0 == (context_size % 2) and 0 < context_size
        assert isinstance(dictionary, EventIndexing) and dictionary.vocabulary_size > 2
        availability_for_negatives = (
            dictionary.vocabulary_size
            - (context_size+target_size)
            - len(EVENT_META_ENTITIES)
        )
        assert \
            0 < negative_sample_size <= MAX_NEGATIVE_SAMPLE_SIZE and \
            negative_sample_size <= availability_for_negatives

        self._M = num_nodes

        # --------------------------------------------------------------------------------
        # (target, context) pair properties
        # --------------------------------------------------------------------------------
        self._target_size = target_size
        self._context_size = context_size

        # --------------------------------------------------------------------------------
        # Negative sampling property
        # --------------------------------------------------------------------------------
        self._negative_sample_size = negative_sample_size

        # --------------------------------------------------------------------------------
        # Dictionary of events that provide event probability, etc.
        # --------------------------------------------------------------------------------
        self._dictionary: EventIndexing = dictionary

        # --------------------------------------------------------------------------------
        # Event vector space
        # Gradient dL/dW varies because only the extracted W rows need to be processed.
        # --------------------------------------------------------------------------------
        if W is None:
            self._W: TYPE_TENSOR = self.build_weights(
                M=dictionary.vocabulary_size,
                D=event_vector_size
            )
        else:
            self._W: TYPE_TENSOR = copy.deepcopy(W)
        assert \
            self.W.shape[0] >= dictionary.vocabulary_size and \
            self.W.shape[1] == event_vector_size,\
            "W shape needs (%s, >=%s) but %s." \
            % (dictionary.vocabulary_size, dictionary.vocabulary_size, self.W.shape)
        self._D = self.W.shape[1]
        self._dW = np.empty(shape=())

        # --------------------------------------------------------------------------------
        # State of the layer
        # --------------------------------------------------------------------------------
        self._S = [self.W]

        # --------------------------------------------------------------------------------
        # Optimizer for gradient descent
        # --------------------------------------------------------------------------------
        assert isinstance(optimizer, optimiser.Optimizer)
        self._optimizer: optimiser.Optimizer = optimizer

        self.logger.debug(
            "Embedding[%s] W.shape is [%s], number of outputs is [%s]",
            name, self.W.shape, self.M
        )

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def _extract_event_vectors(self, X: TYPE_TENSOR):
        """Extract vectors from event vector space W.
        Use numpy 1D array indexing to extract rows from W.
        W[
            [idx, idx, ....]
        ]

        Args:
            X: rank 2 matrix

        Returns: vectors of shape:(N*?, D) where is C or E.
        """
        assert isinstance(self.W, np.ndarray), "Needs numpy array for array indexing"
        vectors = self.reshape(
            self.W[
                self.reshape(X, (-1))
            ],
            (self.N, -1, self.D)
        )
        return vectors

    def _bag_vectors_per_window(self, vectors: TYPE_TENSOR):
        """Sum(vectors) via Group-by window(axis=1) of shape:(N, ?, D)
        where ? is either target or context in (target/E, context/C) pair window.

        Args:
            vectors: Event vectors of shape:(N, C, D) or (N, E, D)
        Returns: Bagged vectors of shape (N, D)
        """
        assert \
            self.tensor_shape(vectors) == (self.N, self.E, self.D) or \
            self.tensor_shape(vectors) == (self.N, self.C, self.D)
        return self.einsum(
            "ncd->nd",
            vectors
        )

    def _bagging(self, X):
        bags = self._bag_vectors_per_window(self._extract_event_vectors(X))
        assert self.tensor_shape(bags) == (self.N, self.D)
        return bags

    def function(self, X: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate the layer output Y
        Process:
        1. BoW (Bag of Words) event vectors for contexts:
            For all the (target, context) pairs:
            1-1. Extract event vectors from W for the 'context' in (target, context) pair.
                 -> context vectors of shape:(R,D)
            1-2. Create a BoW (Bag of words) vector from the context vectors with
                 sum(context_vectors, axis=0)
                 -> a BoW vector of shape:(1,D)
            1-3. Stack all the BoWs into shape:(N, D)

        2. Target event vector for events:
            For all the (target, context) pairs:
            2-1. Extract event vectors from W for the 'target' in (target, context) pair.
               If E > 1, sum the target vectors of shape:(E, D) along axis=0.
               -> a target vector of shape:(1, D).
            2-2. Stack all the targets into shape:(N, D)

        3. Negative samples.
            For all the (target, context) pairs.
            3-1. Sample G negative events from the dictionary. Negative events are
                 events not includes in the (target, context) pair. G is the number of
                 negative samples.
            3-2. Extract event vectors from W for the negative events.
                 -> negative vectors of shape:(G, D).
            3-3. Stack all the negative vectors into shape:(N, G, D)

        Args:
            X:  (event, context) pairs. The shape can be:
                - (num_windows, E+C)    when X has one sequence.
                - (N, num_windows, E+C) when X has multiple sequences.

                num_windows is the number of context windows in a sequence which
                varies per sequence.

        Returns:
            Y: dot products of (context bow) vector against (target, negatives) vectors
               in shape(N, 1+negative_sample_size) where:
               - the first column: scores for true labels
               - reset negative_sample_size columns: scores for false labels

               Hence for all the 1st columns, the label T=1, and 0 for the rest.
        """
        name = "function"
        X = self.assure_tensor(X)
        assert self.tensor_rank(X) in [2, 3], \
            "Expected X shape is (?, E+C) or (N, ?, E+C) but %s" % X.shape

        # --------------------------------------------------------------------------------
        # Stack N x (num_windows, E+C) into (N*num_windows, E+C).
        # The information of which sequence a (event, context) pair belongs to gets lost.
        # However, Embedding only require (event, context) pairs, NOT sequences.
        # --------------------------------------------------------------------------------
        X = self.reshape(X, (-1, self.window_size)) if self.tensor_rank(X) > 2 else X
        self.X = X
        self.logger.debug(
            "layer[%s].%s: X.shape %s W.shape %s",
            self.name, name, self.tensor_shape(self.X), self.tensor_shape(self.W)
        )

        # ================================================================================
        # Score (BoW dot Target) for label=1 (True) classification
        # ================================================================================
        # --------------------------------------------------------------------------------
        # BoW vectors for contexts
        # --------------------------------------------------------------------------------
        bows = self._bagging(X[
            ::,
            self.E:
        ])
        # --------------------------------------------------------------------------------
        # Target vectors
        # --------------------------------------------------------------------------------
        targets = self._bagging(X[
            ::,
            0: self.E
        ])

        # --------------------------------------------------------------------------------
        # Positive score (BoW dot Target)
        # --------------------------------------------------------------------------------
        positive_scores = self.einsum("nd,nd->n", bows, targets)
        del targets

        # ================================================================================
        # Score (BoW dot Negative) for label=0 (False) classification
        # ================================================================================
        # --------------------------------------------------------------------------------
        # Negative samples
        # --------------------------------------------------------------------------------
        negatives = self._extract_event_vectors(self.to_tensor([
            self.dictionary.negative_sample_indices(size=self.negative_sample_size, excludes=X[row])
            for row in range(self.N)
        ]))
        assert self.tensor_shape(negatives) == (self.N, self.negative_sample_size, self.D), \
            "Negative samples \n%s\nexpected shape %s but %s" % \
            (negatives, (self.N, self.negative_sample_size, self.D), self.tensor_shape(negatives))

        # --------------------------------------------------------------------------------
        # Negative score (BoW dot Negatives)
        # --------------------------------------------------------------------------------
        negative_scores = self.einsum("nd,nkd->nk", bows, negatives)
        del bows, negatives

        # ================================================================================
        # Result of np.c_[positive_scores, negative_scores)
        # ================================================================================
        self._Y = np.c_[positive_scores, negative_scores]
        assert self.tensor_shape(self.Y) == (self.N, (1 + self.negative_sample_size))
        assert np.all(self.is_finite(self.Y)), f"NaN or inf detected in {self.Y}"
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
        if self.is_tensor(self._W) and self.tensor_shape(self._W) == state[0].shape:
            np.copyto(self._W, state[0])
        else:
            del self._W
            self._W = state[0]
