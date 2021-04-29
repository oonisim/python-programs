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
from memory_profiler import profile as memory_profile

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
from layer.base import Layer
from layer.constants import (
    MAX_NEGATIVE_SAMPLE_SIZE,
)
from layer.preprocessing import (
    EventIndexing
)
import optimizer as optimiser


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
    # Instance properties (for input X)
    # --------------------------------------------------------------------------------
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
    def target_indices(self) -> TYPE_TENSOR:
        """Indices to extract target event vectors"""
        assert self.tensor_shape(self._target_indices) == (self.N*self.E,), \
            "Indices shape must be %s but %s" \
            % ((self.N*self.E,), self.tensor_shape(self._target_indices))
        return self._target_indices

    @property
    def context_indices(self) -> TYPE_TENSOR:
        """Indices to extract context event vectors"""
        assert self.tensor_shape(self._context_indices) == (self.N*self.C,), \
            "Indices shape must be %s but %s" \
            % ((self.N*self.C,), self.tensor_shape(self._context_indices))
        return self._context_indices

    # --------------------------------------------------------------------------------
    # Instance properties (for negative sampling)
    # --------------------------------------------------------------------------------
    @property
    def dictionary(self) -> EventIndexing:
        """Event dictionary"""
        return self._dictionary

    @property
    def V(self) -> TYPE_INT:
        """vocabulary size."""
        return self.dictionary.vocabulary_size

    @property
    def SL(self) -> TYPE_INT:
        """Negative sample size"""
        return self._negative_sample_size

    @property
    def negative_sample_indices(self) -> TYPE_TENSOR:
        """Indices to extract negative sample event vectors"""
        shape = (self.N*self.SL,)
        assert \
            self.tensor_shape(self._negative_sample_indices) == shape, \
            "Indices shape must be %s but %s" \
            % (shape, self.tensor_shape(self._negative_sample_indices))
        return self._negative_sample_indices

    # --------------------------------------------------------------------------------
    # Instance properties (for event vectors)
    # --------------------------------------------------------------------------------
    @property
    def W(self) -> TYPE_TENSOR:
        """Layer weight vectors W"""
        return self._W

    @property
    def dW(self) -> TYPE_TENSOR:
        """Weight gradients as np.c_[dL/dWe, dL/dWc, dL/dWs]
        dL/dWe:shape(N, E, D)
        dL/dWc:shape(N, C, D)
        dL/dWs:shape(N, SL, D)
        """
        return np.c_[self.dWe, self.dWc, self.dWs]

    @property
    def We(self) -> TYPE_TENSOR:
        """Event vector for target"""
        assert self.tensor_shape(self._We) == (self.N, self.E, self.D), \
            "We is not initialized or deleted"
        return self._We

    @property
    def Be(self) -> TYPE_TENSOR:
        """BoW of event vector for targets"""
        assert self.tensor_shape(self._Be) == (self.N, self.D)
        return self._Be

    @property
    def dWe(self) -> TYPE_TENSOR:
        """Gradients dL/dWe for target event vectors We
        dL/dWe:shape(N, E, D)
        """
        assert self.tensor_size(self._dWe) == (self.N * self.E * self.D), \
            "dWe is not initialized or invalid"
        return self._dWe

    @property
    def Wc(self) -> TYPE_TENSOR:
        """Event vector for context"""
        assert self.tensor_shape(self._Wc) == (self.N, self.C, self.D), \
            "Wc is not initialized or deleted"
        return self._Wc

    @property
    def Bc(self) -> TYPE_TENSOR:
        """BoW of event vector for context"""
        assert self.tensor_shape(self._Bc) == (self.N, self.D)
        return self._Bc

    @property
    def dWc(self) -> TYPE_TENSOR:
        """Gradients combined dL/dWc01 with Be and dL/dWc02 with Ws
        dL/dWc:shape(N, C, D)

        Gradient descent on Wc is subtraction of dL/dWc01 and dL/dWc02
        in a linear manner. The impact on L from Wc=(dL/dWc01 and dL/dWc02).
        """
        assert self.tensor_shape(self._dWc) == (self.N, self.C, self.D), \
            "dWc is not initialized or invalid"
        return self._dWc

    @property
    def dWc01(self) -> TYPE_TENSOR:
        """Gradients dL/dWc01 for context event vectors Wc with Be
        (BoW of target event vectors)
        dL/dWc01:shape(N, C, D)
        """
        assert self.tensor_shape(self._dWc01) == (self.N, self.C, self.D), \
            "dWc01 is not initialized or invalid"
        return self._dWc01

    @property
    def dWc02(self) -> TYPE_TENSOR:
        """Gradients dL/dWc02 for context event vectors Wc with Ws
        (Negative sample event vectors)
        dL/dWc02:shape(N, C, D)
        """
        assert self.tensor_shape(self._dWc02) == (self.N, self.C, self.D), \
            "dWc02 is not initialized or invalid"
        return self._dWc02

    @property
    def Ws(self) -> TYPE_TENSOR:
        """Event vector for negative samples"""
        assert self.tensor_shape(self._Ws) == (self.N, self.SL, self.D), \
            "Ws is not initialized or deleted"
        return self._Ws

    @property
    def dWs(self) -> TYPE_TENSOR:
        """Gradients dL/dWs for negative sample event vectors Ws
        dL/dWs:shape(N, SL, D)
        """
        assert self.tensor_size(self._dWs) == (self.N * self.SL * self.D), \
            "dWs is not initialized or invalid"
        return self._dWs

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

    @property
    def state_elements(self):
        return [
            "name",
            "target_size",
            "context_size",
            "negative_sample_size",
            "dictionary",
            "W",
            "event_vector_size",
            "optimizer"
        ]

    @property
    def S(self) -> Union[List, Dict]:
        """State of the layer instance"""
        self._S = {
            "name": self.name,
            "target_size": self._target_size,
            "context_size": self._context_size,
            "negative_sample_size": self._negative_sample_size,
            "dictionary": self.dictionary,
            "W": self.W,
            "event_vector_size": self.D,
            "optimizer": self.optimizer
        }
        assert set(self._S.keys()) == set(self.state_elements)
        return self._S

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
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
        assert 0 == (context_size % 2) < context_size
        # target_size >= context_size in SkipGram, eg. (target=i,am,red,cat), context=(a)
        # assert 0 < target_size <= context_size
        assert 0 < target_size
        assert isinstance(dictionary, EventIndexing)
        availability_for_negatives = (
            dictionary.vocabulary_size
            - (context_size+target_size)
            - len(EVENT_META_ENTITIES)
        )
        assert \
            0 < negative_sample_size <= MAX_NEGATIVE_SAMPLE_SIZE and \
            0 < negative_sample_size <= availability_for_negatives
        assert isinstance(optimizer, optimiser.Optimizer)

        # --------------------------------------------------------------------------------
        # Number of outputs from the layer.
        # M='1' because Negative Sampling is Logistic regression.
        # The output Y:shape is (N,E+C) which mis-aligns with what the LogLoss expects,
        # hence needs an adapter for shape transformation (N,E+C) <-> (N*(E+C),).
        # --------------------------------------------------------------------------------
        self._M = num_nodes

        # --------------------------------------------------------------------------------
        # (target, context) pair properties
        # --------------------------------------------------------------------------------
        self._X_rank: TYPE_INT = TYPE_INT(0)                # Original rank of input X
        self._target_size: TYPE_INT = target_size
        self._target_indices: TYPE_TENSOR = np.empty(())    # 1D array
        self._context_size: TYPE_INT = context_size
        self._context_indices: TYPE_TENSOR = np.empty(())

        # --------------------------------------------------------------------------------
        # Negative sampling property
        # --------------------------------------------------------------------------------
        self._dictionary: EventIndexing = dictionary
        self._negative_sample_size: TYPE_TENSOR = negative_sample_size
        self._negative_sample_indices: TYPE_TENSOR = np.empty(())

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

        assert isinstance(self._W, np.ndarray) and self.is_float_tensor(self._W), \
            "Needs NumPy array for array indexing"
        assert \
            self.tensor_shape(self.W)[0] >= dictionary.vocabulary_size and \
            self.tensor_shape(self.W)[1] == event_vector_size > 0 and \
            self.tensor_size(self.W) >= dictionary.vocabulary_size * event_vector_size, \
            "W shape needs (%s,%s) but %s." \
            % (dictionary.vocabulary_size, event_vector_size, self.W.shape)

        self._D = self.W.shape[1]

        self._We: TYPE_TENSOR = np.empty(shape=())      # Event vectors for target
        self._Be: TYPE_TENSOR = np.empty(shape=())      # BoW of event vectors for targets
        self._dWe: TYPE_TENSOR = np.empty(shape=())     # dL/dWe

        self._Wc: TYPE_TENSOR = np.empty(shape=())      # Event vectors for context
        self._Bc: TYPE_TENSOR = np.empty(shape=())      # BoW of event vectors for contexts
        self._dWc: TYPE_TENSOR = np.empty(shape=())     # dL/dWc:(N,C,D)=(dL/dWc01 + dL/dWc02).
        self._dWc01: TYPE_TENSOR = np.empty(shape=())   # dL/dWc01:(N,C,D) with Be (BoWs of target event vectors).
        self._dWc02: TYPE_TENSOR = np.empty(shape=())   # dL/dWc02:(N,S,D) with Ws (negative sample event vectors).

        self._Ws: TYPE_TENSOR = np.empty(shape=())      # Event vectors for negative samples
        self._dWs: TYPE_TENSOR = np.empty(shape=())     # dL/dWs

        # --------------------------------------------------------------------------------
        # Optimizer for gradient descent
        # --------------------------------------------------------------------------------
        self._optimizer: optimiser.Optimizer = optimizer

        # --------------------------------------------------------------------------------
        # State of the layer
        # --------------------------------------------------------------------------------
        self._S = {}

        self.logger.debug(
            "Embedding[%s]: W.shape %s, number of outputs [%s], "
            "target size [%s], context size [%s] negative sample size [%s]\n",
            name, self.W.shape, self.M, self.E, self.C, self.SL
        )

    # --------------------------------------------------------------------------------
    # Instance methods
    # --------------------------------------------------------------------------------
    def _extract_event_vectors(self, indices: TYPE_TENSOR):
        """Extract vectors from event vector space W.
        Use numpy 1D array indexing to extract rows from W.
        W[
            [idx, idx, ....]
        ]

        Args:
            X: rank 1 vector

        Returns: vectors of shape:(N*?, D) where is C or E.
        """
        vectors = self.W[indices]
        return vectors

    def _bagging(self, w) -> TYPE_TENSOR:
        """Create a bag of event vectors by sum along axis=1 as einsum("ncd->nd").
        Like SQL sum(context_vectors) GROUP BY context causing (N, ?, D)->(N,D)
        where ? is either target size E or context size C.

        Args:
            w: Event vectors of shape:(N, C, D) or (N, E, D)
        Returns: Bagged vectors of shape (N, D)
        """
        return self.einsum(
            "ncd->nd",
            w
        )

    @memory_profile
    def function(self, X: Union[TYPE_TENSOR, TYPE_FLOAT]) -> TYPE_TENSOR:
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
            3-1. Sample SL number of negative events from the dictionary.
                 Negative events are those not includes in the (target, context) pair.
                 SL is the number of negative samples.
            3-2. Extract event vectors from W for the negative events.
                 -> negative vectors of shape:(SL, D).
            3-3. Stack all the negative vectors into shape:(N, SL, D)

        Args:
            X:  (event, context) pairs. The expected shape is either:
                - Rank 2: (num_windows, E+C)    when X has one sequence.
                - Rank 3: (N, num_windows, E+C) when X has multiple sequences.

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
        expected_ranks = [2,3]
        assert self.is_tensor(X) and self.tensor_rank(X) in expected_ranks, \
            "Expected X rank is %s but %s" % (expected_ranks, X.shape)

        # --------------------------------------------------------------------------------
        # Reshape (N, num_windows, E+C) into (N*num_windows, E+C) if rank(X) > 2.
        # Knowledge about "to which sequence a (event, context) pair belongs" gets lost.
        # However, Embedding only encodes (event, context) pairs, and sequence knowledge
        # is not utilised. Hence reshaping has no impact on Embedding capability.
        #
        # Make sure to restore the original X.shape for dX.
        # --------------------------------------------------------------------------------
        self._X_rank = self.tensor_rank(X)
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
        # BoW vectors for target
        # --------------------------------------------------------------------------------
        self._target_indices = self.reshape(X=X[
            ::,
            0: self.E
        ], shape=(-1))
        self._We = self.reshape(
            X=self._extract_event_vectors(self.target_indices),
            shape=(self.N, self.E, self.D)
        )
        self._Be = self._bagging(self.We)

        # --------------------------------------------------------------------------------
        # BoW vectors for contexts
        # --------------------------------------------------------------------------------
        self._context_indices = self.reshape(X=X[
            ::,
            self.E: self.window_size
        ], shape=(-1))
        self._Wc = self.reshape(
            X=self._extract_event_vectors(self.context_indices),
            shape=(self.N, self.C, self.D)
        )
        self._Bc = self._bagging(self.Wc)

        # --------------------------------------------------------------------------------
        # Positive score (BoW dot Target)
        # --------------------------------------------------------------------------------
        Ye = positive_scores = self.einsum("nd,nd->n", self.Bc, self.Be)

        # ================================================================================
        # Score (BoW dot Negative) for label=0 (False) classification
        # ================================================================================
        # --------------------------------------------------------------------------------
        # Event vectors of negative samples
        # --------------------------------------------------------------------------------
        self._negative_sample_indices = self.reshape(X=self.to_tensor([
            self.dictionary.negative_sample_indices(
                size=self.SL,
                excludes=X[row]
            )
            for row in range(self.N)
        ]), shape=(-1))
        self._Ws = self.reshape(
            X=self._extract_event_vectors(self.negative_sample_indices),
            shape=(self.N, self.SL, self.D)
        )

        # --------------------------------------------------------------------------------
        # Negative score (BoW dot Negatives)
        # --------------------------------------------------------------------------------
        Ys = negative_scores = self.einsum("nd,nsd->ns", self.Bc, self.Ws)

        # ================================================================================
        # Result of np.c_[Ye, Ys]
        # ================================================================================
        self._Y = np.c_[Ye, Ys]
        del Ye, Ys
        assert \
            self.is_finite(self.Y) and \
            self.tensor_shape(self.Y) == (self.N, (1 + self.SL))
        assert self.is_finite(self.Y), f"NaN or inf detected in {self.Y}"
        return self.Y

    @memory_profile
    def gradient(self, dY: Union[TYPE_TENSOR, TYPE_FLOAT]) -> TYPE_TENSOR:
        """Calculate the gradients dL/dWe, dL/dWc, dL/dWs
        dL/dWe = dL/dW[target_indices] = dL/dYe * Bc
        dL/dWc = dL/dW[target_indices] = dL/dYe * Be

        dL/dYe=dL/dY[0:1] is the gradient of target event vectors We=W[target_indices].
        dL/dYc=dL/dY[1:] is the gradient of negative sample vectors Ws=W[negative_sample_indices].

        Args:
            dY: Gradient dL/dY
        Returns: Return the input X as is as there is no gradient of X
        """
        name = "gradient"
        assert \
            self.is_float_tensor(dY) and \
            self.tensor_shape(dY) == self.tensor_shape(self.Y), \
            "Need dY of float tensor in shape %s but got type %s instance \n%s\n" \
            % (self.tensor_shape(self.Y), type(dY), dY)

        self.dY = dY
        self.logger.debug("layer[%s].%s: dY.shape %s", self.name, name, dY.shape)

        dYe: TYPE_TENSOR = dY[
            ::,
            0:1     # To make dYe:(N,1), NOT (N,) so as to multiply (N,1) * (N,D)
        ]
        assert self.tensor_shape(dYe) == (self.N, 1)    # Make sure dYe:(N,1), NOT (N,)
        dYs: TYPE_TENSOR = dY[
            ::,
            1:
        ]
        assert self.tensor_shape(dYs) == (self.N, self.SL)

        # --------------------------------------------------------------------------------
        # dL/dWe:(N,E,D) = dL/dBe:(N,D) OP dBe/dWe:(N,E,D)
        # - dL/dBe:(N,D) = dL/dYe:(N,1) * dYe/dBe:(N,D)
        # - dYe/dBe:(N,D) = Bc:(N,D)
        # - dBe/dWe:(N,E,D) = I:(N,E,D) = ones(shape=(N,E,D))
        # --------------------------------------------------------------------------------
        # Forward path:
        # F1. Be:(N,D) = einsum("ned->nd", We:(N,E,D))
        #   F1-1. Summed itself along axis 1 and,
        #   F1-2. Dropped the axis 1 (rank -1)
        # F2. Ye = einsum("nd,nd->n", Be:(N,D), Bc:(N,D))
        #   F2-1. Element-multiply along axis -1.
        #   F2-2. Sum along axis -1.
        #   F2-3. Drop the axis -1.
        #
        # B1. Gradient dL/dBe:(N,D)
        #   dL/dBe:(N,D) = dL/dYe:(N,1) * dYe/dBe:(N,D)
        #                = dL/dYe:(N,1) * Bc:(N,D)
        #   dL/dYe is (N,1), NOT (N,) because of dY[::,0:1] instead of dY[::,0].
        # B2. Gradient dL/dWe:(N,E,D) = dL/dBe:(N,D) OP I:(N,E,D)
        #   B2-1. [OP]
        #       Restore the rank of We.rank=3 by adding axis 1 dropped at F1-2 in
        #       the forward path with either reshape (N,D)->(N,1,D), or newaxis.
        #       dL/dBe:(N,1,D) = reshape(dL/dBe:(N,D), shape=(N,1,D))
        #   B2-2. multiply with dBe/dWe:(N,E,D) = I:(N,E,D)
        #       dL/dWe:(N,E,D) = dL/dBe:(N,1,D) * I:(N,E,D)
        #
        # Steps in B are reversing "ned->ed" as "ed->ned".
        # --------------------------------------------------------------------------------
        dBe: TYPE_TENSOR = self.multiply(dYe, self.Bc)  # B1
        assert self.tensor_shape(dBe) == (self.N, self.D), \
            f"Expected dBe shape {(self.N, self.D)} but {self.tensor_shape(dBe)}"
        self._dWe = self.multiply(
            x=self.reshape(X=dBe, shape=(self.N, 1, self.D)),   # B2-1
            y=np.ones(shape=(self.N, self.E, self.D))           # B2-2
        )
        del dBe

        # --------------------------------------------------------------------------------
        # dL/dWc01:(N,C,D) for the forward path with Be.
        # --------------------------------------------------------------------------------
        # dL/dWc01:(N,C,D) = dL/dBc01:(N,D) OP dBc01/dWc01:(N,C,D)
        # - dL/dBc01:(N,D) = dL/dYe:(N,D) * dYe/dBc01:(N,D)
        #                  = dL/dYe:(N,D) * Be:(N,D)
        # - dBc/dWc:(N,C,D) = I:(N,C,D)
        #
        # Forward path:
        # F1. Bc = einsum("ncd->nd", Wc)
        #   F1-1. Sum along axis 1 (bag of word vectors) as "ncd->n1d".
        #   F2-2. Rank -1 by dropping axis 1 as "n1d->nd".
        # F2. Ye = einsum("nd,nd->n", Be, Bc)
        #   F2-1. Element-wise multiplication along axis -1. "nd,nd->nd"
        #   F2-2. Sum along axis -1.                         "nd->n1"
        #   F2-3. Rank -1 by dropping axis -1.               "n1->n"
        #
        # B1. Gradient dL/dBc01:(N,D) = dL/dYe:(N,1) * Be:(N,D)
        #     dL/dYe is (N,1), NOT (N,) because of dY[::,0:1] instead of dY[::,0],
        #     which corresponds to reversing F2-3 as "n->n1".
        #     Then implicitly restore the shape (N,D) from (N,1) using broadcast,
        #     which corresponds to reversing F2-2 as "n1->nd".
        # B2. Gradient dL/dWc:(N,C,D) = dL/dBc01:(N,D) OP dBc01/dWc01:(N,C,D)
        #   B2-1. dL/dBc01:(N,1,D) = reshape(dL/dBc01:(N,D), shape=(N,1,D))
        #       Restore the rank of Wc by adding the axis 1, which corresponds
        #       to reversing "n1d->nd" as "nd->n1d".
        #   B2-2. dL/dWc01:(N,C,D) = dL/dBc01:(N,1,D) * dBc01/dWc01:(N,C,D)
        #                          = dL/dBc01:(N,1,D) * I:(N,C,D)
        #       Restore the Wc shape (N,C,D) by multiplying with I:(N,C,D),
        #       which corresponds to reversing "ncd->n1d" as "n1d->ncd".
        # --------------------------------------------------------------------------------
        dBc01: TYPE_TENSOR = self.multiply(dYe, self.Be)    # B1
        assert self.tensor_shape(dBc01) == self.tensor_shape(self.Bc) == (self.N, self.D),\
            f"Expected dBc shape {(self.N, self.D)} but {self.tensor_shape(dBc01)}"

        self._dWc01 = self.multiply(
            x=self.reshape(X=dBc01, shape=(self.N, 1, self.D)),     # B2-1
            y=np.ones(shape=(self.N, self.C, self.D))               # B2-2
        )
        del dBc01

        # --------------------------------------------------------------------------------
        # dL/dWc02:(N,C,D) for the forward path with Ws.
        # --------------------------------------------------------------------------------
        # dL/dWc02:(N,C,D) = dL/dBc02:(N,D) OP2 dBc/dWc:(N,C,D)
        # - dL/dBc02:(N,D) = dL/dYs:(N,SL) OP1 dYs/dBc:(N,SL,D)
        # - dYs/dBc:(N,SL,D) = Ws:(N,SL,D)
        # - dBc/dWc:(N,C,D) = I:((N,C,D))
        #
        # Forward path:
        # F1. Bc:(N,D) = einsum("ncd->nd", Wc)
        #   F1-1. Sum along axis 1 (bag of word vectors) as "ncd->n1d".
        #   F2-2. Rank -1 by dropping axis 1 as "n1d->nd".
        # F2. Ys:(N,SL) = einsum("nd,nsd->ns", Bc, Ws)
        #   F2-1. Implicit broadcast                         "nd,nsd"->"nsd,nsd".
        #   F2-2. Element-wise multiplication along axis -1. "nsd,nsd->nsd"
        #   F2-3. Sum along axis -1.                         "nsd->ns1"
        #   F2-4. Rank -1 by dropping axis -1.               "ns1->ns"
        #
        # B1. Gradient dL/dBc02:(N,D) = dL/dYs:(N,SL) OP1 Ws:(N,SL,D)
        #   B1-1. dL/dYs:(N,SL,1) = reshape(dL/dYs:(N,SL), shape=(N,SL,1))
        #       Restore the rank of Ws by adding the axis -1 as (N,SL)->(N,SL,1),
        #       which corresponds to reversing F2-4 as "ns->ns1".
        #
        #   B2-2. dL/dYs:(N,SL,D) = dL/dYs:(N,SL,1) * I:((N,SL,D))
        #       Restore the shape of Ws as (N,SL,1)->(N,SL,D) by multiplying
        #       with I:((N,SL,D)), which corresponds to reversing F2-3 as
        #       "ns1->nsd".
        #       This step can be omitted by using implicit broadcast.
        #
        #   B3-3. dL/dBc02:(N,SL,D) = dL/dYs:(N,SL,D) * Ws:(N,SL,D)
        #       Element-wise amplify gradients dL/dYs:(N,SL,D) with Ws:(N,SL,D).
        #
        #       Suppose there is only one negative sample (SL=1). Then the
        #       impact by dBc(n)(d) on Ys(n)(s=0) is amplified by Ws(n)(s=0).
        #       In reality as there is SL number of samples, There are SL number
        #       of impacts by dBc(n)(d) on Ys(n)(s=0),Ys(n)(s=1),...,Ys(n)(s=SL-1).
        #       Hence, the gradient dL/dBC02:(N,SL,D) is element-wise multiplied
        #       with Ws:(N,SL,D). This corresponds to F2-2, but not reversing it.
        #
        #   B3-4. dL/dBc02:(N,D) = einsum("nsd->nd")
        #       There are SL number of impacts by dBc(n)(d) on Ys(n)(s=0),
        #       Ys(n)(s=1), ..., Ys(n)(s=SL-1). However, there is only one
        #       dBc(n)(d) to update via the gradient descent. Hence sum up the
        #       SL number of impacts into one. This corresponding to reversing
        #       F2-1 as "nsd,nsd"->"nd,nsd".
        #
        #       Consideration of required about the variability of the SL value.
        #       In practice, SL is in-between 5-20 but could be more. Perhaps
        #       better normalize the impact dL/dBc01 so that the actual size
        #       of SL would not impact the model training.
        #
        #       The normalization is not implemented in this layer because each
        #       's' of Ys(n)(s) has a corresponding negative label t(n)(s)=0,
        #       and each Ys(n)(s) for s is an independent value which is not
        #       impacted by SL. Only the gradient dL/dBc01 needs to incorporate
        #       the SL normalization. Same with the normalization for the batch
        #       size N, handle it at the loss layer L(Y's:(N*SL,), T:(N*SL,)).
        #       (N*SL) because the loss layer is logistic los loss and the
        #       number of outputs from the layer is 1.
        #
        # B2. Gradient dL/dWc02:(N,C,D) = dL/dBc02:(N,D) OP2 dBc/dWc:(N,C,D)
        #   B2-1. dL/dBc02:(N,1,D) = reshape(dL/dBc01:(N,D), shape=(N,1,D))
        #       Restore the rank of Wc by adding the axis 1, which corresponds
        #       to reversing "n1d->nd" as "nd->n1d".
        #   B2-2. dL/dWc02:(N,C,D) = dL/dBc02:(N,1,D) * dBc02/dWc02:(N,C,D)
        #                          = dL/dBc02:(N,1,D) * I:(N,C,D)
        #       Restore the Wc shape (N,C,D) by multiplying with I:(N,C,D),
        #       which corresponds to reversing "ncd->n1d" as "n1d->ncd".
        # --------------------------------------------------------------------------------
        dBc02: TYPE_TENSOR = self.einsum("ns,nsd->nd", dYs, self.Ws)    # B1 (all)
        assert self.tensor_shape(dBc02) == (self.N, self.D),\
            f"Expected dBc shape {(self.N, self.D)} but {self.tensor_shape(dBc02)}"

        self._dWc02 = self.multiply(
            x=self.reshape(X=dBc02, shape=(self.N, 1, self.D)),     # B2-1
            y=np.ones((self.N, self.C, self.D))                     # B2-2
        )
        del dBc02

        # --------------------------------------------------------------------------------
        # dL/dWc: Combined gradient of Wc.
        # Impact of Wc on L is a linear combination of dL/dWc01 and dL/dWc02.
        # --------------------------------------------------------------------------------
        self._dWc = self.add(self._dWc01, self._dWc02)
        del self._dWc01, self._dWc02

        # --------------------------------------------------------------------------------
        # dL/dWs:(N,SL,D) = OP1(dL/dYs:(N,SL)) * OP2(Bc:(N,D))
        # Ys:(N,SL)=einsum("nd,nsd->ns", Be:(N,D), Ws:(N,SL,D)).
        # With respect to Ws, it has done:
        #   (1). Broadcast/reshape Bc(n):(D) into Bc(n):(SL,D)
        #   (2). Amplify sample event vectors in Ws(n):(SL,D) by Bc(n):(SL,D).
        #   (3). Sum along d.
        #
        # To calculate dYs/dWs of einsum("nd,nsd->ns", Bc:(N,D), Ws:(N,SL,D)):
        # [OP1]
        #   1. Restore the rank for 'd' lost during "nd,nsd->ns" at the step (3)
        #      with reshape(dL/dYs:(N,SL), shape=(N, SL, 1)).
        #      This is "ns->n1d" as the inverse of "nd,nsd->nd" with regard to the rank.
        #   2. Restore the shape (N,SL,D) lost during "nsd->ns" at the step (3)
        #      with element-wise multiplication with ones(shape=(N,SL,D)).
        #      This is "n1d->nsd" as the inverse of "nd,nsd->ns".
        #   With step 1 and 2,
        #   dL/dYs:(N,SL,D)
        #      = OP1(dL/dYs)
        #      = reshape(dL/dYs:(N,SL), shape=(N, SL, 1)) * ones(shape=(N,SL,D))
        #
        # [OP2]
        # To amplify OP1(dL/dYs):(N,SL,D) with Be:(N,D), add the axis=1 to Be as:
        #   Be:(N,1,D)
        #      = OP2(Be:(N,D))
        #      = reshape(Be:(N,D), shape=(N,1,D)).
        #
        # Then:
        # dL/dWs:(N,SL,D) = multiply(OP1(dL/dYs):(N,SL,D), OP2(Be):(N,1,D))
        # --------------------------------------------------------------------------------
        self._dWs = self.multiply(
            x=self.multiply(    # OP1
                self.reshape(dYs, shape=(self.N, self.SL, 1)),
                np.ones(shape=(self.N, self.SL, self.D))
            ),
            y=self.reshape(self.Be, shape=(self.N, 1, self.D))
        )
        assert self.is_finite(self.dWs), "NaN/inf in \n%s\n" % self.dWs
        assert \
            self.tensor_shape(self.dWs) == \
            self.tensor_shape(self.Ws) == \
            (self.N, 1, self.D), \
            "Expected shape %s but dWs.shape %s Ws.shape %s " \
            % ((self.N, 1, self.D), self.tensor_shape(self.dWs), self.tensor_shape(self.Ws))

        # --------------------------------------------------------------------------------
        # TODO: Need to consider what to return as dX, because X has no gradient.
        # What to set to self._dx? Use empty for now and monitor how it goes.
        # 'self._dX = self.X' is incorrect and can cause unexpected result e.g.
        # trying to copy data into self.X which can cause unexpected effects.
        # --------------------------------------------------------------------------------
        self._dX = np.empty(shape=(self.N, -1, self.window_size)) \
            if self._X_rank == 3 else np.empty(shape=(self.N, self.window_size))
        return self.X

    def gradient_numerical(
            self, h: Optional[TYPE_FLOAT] = None
    ) -> List[Union[TYPE_TENSOR, TYPE_FLOAT]]:
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

        def objective_Ws(ws: TYPE_TENSOR):
            ys = self.einsum("nd,nsd->ns", self.Bc, ws)
            return self.objective(ys)

        def objective_We(we: TYPE_TENSOR):
            ye = self.einsum("ncd,nd->n", self.Bc, we)
            return self.objective(ye)

        def objective_Wc(wc: TYPE_TENSOR):
            raise NotImplementedError("TBD")

        dWe = numerical_jacobian(objective_We, self.We, delta=h)
        dWc = numerical_jacobian(objective_Wc, self.Wc, delta=h)
        dWs = numerical_jacobian(objective_Ws, self.Ws, delta=h)
        return [dWe, dWc, dWs]

    def _gradient_descent(
            self, w: TYPE_TENSOR, dw: TYPE_TENSOR, indices
    ):
        """Gradient descent on event vector w
        Update the W elements extracted with the indices at the forward path.
        """
        differential: TYPE_TENSOR = self.optimizer.differential(dW=dw)
        np.subtract.at(a=w, indices=indices, b=differential)

    def update(self) -> List[Union[float, np.ndarray]]:
        """
        Responsibility: Update layer state with gradient descent.

        1. Calculate dL/dW = (dL/dY * dY/dW).
        2. Update W with the optimizer.

        Returns:
            [dL/dW]: List of dL/dW.
            dW is dL/dW of shape (M, D) including the bias weight w0.

        Note:
            update() is to update the state of the layer S. Hence not
            include dL/dX which is not part of the layer state.
       """
        self._gradient_descent(w=self.We, dw=self.dWe, indices=self.target_indices)
        self._gradient_descent(w=self.Wc, dw=self.dWc, indices=self.context_indices)
        self._gradient_descent(w=self.Ws, dw=self.dWs, indices=self.negative_sample_indices)
        self._dS = [self.dWe, self.dWc, self.dWs]

        # self.N changes every time function() is called, hence the buffer
        # cannot be re-usable as the out buffer. Free them.
        del self._We, self._Wc, self._Ws, self._Be, self._Bc

        return self.dS

    def load(self, path: str) -> Dict:
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
        assert \
            isinstance(state, dict) \
            and len(state) == len(self.state_elements) \
            and all(element in state for element in self.state_elements)

        if self.dictionary is not None:
            del self._dictionary
        if self.W is not None:
            del self._W
        if self.optimizer is not None:
            del self._optimizer

        self.__init__(
            name=state["name"],
            num_nodes=TYPE_INT(1),
            target_size=state["target_size"],
            context_size=state["context_size"],
            negative_sample_size=state["negative_sample_size"],
            event_vector_size=state["event_vector_size"],
            dictionary=state["dictionary"],
            W=state["W"],
            optimizer=state["optimizer"]
        )

        return self.S
