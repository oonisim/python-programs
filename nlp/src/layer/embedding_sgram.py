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
import tensorflow as tf
from memory_profiler import profile as memory_profile

from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR,
    EVENT_VECTOR_SIZE,
    EVENT_META_ENTITIES,
    EVENT_META_ENTITIES_COUNT,
    EVENT_INDEX_META_ENTITIES
)
from common.constant import (
    EVENT_INDEX_NIL,
    EVENT_INDEX_UNK
)
from layer.base import Layer
from layer.objective import (
    CrossEntropyLogLoss
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
    X can be stacked along axis 0. This has been done at function(X).

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

    Independence of (n) and (s)
    --------------------
    (n), (s), or (n)(s) element are independent and must not be mixed.
    * Be(n=0) and Be(n=1) are independent.
    * Bc(n=0) and Bc(n=1) are independent.
    During the forward/backward paths e.g. Yc = f(Be, Wc), elements at (n)(d)
    interacts with the same 'n' value only, e.g. Yc=einsum("nd,ncd->nc", Be, Wc)
    where Be(n=0) only interact with Wc(n=0), never with Wc(n=i) where i != 0.

    (s) is to index negative samples. Each sample is independent and has its
    own label when calculated with Wc.

    (n) is to index a sequence in X. Each sequence is independent and does not
    interact with other sequence, because the embedding algorithm only works
    WITHIN an sequence only by calculating (target, context) WITHIN a sequence
    for true label, and (sample(s), context) for each independent sample(s)
    for the context WITHIN the sequence of 'target' only for false labels.
    """
    # ================================================================================
    # Class
    # ================================================================================
    # --------------------------------------------------------------------------------
    # EventEmbedding negative sampling size
    # --------------------------------------------------------------------------------
    MAX_NEGATIVE_SAMPLE_SIZE = 20
    MAX_TARGET_SIZE = 5

    # --------------------------------------------------------------------------------
    # Static methods
    # --------------------------------------------------------------------------------
    @staticmethod
    def specification_template():
        raise NotImplementedError("TBD")

    @staticmethod
    def specification(
            name: str,
            num_nodes: int,
            num_features: int,
            weights_initialization_scheme: str = "normal",
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
    def WO(self) -> TYPE_TENSOR:
        """Layer weight vectors W to take event and sample vectors"""
        return self._WO

    @property
    def dW(self) -> TYPE_TENSOR:
        """Weight gradients as np.c_[dL/dWe, dL/dWc, dL/dWs]
        dL/dWe:shape(N, E, D)
        dL/dWc:shape(N, C, D)
        dL/dWs:shape(N, SL, D)
        """
        return self.concat([self.dWe, self.dWc, self.dWs], axis=1)

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
        assert self.tensor_shape(self._dWe) == (self.N, self.E, self.D), \
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
        assert self.tensor_shape(self._dWs) == (self.N, self.SL, self.D), \
            "dWs is not initialized or invalid"
        return self._dWs

    @property
    def U(self) -> TYPE_TENSOR:
        """Normalized W=W/|W| as NumPy array to use numpy indexing to extract
        scattered elements.
        TODO:
            Use framework native element extraction, e.g. gather_nd with
            [[index0, 1], [index1,1],...]  for TF to extract from W:(V,D).
        """
        eps = 1e-8
        if self.tensor_size(self._U) <= 0:
            self._U = self.divide(
                x=self.W,
                y=self.reshape(
                    self.sqrt(self.einsum("vd,vd->v", self.W, self.W)) + eps,
                    shape=(self.tensor_shape(self.W)[0], 1)
                )
            )
            self._U = self._U.numpy() if tf.is_tensor(self._U) else self._U

        assert self.tensor_shape(self._U) == self.tensor_shape(self._W), \
            "Expected shape %s but %s" %\
            (self.tensor_shape(self._W), self.tensor_shape(self._U))
        return self._U

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
            "event_vector_size",
            "W",
            "WO",
        ]

    @property
    def S(self) -> Union[List, Dict]:
        """State of the layer instance"""
        self._S = {
            "name": self.name,
            "target_size": self._target_size,
            "context_size": self._context_size,
            "negative_sample_size": self._negative_sample_size,
            "event_vector_size": self.D,
            "W": self.W,
            "WO": self.WO,
        }
        assert set(self._S.keys()) == set(self.state_elements)
        return self._S

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            num_nodes: TYPE_INT = TYPE_INT(1),
            target_size: TYPE_INT = TYPE_INT(1),
            context_size: TYPE_INT = TYPE_INT(4),
            negative_sample_size: TYPE_INT = TYPE_INT(10),
            event_vector_size: TYPE_INT = EVENT_VECTOR_SIZE,
            dictionary: EventIndexing = None,
            W: Optional[TYPE_TENSOR] = None,
            WO: Optional[TYPE_TENSOR] = None,
            weight_initialization_scheme: str = "normal",
            weight_initialization_parameters: dict = {},
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
            W: Weight for context in shape(V=vocabulary_size, D).
            W: Weight for event and negative samples in shape(V=vocabulary_size, D).
            posteriors: Post layers to which forward the Embedding layer output
            optimizer: Gradient descent implementation e.g SGD, Adam.
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert 0 == (context_size % 2) < context_size
        # target_size >= context_size in SkipGram, eg. (target=i,am,red,cat), context=(a)
        # assert 0 < target_size <= context_size
        assert TYPE_INT(0) < target_size
        assert isinstance(dictionary, EventIndexing)
        availability_for_negatives = (
            dictionary.vocabulary_size
            - (context_size+target_size)
            - len(EVENT_META_ENTITIES)
        )
        assert \
            TYPE_INT(0) < negative_sample_size <= Embedding.MAX_NEGATIVE_SAMPLE_SIZE, \
            "negative_sample_size [%s] needs less than MAX_NEGATIVE_SAMPLE_SIZE [%s]" % \
            (negative_sample_size, Embedding.MAX_NEGATIVE_SAMPLE_SIZE)
        assert \
            TYPE_INT(0) < negative_sample_size <= availability_for_negatives, \
            "negative_sample_size [%s] needs less than availability_for_negatives [%s]" % \
            (negative_sample_size, availability_for_negatives)
        assert isinstance(optimizer, optimiser.Optimizer)

        # --------------------------------------------------------------------------------
        # Number of outputs from the layer 'M'
        # Negative Sampling is to run Logistic Regression which expect 1 input.
        # Each target and sample is independent and has its label (1/target, 0/sample).
        # Hence regard (target, samples) as outputs -> M = (1+SL).
        #
        # The output Y:shape (N,(1+SL)) needs to be transformed via an adapter
        # as (N,E+C) <-> (N*(E+C),) to align with what the logistic log loss layer
        # expects (input num_nodes ==1).
        #
        # No need to check (num_nodes==(1+SL) as it can be auto-calculated.
        # --------------------------------------------------------------------------------
        # assert num_nodes == (1+negative_sample_size), \
        #     "Number of output should be %s" % (1+negative_sample_size)
        self._M = TYPE_INT(1+negative_sample_size)

        # --------------------------------------------------------------------------------
        # (target, context) pair properties
        # --------------------------------------------------------------------------------
        self._X_shape: tuple = ()                           # Original shape of input X
        self._target_size: TYPE_INT = target_size
        self._target_indices: TYPE_TENSOR = np.empty(0)    # 1D array
        self._context_size: TYPE_INT = context_size
        self._context_indices: TYPE_TENSOR = np.empty(0)

        # --------------------------------------------------------------------------------
        # Negative sampling property
        # --------------------------------------------------------------------------------
        self._dictionary: EventIndexing = dictionary
        self._negative_sample_size: TYPE_TENSOR = negative_sample_size
        self._negative_sample_indices: TYPE_TENSOR = np.empty(0)

        # --------------------------------------------------------------------------------
        # Event vector space for context(s)
        # Gradient dL/dW varies because only the extracted W rows need to be processed.
        # TODO:
        #   Use framework native element extraction, e.g. gather_nd with
        #   [[index0, 1], [index1,1],...]  for TF to extract from W:(V,D).
        # --------------------------------------------------------------------------------
        if W is None:
            self._W: TYPE_TENSOR = self.build_weights(
                M=dictionary.vocabulary_size,
                D=event_vector_size,
                scheme=weight_initialization_scheme,
                **weight_initialization_parameters
            )
        else:
            self._W: TYPE_TENSOR = self.tensor_cast(copy.deepcopy(W), dtype=TYPE_FLOAT)

        # Set the event vector for NIL and UNK to zero.
        self._W[EVENT_INDEX_NIL] = TYPE_FLOAT(0)
        self._W[EVENT_INDEX_UNK] = TYPE_FLOAT(0)
        assert isinstance(self._W, np.ndarray), "Needs NumPy array for array indexing"
        assert \
            self.is_same_dtype(self.tensor_dtype(self.W), TYPE_FLOAT), \
            f"Expected {TYPE_FLOAT} but {self.tensor_dtype(self.W)}"
        assert \
            self.tensor_shape(self.W)[0] >= dictionary.vocabulary_size and \
            self.tensor_shape(self.W)[1] == event_vector_size > 0 and \
            self.tensor_size(self.W) >= dictionary.vocabulary_size * event_vector_size, \
            "W shape needs (%s,%s) but %s." \
            % (dictionary.vocabulary_size, event_vector_size, self.W.shape)

        self._D = self.W.shape[1]

        # --------------------------------------------------------------------------------
        # 2nd weights to take event and negative samples
        # --------------------------------------------------------------------------------
        if WO is None:
            self._WO: TYPE_TENSOR = self.tensor_cast(copy.deepcopy(self.W), dtype=TYPE_FLOAT)
        else:
            assert isinstance(WO, np.ndarray), "Needs NumPy array for array indexing"
            self._WO = self.tensor_cast(copy.deepcopy(WO), dtype=TYPE_FLOAT)
            self._WO[EVENT_INDEX_NIL] = TYPE_FLOAT(0)
            self._WO[EVENT_INDEX_UNK] = TYPE_FLOAT(0)
            assert \
                self.is_same_dtype(self.tensor_dtype(self.WO), TYPE_FLOAT), \
                f"Expected {TYPE_FLOAT} but {self.tensor_dtype(self.WO)}"
            assert \
                self.tensor_shape(self.WO)[0] >= dictionary.vocabulary_size and \
                self.tensor_shape(self.WO)[1] == event_vector_size > 0 and \
                self.tensor_size(self.WO) >= dictionary.vocabulary_size * event_vector_size, \
                "WO shape needs (%s,%s) but %s." \
                % (dictionary.vocabulary_size, event_vector_size, WO.shape)

        # --------------------------------------------------------------------------------
        # W/|W| for cosine calculation
        # --------------------------------------------------------------------------------
        self._U = np.empty(shape=0, dtype=TYPE_FLOAT)

        # --------------------------------------------------------------------------------
        # Event vectors for target event vector(s)
        # --------------------------------------------------------------------------------
        self._We: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)      # Event vectors for target
        self._Be: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)      # BoW of event vectors for targets
        self._dWe: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)     # dL/dWe

        # --------------------------------------------------------------------------------
        # Event vectors for context event vectors
        # --------------------------------------------------------------------------------
        self._Wc: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)      # Event vectors for context
        self._Bc: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)      # BoW of event vectors for contexts
        self._dWc: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)     # dL/dWc:(N,C,D)=(dL/dWc01 + dL/dWc02).
        self._dWc01: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)   # dL/dWc01:(N,C,D) with Be (BoWs of target event vectors).
        self._dWc02: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)   # dL/dWc02:(N,S,D) with Ws (negative sample event vectors).

        # --------------------------------------------------------------------------------
        # Event vectors for negative samples
        # --------------------------------------------------------------------------------
        self._Ws: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)      # Event vectors for negative samples
        self._dWs: TYPE_TENSOR = np.empty(shape=0, dtype=TYPE_FLOAT)     # dL/dWs

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
    def _extract_event_vectors(self, W: TYPE_TENSOR, indices: TYPE_TENSOR):
        """Extract vectors from event vector space W.
        Use numpy 1D array indexing to extract rows from W.
        W[
            [idx, idx, ....]
        ]

        Args:
            W: vector space
            indices: indices to extract vectors from W

        Returns: vectors of shape:(N*?, D) where is C or E.
        """
        if self.tensor_rank(indices) == 0:
            vectors = W[indices:indices+1]     # shape:(1,D)
        elif self.tensor_rank(indices) == 1:
            vectors = W[indices]               # shape:(N*?,D)
        else:
            raise AssertionError(
                "Indices to extract event vectors from W must be rank 0 or 1"
            )

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

    # @memory_profile
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
                - Rank 2 for (num_windows, E+C)    when X has one sequence.
                - Rank 2 for (N*num_windows, E+C)  when X has multiple sequences.

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
        expected_ranks = [2, 3]
        assert \
            self.is_tensor(X) and self.tensor_rank(X) in expected_ranks, \
            "Expected X rank is %s but %s" % (expected_ranks, X.shape)
        assert \
            self.tensor_dtype(X) == TYPE_INT, \
            "Expected X dtype %s but %s" % (TYPE_INT, self.tensor_dtype(X) )

        # --------------------------------------------------------------------------------
        # Reshape (N, num_windows, E+C) into (N*num_windows, E+C) if rank(X) > 2.
        # Knowledge about "to which sequence a (event, context) pair belongs" gets lost.
        # However, Embedding only encodes (event, context) pairs, and sequence knowledge
        # is not utilised. Hence reshaping has no impact on Embedding capability.
        #
        # Make sure to restore the original X.shape for dX.
        # --------------------------------------------------------------------------------
        self._X_shape = self.tensor_shape(X)
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
            # X=self._extract_event_vectors(self.W, self.target_indices),
            # X=self._extract_event_vectors(self.WO, self.target_indices),
            X=self._extract_event_vectors(self.W, self.target_indices),
            shape=(self.N, self.E, self.D)
        )
        # self._Be = self._bagging(self.We)
        self._Be = self._bagging(self.We)
        self._Be = self.divide(self.Be, self.E) if self.E > 1 else self.Be

        # --------------------------------------------------------------------------------
        # BoW vectors for contexts
        # --------------------------------------------------------------------------------
        self._context_indices = self.reshape(X=X[
            ::,
            self.E: self.window_size
        ], shape=(-1))
        self._Wc = self.reshape(
            X=self._extract_event_vectors(self.WO, self.context_indices),
            shape=(self.N, self.C, self.D)
        )
        # self._Bc = self._bagging(self.Wc)
        # self._Bc = self.divide(self._bagging(self.Wc), self.C)

        # --------------------------------------------------------------------------------
        # Positive score (BoW dot Target)
        # The order (Bc, Be), not (Be, Bc) is because the positive score is to
        # make the context Bc closer to the truth Be, hence 'primary' is Bc.
        # --------------------------------------------------------------------------------
        # Ye = positive_scores = self.einsum("nd,nd->n", self.Bc, self.Be)
        Yc = positive_scores = self.einsum("ncd,nd->nc", self.Wc, self.Be)

        # ================================================================================
        # Score (Negative) for label=0 (False) classification
        # ================================================================================
        # --------------------------------------------------------------------------------
        # Event vectors of negative samples
        # --------------------------------------------------------------------------------
        # Prevent negative samples from including any events in all the (event, context).
        # If v is event or context of a sequence Xi but not Xj, then negative samples of
        # Xj may include v. Then rewarding and penalizing on v happen at the same time.
        #
        # set() cannot take a list as it is mutable.
        # set([1,2,3]) is a Python runtime hack. It should be set((1,2,3))
        # --------------------------------------------------------------------------------
        self._negative_sample_indices = self.reshape(X=self.to_tensor([
            self.dictionary.negative_sample_indices(
                size=self.SL,
                excludes=set(tuple(self.to_flat_list(X[row])))  # unique values
            )
            for row in range(self.N)
        ]), shape=(-1))

        # --------------------------------------------------------------------------------
        # [Obsolete] An attempt to take negative samples from the set of events
        # where all the events in (event, targets) in the batch are excluded,
        # so that all events in (event, targets) in the batch will never include
        # those in negative sampels in the batch.
        # --------------------------------------------------------------------------------
        # excludes = set(tuple(self.to_flat_list(X)))
        # self._negative_sample_indices = self.reshape(X=self.to_tensor([
        #     self.dictionary.negative_sample_indices(
        #         size=self.SL,
        #         excludes=excludes
        #     )
        #     for row in range(self.N)
        # ]), shape=(-1))

        # No need to normalize with SL because (n)(s) is independent.
        self._Ws = self.reshape(
            # X=self._extract_event_vectors(self.W, self.negative_sample_indices),
            X=self._extract_event_vectors(self.WO, self.negative_sample_indices),
            shape=(self.N, self.SL, self.D)
        )

        # --------------------------------------------------------------------------------
        # Negative score (BoW dot Negatives)
        # --------------------------------------------------------------------------------
        # Ys = negative_scores = self.einsum("nd,nsd->ns", self.Bc, self.Ws)
        Ys = negative_scores = self.einsum("nd,nsd->ns", self.Be, self.Ws)

        # ================================================================================
        # Result of np.c_[Yc, Ys]
        # ================================================================================
        Y = self.concat([Yc, Ys], axis=1)
        Y = Y.numpy() if tf.is_tensor(Y) else Y
        self._Y = Y
        del Yc, Ys
        # assert self.tensor_shape(self.Y) == (self.N, (1 + self.SL))
        assert self.tensor_shape(self.Y) == (self.N, (self.C + self.SL)), \
            "Expected shape %s but %s" % \
            ((self.N, (self.C + self.SL)), self.tensor_shape(self.Y))
        assert self.is_finite(self.Y), f"NaN or inf detected in {self.Y}"
        return self.Y

    # @memory_profile
    def gradient(self, dY: Union[TYPE_TENSOR, TYPE_FLOAT]) -> TYPE_TENSOR:
        """Calculate gradients dL/dWe:(N,E,D), dL/dWc:(N,C,D), dL/dWs:(N,SL,D)
        dL/dWe = dL/dW[target_indices] = dL/dYc * Wc
        dL/dWc = dL/dW[target_indices] = dL/dYc * Be

        dL/dYc=dL/dY[0:C] is the gradient of context event vectors Wc=W[context_indices].
        dL/dYs=dL/dY[C::] is the gradient of negative sample vectors Ws=W[negative_sample_indices].

        Args:
            dY: Gradient dL/dY
        Returns:
            Return empty tensor of input X shape, as there is no gradient of X

        TODO:
            Think if there is a meaning to calculate the score with (Be, Ws)
            for labels 0.

            (Be(n), Bc(n)) with label 1 is to make them closer.
            (Ws(n)(s), Bc(n)) with labels 0 are to make them away.

            If (Be(n),Bc(n)) is closer, then Be(n) should be away from Ws.
            Then label 0 for (Be(n), Ws(n)(s)) seems making a sense.
        """
        assert self.is_float_tensor(dY) and "Need dY of float tensor but \n%s\n" % dY
        name = "gradient"
        self.dY = dY
        self.logger.debug(
            "layer[%s].%s: dY.shape %s", self.name, name, self.tensor_shape(dY)
        )

        dYc: TYPE_TENSOR = dY[
            ::,
            0: self.C
        ]
        assert self.tensor_shape(dYc) == (self.N, self.C)
        dYs: TYPE_TENSOR = dY[
            ::,
            self.C:
        ]
        assert self.tensor_shape(dYs) == (self.N, self.SL)

        # --------------------------------------------------------------------------------
        # dL/dWe01:(N,E,D)
        # --------------------------------------------------------------------------------
        # Forward path:
        # F1. Be:(N,D) = einsum("ned->nd", We:(N,E,D)) / self.E
        #   F1-1. Summed itself along axis 1 and,       "ned->n1d"
        #   F1-2. Dropped the axis 1 (rank -1)          "n1d->nd"
        #
        # F2. Yc:(N,C) = einsum("nd,ncd->nc", Be:(N,D), Wc:(N,C,D))
        #   F2-1. Add axis 1 to Be                      "nd,ncd->n1d,ncd"
        #   F2-2. Broadcast Be along axis 1             "n1d,ncd->ncd,ncd"
        #   F2-3. Element-multiply              .       "ncd,ncd->ncd"
        #   F2-4. Sum along axis -1.                    "ncd->nc1"
        #   F2-5. Drop the axis -1.                     "nc1->nc"
        #
        # B1. Gradient dL/dBe:(N,D)
        #   B1-1. dL/Yc:(N,C) -> dL/dYc(N,C,1)          "nc->nc1"
        #       Restore the rank lost during "nc1->nc" at F2-5 by adding axis=-1
        #
        #   B1-2. dL/dYc(N,C,1) -> dL/dYc(N,C,D)        "nc1->ncd"
        #       Restore the shape (N,C,D) lost during "ncd->nc1" at F2-4.
        #       This is done via the implicit broadcast.
        #
        #   B1-3. dL/dBe:(N,C,D)                        "ncd->n1d" & "n1d->nd"
        #       = dL/dYc:(N,C,D) * dYc/dBe:(N,C,D)
        #       = dL/dYc:(N,C,D) * Wc:(N,C,D)
        #       = einsum("ncd,ncd->nd", dL/dYc:(N,C,D) * Wc:(N,C,D))
        #       Impact on L by dBe has been amplified by Wc:(N,C,D).
        #
        # B2. Gradient dL/dWe:(N,E,D) = dL/dBe:(N,D) OP I:(N,E,D)
        #   B2-1. [OP] dL/dBe:(N,D) -> dL/dBe:(N,1,D)
        #       Restore the rank of We lost during "n1d->nd" at F1-2 by adding
        #       axis=1 as (N,D)->(N,1,D) as "nd->n1d" (reverse of F1-2).
        #       dL/dBe:(N,1,D) = reshape(dL/dBe:(N,D), shape=(N,1,D))
        #
        #   B2-2. dL/dBe:(N,1,D) -> dBe/dWe:(N,E,D) / E
        #       Restore the shape of We:(N,E,D) lost during "ned->n1d" at F1-1 as
        #       "n1d->ned" (reverse of F1-1) by multiply with dBe/dWe:(N,E,D) = I:(N,E,D).
        #       dL/dWe:(N,E,D) = dL/dBe:(N,1,D) * I:(N,E,D)
        # --------------------------------------------------------------------------------
        # dBe: TYPE_TENSOR = self.multiply(dYc, self.Bc)  # B1
        dWe01: TYPE_TENSOR = self.einsum(
            "ncd,ncd->nd",
            self.multiply(  # TF einsum does not auto-broadcast as NumPy does
                self.reshape(dYc, shape=(self.N, self.C, 1)),
                self.ones(shape=(self.N, self.C, self.D))
            ),
            self.Wc
        )
        dWe02: TYPE_TENSOR = self.einsum(
            "nsd,nsd->nd",
            self.multiply(  # TF einsum does not auto-broadcast as NumPy does
                self.reshape(dYs, shape=(self.N, self.SL, 1)),
                self.ones(shape=(self.N, self.SL, self.D))
            ),
            self.Ws
        )
        assert self.tensor_shape(dWe01) == self.tensor_shape(dWe02) == (self.N, self.D), \
            f"Expected dWe shape {(self.N, self.D)} but " \
            f"dWe01:{self.tensor_shape(dWe01)} dWe02:{self.tensor_shape(dWe02)}"

        self._dWe = self.multiply(
            x=self.reshape(X=self.add(dWe01, dWe02), shape=(self.N, 1, self.D)),
            y=self.full(shape=(self.N, self.E, self.D), value=(1/self.E))
        )
        del dWe01, dWe02
        assert self.is_finite(self.dWe), "NaN/inf in \n%s\n" % self.dWe
        assert \
            self.tensor_shape(self.dWe) == (self.N, self.E, self.D), \
            "Expected shape %s but dWe.shape %s" \
            % ((self.N, self.E, self.D), self.tensor_shape(self.dWe))

        # --------------------------------------------------------------------------------
        # dL/dWc:(N,C,D) for the forward path with Be.
        # Forward path:
        # F1. Yc:(N,C) = einsum("nd,ncd->nc", Be:(N,D), Wc:(N,C,D))
        #   F1-1. Add axis 1 to Be                      "nd,ncd->n1d,ncd"
        #   F1-2. Broadcast Be along axis 1             "n1d,ncd->ncd,ncd"
        #   F1-3. Element-multiply              .       "ncd,ncd->ncd"
        #   F1-4. Sum along axis -1.                    "ncd->nc1"
        #   F1-5. Drop the axis -1.                     "nc1->nc"
        #
        # Backward path:
        # B1: dL/dWc:(N,C,D)
        #   = dL/dYc:(N,C) * dYc/dWc:(N,D)
        #   = dL/dYc:(N,C) * Be:(N,D)
        # --------------------------------------------------------------------------------
        self._dWc = self.einsum(
            "ncd,ncd->ncd",
            self.multiply(
                self.reshape(dYc, shape=(self.N, self.C, 1)),
                self.ones(shape=(self.N, self.C, self.D))
            ),
            self.multiply(
                self.reshape(self.Be, shape=(self.N, 1, self.D)),
                self.ones(shape=(self.N, self.C, self.D))
            )
        )
        assert self.is_finite(self.dWc), "NaN/inf in \n%s\n" % self.dWc
        assert \
            self.tensor_shape(self.dWc) == (self.N, self.C, self.D), \
            "Expected shape %s but dWc.shape %s" \
            % ((self.N, self.C, self.D), self.tensor_shape(self.dWc))

        # --------------------------------------------------------------------------------
        # Forward path:
        # F1: Ys:(N,SL)=einsum("nd,nsd->ns", Be:(N,D), Ws:(N,SL,D)).
        #
        # Backward path:
        # B1: dL/dWs:(N,SL,D) = dL/dYs:(N,SL) * dYs/dWs:(N,D)
        #                     = dL/dYs:(N,SL) * Be:(N,D)
        # --------------------------------------------------------------------------------
        self._dWs = self.einsum(
            "nsd,nsd->nsd",
            self.multiply(
                self.reshape(dYs, shape=(self.N, self.SL, 1)),
                self.ones(shape=(self.N, self.SL, self.D))
            ),
            self.multiply(
                self.reshape(self.Be, shape=(self.N, 1, self.D)),
                self.ones(shape=(self.N, self.SL, self.D))
            )
        )
        assert self.is_finite(self.dWs), "NaN/inf in \n%s\n" % self.dWs
        assert \
            self.tensor_shape(self.dWs) == (self.N, self.SL, self.D), \
            "Expected shape %s but dWs.shape %s" \
            % ((self.N, self.SL, self.D), self.tensor_shape(self.dWs))

        # --------------------------------------------------------------------------------
        # TODO: Need to consider what to return as dX, because X has no gradient.
        # What to set to self._dx? Use empty for now and monitor how it goes.
        # 'self._dX = self.X' is incorrect and can cause unexpected result e.g.
        # trying to copy data into self.X which can cause unexpected effects.
        #
        # Changed to zeros from empty as empty includes "Nan".
        # --------------------------------------------------------------------------------
        # TODO: Replace np.zeros with self.zeros()
        self._dX = np.zeros(shape=self._X_shape, dtype=TYPE_FLOAT)
        # self._dX = self.zeros(shape=self._X_shape, dtype=TYPE_FLOAT)
        return self.dX

    def gradient_numerical(
            self,
            h: Optional[TYPE_FLOAT] = None,
            condition: TYPE_TENSOR = None
    ) -> List[Union[TYPE_TENSOR, TYPE_FLOAT]]:
        """
        Calculate gradients dL/dWe:(N,E,D), dL/dWc:(N,C,D), dL/dWs:(N,SL,D)
        using TF autodiff on L = Li(y=f(w)) where Li=self.objective

        Li is a composition of post layer functions. As each layer can have
        its state, only layer output Yi=fi(Xi) should go into the next layer
        function fi+1.

        If fi+1 is invoked from the layer i by other than fi, it would cause
        unexpected state transfers in the network. Hence invoking the post
        layers from e.g. numerical gradient gn() must NOT happen.
        Set a stateless Li to the objective function Li for the layer i.

        dL/dYc = dL/dY[0:1] is the gradient of target event vectors We.
                 (We=W[target_indices])
        dL/dYs = dL/dY[1:] is the gradient of negative sample vectors Ws.
                 (Ws=W[negative_sample_indices])

        Args:
            h: small number for delta to calculate the numerical gradient
               Not used but for compatibility.
            condition: boolean indices to select elements to run calculations
                       Not used for TF autodiff
        Returns:
            [dWe, dWs, dWc]: gradients of We, Ws, Ws
        """
        name = "gradient_numerical"
        self.logger.debug("layer[%s].%s", self.name, name)

        # --------------------------------------------------------------------------------
        # Can NOT apply the objective function Li to Yc:(N,C), Ys:(N,SL).
        # The objective L (of the network) results from applying the composition
        # of functions in the post layers. Fi is the function of the Embedding.
        #
        # L=(Fi o Fi+1 o ... o Fn-1)(X)
        #  = Fn-1(Fn-2(...(Fi+1(Fi(X)))...))
        #  = Fn-1(Fn-2(...(Fi+1(Y))...))
        #  =(Fi+1 o ... Fn-1)(Y)
        #  =Li(Y)
        #
        # The objective function Li for the Embedding is (Fi+1 o ... o Fn-1)
        # which takes 'Y:(N,1+SL)' as the input, NOT Yc:(N,C) nor Ys:(N,SL).
        #
        # Hence, to run the numerical gradient that applies Li=self.objective,
        # need the shape (N,1+SL).
        #
        # TODO:
        #  Convert to tf.function without using numpy functions.
        #  Highly likely impossible and require rewrite only with tf.tensor.
        #  1. Fi+1, ..., Fn-1 needs to be TF functions.
        #  2. Use tf.range instead of Python range.
        #  See:
        #  * https://stackoverflow.com/questions/56547737 and more
        #  * https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/
        #  * https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/
        # --------------------------------------------------------------------------------
        # def objective_Ws(ws: TYPE_TENSOR):
        #     ys = self.einsum("nd,nsd->ns", self.Bc, ws)
        #     return self.objective(ys)
        #
        # def objective_We(we: TYPE_TENSOR):
        #     yc = self.einsum("nd,nd->n", self.Bc, we)
        #     return self.objective(yc)
        #
        # def objective_Wc(wc: TYPE_TENSOR):
        #     raise NotImplementedError("TBD")
        #
        # dWe = numerical_jacobian(objective_We, self.We, delta=h)
        # dWc = numerical_jacobian(objective_Wc, self.Wc, delta=h)
        # dWs = numerical_jacobian(objective_Ws, self.Ws, delta=h)
        # return [dWe, dWc, dWs]
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        # Tensorflow bug: tape.gradient(objective, X) returns None after tf.concat.
        # Gradients do not exist for variables after tf.concat()
        # https://github.com/tensorflow/tensorflow/issues/37726
        # --------------------------------------------------------------------------------
        def objective_We(we: TYPE_TENSOR):
            """Objective function for We to calculate numerical dL/dWe"""
            _be = self._bagging(we) / self.E
            _yc = self.einsum("nd,ncd->nc", _be, self.Wc)
            _ys = self.einsum("nd,nsd->ns", _be, self.Ws)
            del _be
            return self.objective(_yc) + self.objective(_ys)

        def objective_Wc(wc: TYPE_TENSOR):
            """Objective function for Wc to calculate numerical dL/dWc"""
            _yc = self.einsum("nd,ncd->nc", self.Be, wc)
            return self.objective(_yc)

        def objective_Ws(ws: TYPE_TENSOR):
            """Objective function for Ws to calculate numerical dL/dWs"""
            _ys = self.einsum("nd,nsd->ns", self.Be, ws)
            return self.objective(_ys)

        dWe = self.numerical_jacobian(objective_We, self.We, condition=condition)
        dWs = self.numerical_jacobian(objective_Ws, self.Ws, condition=condition)
        dWc = self.numerical_jacobian(objective_Wc, self.Wc, condition=condition)

        return [dWe, dWs, dWc]

    def _gradient_descent(
            self, w: TYPE_TENSOR, dw: TYPE_TENSOR, indices
    ):
        """Gradient descent on event vector w
        Update the W elements extracted with the indices at the forward path.
        """
        differential: TYPE_TENSOR = self.reshape(
            self.optimizer.differential(dW=dw), shape=(-1, self.D)
        )
        assert self.tensor_size(differential) == (len(indices) * self.D), \
            "dW size is expected to be %s but %s" % \
            (self.tensor_size(differential)/self.D, len(indices))
        np.subtract.at(w, indices, differential)

    def update(self) -> List[Union[TYPE_FLOAT, TYPE_TENSOR]]:
        """
        Responsibility: Update layer state with gradient descent.

        Returns:
            Event vector gradients [dWe:(N,E,D), dWs:(N,SL,D), dWc:(N,C,D)]
            for target, samples, and context.

        Note:
            update() is to update the state of the layer S. Hence not
            include dL/dX which is not part of the layer state.
       """
        self._gradient_descent(w=self._W, dw=self.dWe, indices=self.target_indices)
        self._gradient_descent(w=self._WO, dw=self.dWc, indices=self.context_indices)
        self._gradient_descent(w=self._WO, dw=self.dWs, indices=self.negative_sample_indices)
        self._dS = [self.dWe, self.dWs, self.dWc]

        # self.N changes every time function() is called, hence the buffer
        # cannot be re-usable as the out buffer. Free them.
        # TODO:
        #  * Make sure access to the deleted property access will raise asserts.
        #  * Make sure numerical gradient will not use them or raise asserts.
        del self._We, self._Wc, self._Ws, self._Be

        return self.dS

    # @tf.function
    def _smallest_vector_indices_for_context(
            self,
            vectors: TYPE_TENSOR,
            excludes: TYPE_TENSOR,
            n: TYPE_INT = TYPE_INT(1),
    ):
        """Indices to the nearest distance vectors to the context BoW
        1. Calculate a normalized BoW from the context vectors.
        2. Calculate distance^2 = sum((W-bow)^2) from the BoW to each vector in W.
        3. Taken indices to the top 'n' nearest vectors, which do not include
           those vectors in the context 'vectors'.

        Note:
            When context size = 1, e,g, the context is "woman", then "woman"
            itself must not be the target. Then select the second nearest target.

        Args:
            vectors:
                Context event vectors of shape (C,D) where 'C' is the context_size.
            excludes:
                indices to exclude (e.g. indices to those in the context vectors).
        Returns:
            Indices to W for the predicted target event vector(s).
            Shape () for n==1 or (n,) for n>1.
        """
        assert self.tensor_rank(vectors) == 2, \
            f"Expected shape (C,{self.D}) but {self.tensor_shape(vectors)}"
        context_size = self.tensor_shape(vectors)[0]
        # --------------------------------------------------------------------------------
        # NOT select NIL, UNK, and events in the context themselves.
        # TODO:
        #   Consider the situations where choosing the events in the context
        #   as the target should be justified.
        # --------------------------------------------------------------------------------
        availability = \
            self.dictionary.vocabulary_size - \
            (EVENT_META_ENTITIES_COUNT + context_size)
        assert 0 < n < availability, \
            "Cannot select more than available event vector size."

        # --------------------------------------------------------------------------------
        # Normalized BoW of the context as the target event must be independent
        # from the context size. bow needs the shape (D,) to be 'd'.
        # --------------------------------------------------------------------------------
        bow = self.einsum("cd->d", vectors) / context_size \
            if context_size > 1 else self.reshape(vectors, shape=(-1))

        # --------------------------------------------------------------------------------
        # Distances from event vectors (*excluding* UNK, NIL) to the BoW.
        # The index into the "distances" is EVENT_META_ENTITIES_COUNT less than
        # than the actual index because EVENT_META_ENTITIES are excluded.
        # Need to restore the actual index later.
        # --------------------------------------------------------------------------------
        distances = self.sum(
            # self.pow(self.W[EVENT_META_ENTITIES_COUNT:] - bow, TYPE_FLOAT(2)),
            self.pow(self.W[EVENT_META_ENTITIES_COUNT:] - bow, TYPE_FLOAT(2)),
            axis=-1,
        )
        assert self.tensor_shape(distances) == \
               (self.dictionary.vocabulary_size-EVENT_META_ENTITIES_COUNT,)

        if context_size == 1:
            # --------------------------------------------------------------------------------
            # When context_size = 1, bow is the context event vector itself.
            # Hence the distances include a distance to the context event itself,
            # which must be 0. Update its distance with the max distance value+
            # so as not to match itself. There should be no other situation where
            # the distance gets zero other than the distance to itself.
            # --------------------------------------------------------------------------------
            if distances[self.argmin(distances)] != TYPE_FLOAT(0):
                raise RuntimeError("There must be distance 0 when context_size==1")

            # TF has no in place update
            # distances[self.argmin(distances)] = self.max(distances) + 1e-5
            distances = self.where(
                distances == TYPE_FLOAT(0),
                self.add(self.max(distances), TYPE_FLOAT(1e-5)),
                distances
            )

        if n == 1:
            args = self.to_tensor(self.argmin(distances))
            assert self.tensor_shape(args) == ()
        else:
            # --------------------------------------------------------------------------------
            # Take indices to the top 'n' + context_size nearest distances.
            # +context_size to be able to remove those in 'context'.
            # --------------------------------------------------------------------------------
            args = self.argsort(x=distances)[0:n+context_size]

            # --------------------------------------------------------------------------------
            # Restore the actual indices into W.
            # The 'distances' tensor does not include EVENT_META_ENTITIES.
            # Hence the indices is EVENT_META_ENTITIES_COUNT less than the
            # actual indices into W which includes EVENT_META_ENTITIES.
            # --------------------------------------------------------------------------------
            args = self.add(args, self.to_tensor(EVENT_META_ENTITIES_COUNT))
            self.logger.debug(
                "Selected event indices for predicted target events before "
                "excluding those to the context vectors:\n%s\n",
                args
            )

            # --------------------------------------------------------------------------------
            # Excludes those args that point to those in context 'vectors'.
            # --------------------------------------------------------------------------------
            args = self.mask(x=args, mask=self.in1d(target=args, source=excludes, invert=True))
            args = args[0:n]
            self.logger.debug(
                "Selected event indices for predicted target events:\n%s\n"
                "excluding those to the context vectors=\n%s\n",
                args
            )
            assert self.tensor_shape(args) == (n,), "expected shape:(n,) but %s" % args

        return args

    def _largest_cosine_vector_indices_for_context(
            self,
            vectors: TYPE_TENSOR,
            excludes: TYPE_TENSOR,
            n: TYPE_INT = TYPE_INT(1),
    ):
        """Indices to the largest cosine vectors to the context BoW
        1. Calculate a normalized BoW from the context vectors.
        2. Calculate cosine (dot product) from the BoW to each vector in W.
        3. Taken indices to the top 'n' largest cosines, which do not include
           the cosines to those in the context 'vectors'.

        Note:
            When context size = 1, e,g, the context is "woman", then "woman"
            itself must not be the target. Then select the second nearest target.

        Args:
            vectors:
                Context *norm* vectors of shape (C,D) where 'C' is the context_size.

            excludes:
                indices to exclude (e.g. indices to those in the context vectors).
        Returns:
            Indices to W for the predicted target event vector(s).
            Shape () for n==1 or (n,) for n>1.
        """
        assert self.tensor_rank(vectors) == 2, \
            f"Expected shape (C,{self.D}) but {self.tensor_shape(vectors)}"
        context_size = self.tensor_shape(vectors)[0]
        # --------------------------------------------------------------------------------
        # NOT select NIL, UNK, and events in the context themselves.
        # TODO:
        #   Consider the situations where choosing the events in the context
        #   as the target should be justified.
        # --------------------------------------------------------------------------------
        availability = \
            self.dictionary.vocabulary_size - \
            (EVENT_META_ENTITIES_COUNT + context_size)
        assert 0 < n < availability, \
            "Cannot select more than available event vector size."

        # --------------------------------------------------------------------------------
        # Normalized BoW of the context as the target event must be independent
        # from the context size. bow needs the shape (D,) to be 'd'.
        # --------------------------------------------------------------------------------
        bow = self.einsum("cd->d", vectors) / context_size \
            if context_size > 1 else self.reshape(vectors, shape=(-1))

        # --------------------------------------------------------------------------------
        # Cosines from event vectors (*excluding* UNK, NIL) to the BoW.
        # The index into the "cosines" is EVENT_META_ENTITIES_COUNT less than
        # than the actual index because EVENT_META_ENTITIES are excluded.
        # Need to restore the actual index later.
        # --------------------------------------------------------------------------------
        cosines = self.einsum("d,vd->v", bow, self.U[EVENT_META_ENTITIES_COUNT:])
        assert self.tensor_shape(cosines) == \
               (self.dictionary.vocabulary_size-EVENT_META_ENTITIES_COUNT,)

        # --------------------------------------------------------------------------------
        # Take indices to the top 'n' + context_size largest cosines,
        # to be able to remove those in 'context'.
        # --------------------------------------------------------------------------------
        args = self.argsort(x=cosines, direction='DESCENDING')[0:n+context_size]

        # --------------------------------------------------------------------------------
        # Restore the actual indices into W.
        # The 'cosines' tensor does not include EVENT_META_ENTITIES.
        # Hence the indices is EVENT_META_ENTITIES_COUNT less than the
        # actual indices into W which includes EVENT_META_ENTITIES.
        # --------------------------------------------------------------------------------
        args = self.add(args, self.to_tensor(EVENT_META_ENTITIES_COUNT))
        self.logger.debug(
            "Selected event indices for predicted target events before "
            "excluding those to the context vectors:\n%s\n",
            args
        )

        # --------------------------------------------------------------------------------
        # Excludes those args that point to those in context 'vectors'.
        # --------------------------------------------------------------------------------
        args = self.mask(x=args, mask=self.in1d(target=args, source=excludes, invert=True))
        args = args[0:n]
        self.logger.debug(
            "Selected event indices for predicted target events:\n%s\n"
            "excluding those to the context vectors=\n%s\n",
            args
        )
        assert self.tensor_shape(args) == (n,), "expected shape:(n,) but %s" % args
        return args

    def _predict_by_distance(
            self, x: TYPE_TENSOR, n: TYPE_INT
    ):
        indices = self._smallest_vector_indices_for_context(
            vectors=self._extract_event_vectors(self.W, x),  # shape:(1,)
            excludes=x,
            n=n
        )
        return indices

    def _predict_by_cosine(
            self, x: TYPE_TENSOR, n: TYPE_INT
    ):
        indices = self._largest_cosine_vector_indices_for_context(
            vectors=self._extract_event_vectors(self.U, x),  # shape:(1,)
            excludes=x,
            n=n
        )
        return indices

    def predict(self, X: TYPE_TENSOR, n: TYPE_INT = TYPE_INT(1)) -> TYPE_TENSOR:
        """Predict the target event() for the given context event(s) X.
        The model was trained to get the BoW of contexts Bc=einsum("ncd->nd")
        to the BoW of targets Be=einsum("ned->nd").

        The target events can be multiple in We but returns a single prediction
        event vector.

        TODO:
            How to predict multiple target events from contexts at prediction?
            Need to look at SkipGram (a context to targets) model.

        Args:
            X: Indices of the contexts in shape (C,) for single prediction or
               (N,C) for batch.
            n: Number of candidates to provide
        Return: Indices to W for the predicted target event vectors.
        """
        assert \
            self.is_tensor(X) and \
            self.is_same_dtype(self.tensor_dtype(X), TYPE_INT), \
            "Expected X dtype %s but %s" % (TYPE_INT, self.tensor_dtype(X))

        # --------------------------------------------------------------------------------
        # Context not include UNK nor NIL.
        # TODO:
        #   Consider the situation where UNK could be allowed.
        # --------------------------------------------------------------------------------
        assert self.all(X > self.max(self.to_tensor(EVENT_INDEX_META_ENTITIES))), \
            "Context X cannot include NIL/UNK"

        #_predict = self._predict_by_distance
        _predict = self._predict_by_cosine
        rank = self.tensor_rank(X)
        if rank == 1:       # X:(C,) for single prediction
            # indices = _predict(x=X, n=n)
            indices_distance = self._predict_by_distance(X, n)
            indices_cosine = self._predict_by_cosine(X, n)
        elif rank == 2:     # X:(N,C) for batch predictions
            indices_distance = self.to_tensor([
                self._predict_by_distance(x=x, n=n)
                for x in X
            ])
            indices_cosine = self.to_tensor([
                self._predict_by_cosine(x, n)
                for x in X
            ])

        else:
            raise AssertionError(
                "Expected shape (C,) or (N,C) but %s" %
                self.tensor_shape(X)
            )

        indices = indices_cosine, indices_distance
        return indices

    def adapt_function_to_logistic_log_loss(self, loss: CrossEntropyLogLoss):
        """Adapter function to bridge between Embedding layer function() and
        Logistic Log Loss layer function() to adapt the shapes.

        The logistic log loss layer expects the shape (N,1) for both X and T.
        This is because of supporting labels in OHE format and index format.
        For OHE label, X:(N,1), T(N,1). For index label, X:(N,M), T:(N,).

        Due to the issue https://github.com/tensorflow/tensorflow/issues/37726,
        gradient_numerical() method sends yc:(N,C), ys:(N,S).

        For shape Y:(N,1+SL): function() output.
            1. Create labels T of shape:(N,1+SL) to match Y:(N,1+SL) as zeros.
            2. Set true labels 1 for the targets at the column 0 in T.
            3. Reshape Y and T into (1,-1).

        For shape yc:(N,)C: objective_We() output.
            1. Create labels T of shape:(N,1) to match yc:(N,C) as zeros.
            2. Set true labels 1 for the targets at the column 0 in T.
            3. Reshape yc and T into (1,-1).

        For shape ys:(N,SL): objective_Ws() output.
            1. Create labels T of shape:(N,SL) to match Y:(N,SL).
            2. Set true labels 0 as for the negative samples
            3. Reshape ys and T into (1,-1).

        Args:
            loss: Logistic log loss layer instance
        Returns: Adapter function
        """
        def _adapt_function_handle_Y(Y: TYPE_TENSOR):
            # --------------------------------------------------------------------------------
            # Labels T.
            # 1. T[::, 0:self.C] = 1 for the targets and T[::, self.C:] = 0 for negative samples.
            # 2. Reshape into (-1,1) to align with what Log Loss expects.
            # --------------------------------------------------------------------------------
            T: TYPE_TENSOR = np.zeros(shape=(self.N, self.C+self.SL), dtype=TYPE_INT)
            T[
                ::,
                0:self.C
            ] = TYPE_INT(1)

            # --------------------------------------------------------------------------------
            # Set labels to the Loss layer.
            # --------------------------------------------------------------------------------
            # The network.train() method is supposed to:
            # 1. sets labels T to the layers.
            # 2. call function() methods of all the network layers.
            #
            # For embedding, until it comes through the layers in the network to
            # the EventIndexing or the Embedding layer, labels are unknown.
            # Hence Embedding layer drives the labels settings.
            # --------------------------------------------------------------------------------
            loss.T = self.reshape(T, (-1, 1))

            # --------------------------------------------------------------------------------
            # Reshape Y into (-1,1) to align with what Log Loss expects.
            # --------------------------------------------------------------------------------
            Y = self.reshape(X=Y, shape=(-1, 1))
            return Y

        def _adapt_function_handle_ys(ys: TYPE_TENSOR):
            T: TYPE_TENSOR = np.zeros(shape=(self.N, self.SL), dtype=TYPE_INT)
            loss.T = self.reshape(T, (-1, 1))
            ys = self.reshape(X=ys, shape=(-1, 1))   # (N,1)
            return ys

        def _adapt_function_handle_yc(yc: TYPE_TENSOR):
            T: TYPE_TENSOR = np.ones(shape=(self.N, self.C), dtype=TYPE_INT)
            loss.T = T
            yc = self.reshape(X=yc, shape=(-1, 1))   # (N,1)
            return yc

        def f(Y: TYPE_TENSOR, adapter: Layer = None):
            assert adapter.M == 1, \
                "The number of adapter layer outputs must be 1 " \
                "to adapt to logistic log loss but {}".format(adapter.M)

            shape = self.tensor_shape(Y)
            if shape == (self.N, (self.C+self.SL)):  # Y:(N,C+SL)
                return _adapt_function_handle_Y(Y)
            elif shape == (self.N, self.SL):    # ys:(N,SL)
                return _adapt_function_handle_ys(Y)
            elif shape == (self.N, self.C):            # yc:(N,C)
                return _adapt_function_handle_yc(Y)
            else:
                raise AssertionError("Unexpected ")

        return f

    def adapt_gradient_to_logistic_log_loss(self):
        """Adapter function to bridge between Logistic Log Loss layer gradient()
        and Embedding layer gradient().

        Logistic log loss back-propagates dL/dY of shape (N*(1+SL),1).
        Transform the shape into (N, 1+SL) to match the Embedding output Y.
        """
        def g(dY: TYPE_TENSOR, adapter: Layer = None):
            """Reshape the dY:(N*(C+SL), 1) from the Log Loss layer to
            (N, C+SL) to back-propagate to the Embedding.

            Logistic Log Loss layer function () expects (N*(1+SL), 1) shape.
            Hence the gradient dL/dY from the Loss has (N*(1+SL), 1) shape too.
            """
            assert \
                self.tensor_shape(dY) == (self.N * (self.C+self.SL), 1), \
                "Expected shape is %s but %s" \
                % ((self.N * (self.C+self.SL), 1), self.tensor_shape(dY))

            dY = self.reshape(X=dY, shape=(-1, (self.C+self.SL)))
            assert self.tensor_shape(self.Y) == self.tensor_shape(dY)
            return dY

        return g

    def load(self, path: str) -> Dict:
        """Load and restore the layer state
        Restore only the variables, not class instances, e.g. dictionary or
        optimizer. Such objects are set at the instantiation, and then the
        load method is called to restore the variables only.

        Args:
            path: state file path
        Returns:
            Restored state
        """
        state = super().load(path)
        assert \
            isinstance(state, dict) \
            and len(state) == len(self.state_elements) \
            and all(element in state for element in self.state_elements)

        if self.W is not None:
            del self._W
        # --------------------------------------------------------------------------------
        # Class instances are initialized at the instantiation and reused at load().
        # save() only serializes variables.
        # --------------------------------------------------------------------------------
        # if self.dictionary is not None:
        #     del self._dictionary
        # if self.optimizer is not None:
        #     del self._optimizer

        self.__init__(
            name=state["name"],
            num_nodes=TYPE_INT(1),
            target_size=state["target_size"],
            context_size=state["context_size"],
            negative_sample_size=state["negative_sample_size"],
            event_vector_size=state["event_vector_size"],
            dictionary=self.dictionary,
            W=state["W"],
            WO=state["WO"],
            optimizer=self.optimizer
        )

        return self.S
