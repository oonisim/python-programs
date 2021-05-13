import logging
from typing import (
    Dict,
    List,
    Iterable,
    Union
)

import numpy as np
import ray

import function.fileio as fileio
import function.text as text
import function.utility as utility
from common.constant import (
    EOL,
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR,
    EVENT_NIL,
    EVENT_UNK,
    EVENT_INDEX_UNK,
    EVENT_INDEX_META_ENTITIES,
    EVENT_META_ENTITIES,
    EVENT_META_ENTITIES_COUNT,
    EVENT_CONTEXT_WINDOW_SIZE
)
from layer import (
    Layer
)
from layer.constants import (
    _NAME,
    _SCHEME,
    _NUM_NODES,
    _PARAMETERS
)


class EventIndexing(Layer):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return EventIndexing.specification(
            name="w2i001",
            num_nodes=1,
            path_to_corpus="path_to_corpus"
        )

    @staticmethod
    def specification(
            name: str,
            num_nodes: int,
            path_to_corpus: str
    ):
        """Generate a layer specification
        Args:
            name: layer name
            num_nodes: number of nodes (outputs) in the layer
            path_to_corpus: path to the corpus file.
        """
        return {
            _SCHEME: EventIndexing.class_id(),
            _PARAMETERS: {
                _NAME: name,
                _NUM_NODES: num_nodes,
                "path_to_corpus": path_to_corpus
            }
        }

    @staticmethod
    def build(parameters: Dict):
        """Build a layer based on the parameters
        """
        assert (
            isinstance(parameters, dict) and
            (_NAME in parameters and len(parameters[_NAME]) > 0) and
            (_NUM_NODES in parameters and parameters[_NUM_NODES] > 0) and
            ("path_to_corpus" in parameters and fileio.Function.is_file(parameters["path_to_corpus"]))
        ), "build(): missing mandatory elements %s in the parameters\n%s" \
           % ((_NAME, _NUM_NODES, "path_to_corpus"), parameters)

        name = parameters[_NAME]
        num_nodes = parameters[_NUM_NODES]
        with open(parameters["path_to_corpus"], 'r') as _file:
            corpus = _file.read()

        event_indexing = EventIndexing(
            name=name,
            num_nodes=num_nodes,
            corpus=corpus,
            log_level=parameters["log_level"] if "log_level" in parameters else logging.ERROR
        )

        return event_indexing

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def min_sequence_length(self) -> TYPE_INT:
        """Minimum length required for the output sequence"""
        return self._min_sequence_length

    @property
    def vocabulary(self):
        """Unique events observed in the corpus"""
        return self._vocabulary

    @property
    def vocabulary_size(self) -> TYPE_INT:
        """Number of unique events in the vocabulary"""
        return TYPE_INT(len(self.vocabulary))

    @property
    def probabilities(self) -> TYPE_TENSOR:
        """Probability of a specific event to occur"""
        return self._probabilities

    @property
    def event_to_probability(self) -> Dict[str, TYPE_INT]:
        """Event to event probability mapping"""
        return self._event_to_probability

    @property
    def event_to_index(self) -> Dict[str, TYPE_INT]:
        """Event to event index mapping"""
        return self._event_to_index

    @property
    def S(self) -> List:
        """State of the layer"""
        self._S = [
            self.vocabulary,
            self.probabilities,
            self.event_to_index,
            self.event_to_probability
        ]
        return self._S

    def __init__(
            self,
            name: str,
            num_nodes: int = 1,
            corpus: str = "",
            min_sequence_length: TYPE_INT = 3,
            power: TYPE_FLOAT = 0.75,
            log_level: int = logging.ERROR
    ):
        """Initialize the instance
        Args:
            name: Layer identity name
            num_nodes: Number of outputs from the layer.
            corpus: source of sequential events
            min_sequence_length: length to which the output sequence to meet
            log_level: logging level

        Note:
            Use "sentence" as a sequence of event-word in string format.
            Those "event-words" are converted into numerical "event-indices"
            which is called "sequence".

            To generate a (event, context) pair from a generated sequence,
            (event_size + context+size) is required as 'min_sequence_length'.
            At least 3 (event_size=1, context_size=2) is expected.

            The number of outputs (num_nodes) is not fixed as the length of
            sequence(s) generated varies depending on the incoming sentences.
            Hence set to 1.

            min_sequence_length depends on the window_size of the EventContext.
            To be able to create a (event, context) pair, the requirement is:
            min_sequence_length >= window_size = (event_size + context_size)
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert len(corpus) > 0
        assert min_sequence_length >= 3, "Minimum 3 events expected in the corpus"
        self.logger.debug("%s: corpus length is %s", self.name, len(corpus))

        assert num_nodes == 1
        self._M = num_nodes

        self._min_sequence_length = min_sequence_length

        # --------------------------------------------------------------------------------
        # Event statistics (vocabulary, probability, mappings, etc)
        # --------------------------------------------------------------------------------
        self._event_to_index, _, _vocabulary, self._event_to_probability = \
            text.Function.event_indexing(corpus=corpus, power=power)
        del _
        # self._ray_event_to_index_id = ray.put(self._event_to_index)

        # --------------------------------------------------------------------------------
        # Vocabulary as np.array so as to select multiple events with its list indexing.
        # --------------------------------------------------------------------------------
        self._vocabulary = np.array(_vocabulary)
        del _vocabulary
        assert 0 < len(self.vocabulary) == len(self.event_to_index)

        # --------------------------------------------------------------------------------
        # Probabilities
        # --------------------------------------------------------------------------------
        self._probabilities = np.array(list(self.event_to_probability.values()))

        self.logger.debug("%s: vocabulary size is %s)", self.name, self.vocabulary_size)

    def list_events(self, indices: Iterable[TYPE_INT]) -> Iterable[str]:
        """Provides events at the indices in the vocabulary
        Args:
            indices: indices of events to get events
        """
        return self._vocabulary[list(iter(indices))]

    def list_probabilities(self, events: Iterable[str]) -> Iterable[TYPE_FLOAT]:
        """Provides probabilities of events
        Args:
            events: events to get the indices
        """
        return [
            self.event_to_probability.get(event, TYPE_FLOAT(0))
            for event in events
        ]

    @staticmethod
    @ray.remote
    def _get_index(event_to_index, event: str):
        return event_to_index.get(event, EVENT_INDEX_UNK)

    def list_indices(self, events: Iterable[str]) -> Iterable[TYPE_INT]:
        """Provides indices of events
        Args:
            events: events to get the indices
        """
        return [
            self.event_to_index.get(event, EVENT_INDEX_UNK)
            for event in events
        ]
        # return ray.get([
        #     EventIndexing._get_index.remote(self._ray_event_to_index_id, event)
        #     for event in events
        # ])

    def take(self, start: TYPE_INT = 0, size: TYPE_INT = 5) -> Iterable[str]:
        assert 0 <= start < (self.vocabulary_size - EVENT_META_ENTITIES_COUNT)
        assert 0 < size <= (self.vocabulary_size - EVENT_META_ENTITIES_COUNT - start)
        return self.vocabulary[
            EVENT_META_ENTITIES_COUNT+start: EVENT_META_ENTITIES_COUNT+start+size
        ]

    def sample(self, size: TYPE_INT) -> Iterable[str]:
        assert 0 < size <= (self.vocabulary_size - len(EVENT_META_ENTITIES))

        # --------------------------------------------------------------------------------
        # choice is an expensive operation costing nearly half of the entire
        # word2vec CPU time
        # {method 'choice' of 'numpy.random.mtrand.RandomState' objects}
        #     77061   15.993    0.000   41.261    0.001
        # --------------------------------------------------------------------------------
        USE_PROBABILITY = True
        if USE_PROBABILITY:
            sampled: List[str] = list(np.random.choice(
                a=self.vocabulary,
                size=size + EVENT_META_ENTITIES_COUNT,
                replace=False,
                p=self.probabilities
            ))
            # Do not sample meta events
            return list(set(sampled) - set(EVENT_META_ENTITIES))[:size]
        else:
            sampled: List[str] = self.vocabulary[
                np.random.randint(
                    low=EVENT_META_ENTITIES_COUNT,
                    high=self.vocabulary_size,
                    size=size
                )
            ]
            return sampled

    def negative_sample_indices(
            self, size: TYPE_INT,
            excludes: Iterable[TYPE_INT]
    ) -> Iterable[TYPE_INT]:
        """Generate indices of events that are not included in the negatives
        Args:
            size: sample size
            excludes: event indices which should not be included in the sample

        Return: Indices of events not included in negatives
        """
        excludes_length = len(list(excludes))
        availability = (self.vocabulary_size - excludes_length - EVENT_META_ENTITIES_COUNT)
        assert size <= 20, "Verify if the negative sample size is correct"
        assert 0 < size <= availability, \
            "availability %s > sample size %s" % (availability, size)

        candidates = self.list_indices(self.sample(size+excludes_length))
        self.list_indices(candidates)
        return list(set(candidates) - set(excludes))[:size]

    def sentence_to_sequence(self, sentences: str) -> List[List[TYPE_INT]]:
        """Generate a list of event indices per sentence
        Args:
            sentences: one or more sentences delimited by '\n'.
        Returns: List of (event indices per sentence)
        """
        assert len(sentences) > 0
        self.logger.debug(
            "%s:%s sentences are [%s]",
            self.name, "sentence_to_sequence", sentences
        )
        sequences = text.Function.sentence_to_sequence(
            sentences=sentences,
            event_to_index=self.event_to_index,
            minimum_length=self.min_sequence_length
        )
        return sequences

    def sequence_to_sentence(self, sequences: Iterable[TYPE_INT]) -> List[List[str]]:
        """Generate a list of events from
        Args:
            sequences: event indices.
        Returns: List of (event indices per sentence)
        """
        self.logger.debug(
            "%s:%s sequences are [%s]",
            self.name, "sequence_to_sentence", sequences
        )
        sentences = [
            self._vocabulary[seq].tolist()
            for seq in sequences
        ]
        assert \
            len(sentences) > 0, \
            f"Sequences has no valid indices\n{sequences}."
        return sentences

    def function(self, X: Union[str, TYPE_TENSOR]) -> TYPE_TENSOR:
        """Generate a event index sequence for a sentence
        Args:
              X: One more more sentences delimited by EOL to convert into event
                 sequences. Each generated sequence has the min_sequence_length.

        Returns: Event sequences Y of shape (N,sequence_length) where the N is
                 the number of valid sentences in the input, and sequence_length
                 is aligned to the maximum length of the sequences generated.

        Raises:
            AssertionError: When there is no valid sentences in X.
        """
        if self.is_tensor(X):
            sentences = EOL.join(self.to_list(X))
        elif isinstance(X, str):
            sentences = X
        else:
            raise AssertionError("Unexpected input type %s" % type(X))

        sequences = self.to_tensor(self.sentence_to_sequence(sentences))
        assert \
            self.tensor_size(sequences) > 0 and self.tensor_rank(sequences) == 2, \
            "Expected non-empty rank 2 tensor but rank is %s and sequence=\n%s\n" \
            % (self.tensor_rank(sequences), sequences)

        if isinstance(X, str):
            X = X.split(EOL)
            if len(X) > 1:
                self.X = self.to_tensor(X)
            else:
                self.X = self.reshape(self.to_tensor(X), shape=(1, -1))
        else:
            self.X = X

        self.logger.debug("sequence generated \n%s", sequences)
        self._Y = sequences
        return self.Y

    def gradient(
            self, dY: Union[TYPE_TENSOR, TYPE_FLOAT]
    ) -> Union[TYPE_TENSOR, TYPE_FLOAT]:
        # --------------------------------------------------------------------------------
        # TODO: Need to consider what to return as dX, because X has no gradient.
        # What to set to self._dx? Use empty for now and monitor how it goes.
        # 'self._dX = self.X' is incorrect and can cause unexpected result e.g.
        # trying to copy data into self.X which can cause unexpected effects.
        #
        # Changed to zeros as empty includes nan
        # --------------------------------------------------------------------------------
        self._dX = np.zeros(shape=self.tensor_shape(self.X), dtype=TYPE_FLOAT)
        return self.dX

    def load(self, path: str) -> List:
        """Load and restore the layer state
        Args:
            path: state file path
        """
        state = super().load(path)
        del self._vocabulary, \
            self._probabilities, \
            self._event_to_index, \
            self._event_to_probability

        self._vocabulary = state[0]
        self._probabilities = state[1]
        self._event_to_index = state[2]
        self._event_to_probability = state[3]

        assert \
            isinstance(self.event_to_index, dict) and \
            isinstance(self.event_to_probability, dict) and \
            self.event_to_index[EVENT_NIL.lower()] == 0 and \
            self.event_to_index[EVENT_UNK.lower()] == 1 and \
            self.event_to_probability[EVENT_NIL.lower()] == 0 and \
            self.vocabulary[0] == EVENT_NIL.lower() and \
            self.probabilities[0] == TYPE_FLOAT(0) and \
            self.probabilities[1] == self.event_to_probability[EVENT_UNK.lower()]

        return self.S


class EventContext(Layer):
    """Generate (event, context) pairs
    """
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return EventContext.specification(
            name="ec001",
            num_nodes=1,
            window_size=3,
            event_size=1
        )

    @staticmethod
    def specification(
            name: str,
            num_nodes: int = 1,
            window_size: int = EVENT_CONTEXT_WINDOW_SIZE,
            event_size: int = 1,
    ):
        """Generate a layer specification
        Args:
            name: layer name
            num_nodes: number of nodes (outputs) in the layer
            window_size: Size of the context window including the event
            event_size: Size of the events
       """
        return {
            _SCHEME: EventContext.class_id(),
            _PARAMETERS: {
                _NAME: name,
                _NUM_NODES: num_nodes,
                "window_size": window_size,
                "event_size": event_size
            }
        }

    @staticmethod
    def build(parameters: Dict):
        """Build a layer based on the parameters
        """
        assert (
            isinstance(parameters, dict) and
            (_NAME in parameters and len(parameters[_NAME]) > 0) and
            (_NUM_NODES in parameters and parameters[_NUM_NODES] > 0) and
            ("window_size" in parameters and parameters["window_size"] > 0) and
            ("event_size" in parameters and parameters["event_size"] > 0)
        ), "build(): missing mandatory elements %s in the parameters\n%s" \
           % ((_NAME, _NUM_NODES, "window_size", "event_size"), parameters)

        event_context = EventContext(
            name=parameters[_NAME],
            num_nodes=parameters[_NUM_NODES],
            window_size=parameters["window_size"],
            event_size=parameters["event_size"],
            log_level=parameters["log_level"] if "log_level" in parameters else logging.ERROR
        )

        return event_context

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def window_size(self) -> TYPE_INT:
        """Context window size
        For a sequence (b,c,d, e, f,g,d), (b,c,d) are preceding context to the
        event e and (f,g,d) are succeeding context.

        Event size is 1.
        Context window size including the event is 7.
        Stride (size of preceding and succeeding context) is 3.
        """
        return self._window_size

    @property
    def event_size(self) -> TYPE_INT:
        """Length of events e.g. 2 for (announce, market-plunge)"""
        return self._event_size

    @property
    def stride(self) -> TYPE_INT:
        """Length of preceding and succeeding context"""
        return TYPE_INT((self.window_size - self.event_size) / 2)

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            num_nodes: int = 1,
            window_size: TYPE_INT = EVENT_CONTEXT_WINDOW_SIZE,
            event_size: TYPE_INT = 1,
            log_level: int = logging.ERROR
    ):
        """Initialize
        Args:
            name: Layer identity name
            num_nodes: Number of outputs from the layer
            window_size: Size of the context window including the event
            event_size: Size of the events
            log_level: logging level

        Note:
            Number of outputs (num_nodes) is window_size, as each event
            and context are features to be processed at the next layer.
            No need to check if "windows_size = num_nodes".
        """
        assert \
            1 <= event_size < window_size and \
            (window_size - event_size) % 2 == 0
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        self._M = window_size
        self._window_size = window_size
        self._event_size = event_size
        self.logger.debug(
            "%s: window_size %s event_size %s",
            self.name, self.window_size, self.event_size
        )

    def function(self, X: TYPE_TENSOR) -> TYPE_TENSOR:
        """Generate (event, context) pairs from event sequence X. The shape of
        X can be (sequence_size,) for a single sequence or (N, sequence_size)
        for multiple sequences.

        For X: [0,1,2,3,4,5] of shape (sequence_size,), the output Y is shape:
        (num_windows, E+C) where each row is (event, context). C is the size of
        the context.

        Y:[
          [2, 0, 1, 3, 4],
          [3, 1, 2, 4, 5],
          [4, 2, 3, 5, 6]
        ]

        The number of windows (num_windows) of each sentences is the number of
        rows in Y, which varies as the sequence size is not fixed.
        There first column(s) of size E (event can be multiple consecutive) of
        Y is the event, and the rest is the context for the event.
        Y[
            ::,
            range(E)
        ]

        For X of shape (N, sequence_size), Y:shape is (N, num_windows, E+C).

        Args:
            X: One or more event index sequences of shape:(sequence_size,) or
               shape:(N, sequence_size)

        Returns:
            (event, context) pairs. The output shape is (N*num_windows, (E+C))
            regardless if X is one sequence (N,) or multiple sequences (N,-1).

            This is because of excluding the (event, context) pairs where
            'event' has NIL or UNK. The objective of event embedding is
            to identify the specific target 'event', which cannot be NIL or UNK.

            Due to the restriction of not be able to have ragged tensor shape
            (in numpy), it is not possible to have a shape (N, W, (E+C)) where
            W varies, the shape must be (N*W, (E+C))
        """
        X = self.assure_tensor(X)
        self.X = X
        if self.tensor_rank(X) == 1:
            event_context_pairs: TYPE_TENSOR = utility.Function.event_context_pairs(
                sequence=X,
                window_size=self.window_size,
                event_size=self.event_size
            )
        elif self.tensor_rank(X) == 2:
            event_context_pairs: TYPE_TENSOR = self.reshape(self.to_tensor([
                utility.Function.event_context_pairs(
                    sequence=_x,
                    window_size=self.window_size,
                    event_size=self.event_size
                ) for _x in X
            ], dtype=TYPE_INT), shape=(-1, self.window_size))
        else:
            raise RuntimeError(f"Unexpected tensor dimension {X.ndim}.")

        # --------------------------------------------------------------------------------
        # The context needs to identify specific target event(s), hence the target
        # cannot be all either NIL or UNK. It occurs as a sequence is padded to the
        # longest sequence size in a batch.
        #
        # Hence remove the (event,context) pairs where 'event' includes UNK or NIL.
        # However, when the event_size is long, chances are UNK can be included in
        # the target. If all the sequences resulted in events that include NIL or UNK
        # there will be no (event, context) pairs. Then need to reduce the event_size,
        # or need to revise the original sources of the sequences not to include them.
        #
        # Note that the output shape will be rank 2.
        # --------------------------------------------------------------------------------
        event_context_pairs = event_context_pairs[
            np.logical_not(         # Exclude those pairs.
                np.sum(np.isin(     # Select pairs where 'event' is IN (NIL,UNK)
                    event_context_pairs[
                        ::,                 # All (event, context) pairs,
                        :self.event_size    # 'event' part
                    ],
                    EVENT_INDEX_META_ENTITIES
                ), axis=-1).astype(bool)
            )
        ]

        # --------------------------------------------------------------------------------
        # Because of padding, when there are only a few sentence provided,
        # the 'event' in (event,context) can result in 0.
        # [[562  58 117   0   0   0   0   0   0   0   0]]
        # --------------------------------------------------------------------------------
        assert \
            self.tensor_size(event_context_pairs) > 0, \
            "Resulted in no (event,context) pairs. " \
            "Validate the input and target size, or increase sentences to feed." \
            "Target event size [%s], Window size [%s], Input sequences:\n%s\n" % \
            (self.event_size, self.window_size, X)

        assert \
            self.tensor_dtype(event_context_pairs) == TYPE_INT, \
            "The output must be dtype %s" % TYPE_INT

        self._Y = event_context_pairs
        return self.Y

    def gradient(
            self, dY: Union[TYPE_TENSOR, TYPE_FLOAT]
    ) -> Union[TYPE_TENSOR, TYPE_FLOAT]:
        # --------------------------------------------------------------------------------
        # TODO: Need to consider what to return as dX, because X has no gradient.
        # What to set to self._dx? Use empty for now and monitor how it goes.
        # 'self._dX = self.X' is incorrect and can cause unexpected result e.g.
        # trying to copy data into self.X which can cause unexpected effects.
        #
        # Changed to np.zeros as empty include nan
        # --------------------------------------------------------------------------------
        self._dX = np.zeros(shape=self.X.shape, dtype=TYPE_FLOAT)
        return self.dX
