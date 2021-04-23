import logging
from typing import (
    Dict,
    List,
    Iterable
)

import numpy as np

import function.text as text
import function.utility as utility
import function.fileio as fileio
from common.constant import (
    EOL,
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR,
    EVENT_NIL,
    EVENT_UNK,
    EVENT_META_ENTITIES,
    EVENT_META_ENTITIES_COUNT,
    EVENT_META_ENTITY_TO_INDEX,
    EVENT_CONTEXT_WINDOW_SIZE
)
from layer import (
    Layer
)
from layer.constants import (
    _NAME,
    _SCHEME,
    _NUM_NODES,
    _PARAMETERS,
    MAX_NEGATIVE_SAMPLE_SIZE
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
            _SCHEME: EventIndexing.__qualname__,
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
            num_nodes: Number of nodes in the layer
            corpus: source of sequential events
            min_sequence_length: length to which the output sequence to meet
            log_level: logging level

        Note:
            To generate a (event, context) pair from a generated sequence,
            (event_size + context+size) is required as 'min_sequence_length'.
            At least 3 (event_size=1, context_size=2) is expected.

        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert len(corpus) > 0
        assert min_sequence_length >= 3, "Minimum 3 events expected in the corpus"
        self.logger.debug("%s: corpus length is %s", self.name, len(corpus))

        self._min_sequence_length = min_sequence_length
        # --------------------------------------------------------------------------------
        # Event statistics (vocabulary, probability, mappings, etc)
        # --------------------------------------------------------------------------------
        self._event_to_index, _, _vocabulary, self._event_to_probability = \
            text.Function.event_indexing(corpus=corpus, power=power)
        del _

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

    def list_indices(self, events: Iterable[str]) -> Iterable[TYPE_INT]:
        """Provides indices of events
        Args:
            events: events to get the indices
        """
        return [
            self.event_to_index.get(
                event, TYPE_INT(EVENT_META_ENTITY_TO_INDEX[EVENT_UNK.lower()])
            )
            for event in events
        ]

    def take(self, start: TYPE_INT = 0, size: TYPE_INT = 5) -> Iterable[str]:
        assert 0 <= start < (self.vocabulary_size - EVENT_META_ENTITIES_COUNT)
        assert 0 < size <= (self.vocabulary_size - EVENT_META_ENTITIES_COUNT - start)
        return self.vocabulary[
            EVENT_META_ENTITIES_COUNT+start: EVENT_META_ENTITIES_COUNT+start+size
        ]

    def sample(self, size: TYPE_INT) -> Iterable[str]:
        assert 0 < size <= (self.vocabulary_size - len(EVENT_META_ENTITIES))
        candidates: List[str] = list(np.random.choice(
            a=self.vocabulary,
            size=size + EVENT_META_ENTITIES_COUNT,
            replace=False,
            p=self.probabilities
        ))
        # Do not sample meta events
        return list(set(candidates) - set(EVENT_META_ENTITIES))[:size]

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
        assert size <= MAX_NEGATIVE_SAMPLE_SIZE
        assert 0 < excludes_length <= (self.vocabulary_size - size - len(EVENT_META_ENTITIES))

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

    def function(self, X: str) -> TYPE_TENSOR:
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
        sequences = self.to_tensor(self.sentence_to_sequence(X))
        assert \
            self.tensor_size(sequences) > 0 and self.tensor_rank(sequences) == 2, \
            "Expected non-empty rank 2 tensor but rank is %s and sequence=\n%s\n" \
            % (self.tensor_rank(sequences), sequences)

        self.logger.debug("sequence generated \n%s", sequences)
        return sequences

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
            _SCHEME: EventContext.__qualname__,
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
            num_nodes: Number of nodes in the layer
            window_size: Size of the context window including the event
            event_size: Size of the events
            log_level: logging level
        """
        assert \
            1 <= event_size < window_size and \
            (window_size - event_size) % 2 == 0
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

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

        Returns: (event, context) pairs. The output shape can be
                 - (num_windows, E+C)    when X is one sequence.
                 - (N, num_windows, E+C) when X is multiple sequences.

                 Not stacking N x (num_windows, E+C).
        """
        X = self.assure_tensor(X)
        if self.tensor_rank(X) == 1:
            event_context: TYPE_TENSOR = utility.Function.event_context_pairs(
                sequence=X,
                window_size=self.window_size,
                event_size=self.event_size
            )
        elif self.tensor_rank(X) == 2:
            event_context: TYPE_TENSOR = np.array([
                utility.Function.event_context_pairs(
                    sequence=_x,
                    window_size=self.window_size,
                    event_size=self.event_size
                ) for _x in X
            ])
        else:
            raise RuntimeError(f"Unexpected tensor dimension {X.ndim}.")

        assert self.tensor_dtype(event_context) == TYPE_INT
        return event_context
