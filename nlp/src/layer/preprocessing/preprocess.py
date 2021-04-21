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
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_TENSOR,
    EVENT_NIL,
    EVENT_UNK,
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
    def vocabulary(self):
        """Unique events observed in the corpus"""
        return self._vocabulary

    @property
    def vocabulary_size(self) -> TYPE_INT:
        """Number of unique events in the vocabulary"""
        return TYPE_INT(len(self.vocabulary))

    @property
    def probabilities(self) -> Dict[str, TYPE_FLOAT]:
        """Probability of a specific event to occur"""
        return self._probabilities

    @property
    def event_to_index(self) -> Dict[str, TYPE_INT]:
        """Event to event index mapping"""
        return self._event_to_index

    @property
    def S(self) -> List:
        """State of the layer"""
        self._S = [self.event_to_index, self.vocabulary, self.probabilities]
        return self._S

    def __init__(
            self,
            name: str,
            num_nodes: int = 1,
            corpus: str = "",
            power: TYPE_FLOAT = 0.75,
            log_level: int = logging.ERROR
    ):
        """Initialize
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            corpus: source of sequential events
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert len(corpus) > 0
        self.logger.debug("%s: corpus length is %s", self.name, len(corpus))

        self._event_to_index, _, _vocabulary, self._probabilities = \
            text.Function.event_indexing(corpus=corpus, power=power)

        # self._vocabulary = super().to_tensor(_vocabulary)
        self._vocabulary = np.array(_vocabulary)
        assert 0 < len(self.vocabulary) == len(self.event_to_index)

        self.logger.debug("%s: vocabulary size is %s)", self.name, len(_vocabulary))
        del _, _vocabulary

    def list_events(self, indices: Iterable[TYPE_INT]) -> Iterable[str]:
        return self._vocabulary[list(iter(indices))]

    def list_probabilities(self, events: Iterable[str]) -> Iterable[TYPE_FLOAT]:
        return [
            self._probabilities.get(event, TYPE_FLOAT(0))
            for event in events
        ]

    def list_event_indices(self, events: Iterable[str]) -> Iterable[TYPE_INT]:
        return [
            self.event_to_index.get(
                event, TYPE_INT(EVENT_META_ENTITY_TO_INDEX[EVENT_UNK.lower()])
            )
            for event in events
        ]

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
        sequences = text.Function.sentence_to_sequence(sentences, self.event_to_index)
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
        """Generate a event index sequence for a sentence"""
        # return super().to_tensor(self.sentence_to_sequence(X))

        sequence = super().to_tensor(self.sentence_to_sequence(X))
        self.logger.debug("sequence generated \n%s", sequence)
        assert \
            super().tensor_rank(sequence) == 2, \
            "Expected ran 2 but sequence %s" % sequence
        return sequence

    def load(self, path: str):
        """Load and restore the layer state
        Args:
            path: state file path
        """
        state = super().load(path)
        del self._event_to_index, self._vocabulary, self._probabilities
        self._event_to_index = state[0]
        self._vocabulary = state[1]
        self._probabilities = state[2]

        assert \
            isinstance(self.event_to_index, dict) and \
            self.event_to_index[EVENT_NIL] == 0 \
            and self.vocabulary[0] == EVENT_NIL


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
        X = super().assure_tensor(X)
        if super().tensor_rank(X) == 1:
            return utility.Function.event_context_pairs(
                sequence=X,
                window_size=self.window_size,
                event_size=self.event_size
            )
        elif super().tensor_rank(X) == 2:
            return np.array([
                utility.Function.event_context_pairs(
                    sequence=_x,
                    window_size=self.window_size,
                    event_size=self.event_size
                ) for _x in X
            ], dtype=TYPE_INT)
        else:
            raise RuntimeError(f"Unexpected tensor dimension {X.ndim}.")
