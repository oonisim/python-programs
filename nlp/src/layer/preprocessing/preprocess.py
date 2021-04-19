import logging
from typing import (
    Dict,
    List
)

import numpy as np

import function.text as text
import function.utility as utility
import function.fileio as fileio
from common.constant import (
    TYPE_FLOAT,
    TYPE_TENSOR,
    NIL,
    CONTEXT_WINDOW_SIZE
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


class WordIndexing(Layer):
    # ================================================================================
    # Class
    # ================================================================================
    @staticmethod
    def specification_template():
        return WordIndexing.specification(
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
            _SCHEME: WordIndexing.__qualname__,
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

        word_indexing = WordIndexing(
            name=name,
            num_nodes=num_nodes,
            corpus=corpus,
            log_level=parameters["log_level"] if "log_level" in parameters else logging.ERROR
        )

        return word_indexing

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    @property
    def vocabulary(self) -> np.ndarray:
        """Vocabulary of the corpus"""
        return self._vocabulary

    @property
    def probabilities(self) -> Dict[str, TYPE_FLOAT]:
        """Word occurrence ratio"""
        return self._probabilities

    @property
    def word_to_index(self) -> Dict[str, int]:
        """Word occurrence ratio"""
        return self._word_to_index

    @property
    def S(self) -> List:
        """State of the layer"""
        self._S = [self.word_to_index, self.vocabulary, self.probabilities]
        return self._S

    def __init__(
            self,
            name: str,
            num_nodes: int = 1,
            corpus: str = None,
            log_level: int = logging.ERROR
    ):
        """Initialize
        Args:
            name: Layer identity name
            num_nodes: Number of nodes in the layer
            corpus:
            log_level: logging level
        """
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)
        assert len(corpus) > 0
        self.logger.debug("%s: corpus length is %s", self.name, len(corpus))

        self._word_to_index, _, _vocabulary, self._probabilities = text.Function.word_indexing(corpus)
        self._vocabulary = super().to_tensor(_vocabulary)
        assert len(self.vocabulary) == len(self.word_to_index)

        self.logger.debug("%s: vocabulary size is %s)", self.name, len(_vocabulary))
        del _vocabulary, _

    def sentence_to_sequence(self, sentences: str) -> List[List[int]]:
        """Generate a list of word indices per sentence
        Args:
            sentences: one or more sentences delimited by '\n'.
        Returns: List of (word indices per sentence)
        """
        assert len(sentences) > 0
        self.logger.debug(
            "%s:%s sentences are [%s]",
            self.name, "sentence_to_sequence", sentences
        )
        sequences = text.Function.sentence_to_sequence(sentences, self.word_to_index)
        return sequences

    def function(self, X) -> TYPE_TENSOR:
        """Generate a word index sequence for a sentence"""
        return super().to_tensor(self.sentence_to_sequence(X), dtype=object)

    def load(self, path: str):
        """Load and restore the layer state
        Args:
            path: state file path
        """
        state = super().load(path)
        del self._word_to_index, self._vocabulary, self._probabilities
        self._word_to_index = state[0]
        self._vocabulary = state[1]
        self._probabilities = state[2]

        assert \
            isinstance(self.word_to_index, dict) and self.word_to_index[NIL] == 0 \
            and self.vocabulary[0] == NIL


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
            window_size: int = CONTEXT_WINDOW_SIZE,
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
            ("window_size" in parameters and fileio.Function.is_file(parameters["window_size"])) and
            ("event_size" in parameters and fileio.Function.is_file(parameters["event_size"]))
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
    def window_size(self) -> int:
        """Context window size
        For a sequence (b,c,d, e, f,g,d), (b,c,d) are preceding context to the
        event e and (f,g,d) are succeeding context.

        Event size is 1.
        Context window size including the event is 7.
        Stride (size of preceding and succeeding context) is 3.
        """
        return self._window_size

    @property
    def event_size(self) -> int:
        """Length of events e.g. 2 for (announce, market-plunge)"""
        return self._event_size

    @property
    def stride(self) -> int:
        """Length of preceding and succeeding context"""
        return int((self.window_size - self.event_size) / 2)

    def __init__(
            self,
            name: str,
            num_nodes: int = 1,
            window_size: int = CONTEXT_WINDOW_SIZE,
            event_size: int = 1,
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
        assert 1 <= event_size < window_size and (window_size - event_size) % 2 == 0
        super().__init__(name=name, num_nodes=num_nodes, log_level=log_level)

        self._window_size = window_size
        self._event_size = event_size
        self.logger.debug(
            "%s: window_size %s event_size %s",
            self.name, self.window_size, self.event_size
        )

    def forward(self, X: TYPE_TENSOR) -> TYPE_TENSOR:
        if X.ndim == 1:
            return super().to_tensor(utility.Function.event_context_pairs(
                sequence=X,
                window_size=self.window_size,
                event_size=self.event_size
            ))
        elif X.ndim == 2:
            super().to_tensor([
                utility.Function.event_context_pairs(
                    sequence=_x,
                    window_size=self.window_size,
                    event_size=self.event_size
                ) for _x in X
            ])
        else:
            raise RuntimeError(f"Unexpected tensor dimension {X.ndim}.")
