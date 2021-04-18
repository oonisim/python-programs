from typing import(
    Dict,
    List
)
import logging
import numpy as np
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT
)
from common.utility import (
    is_file
)
from layer import (
    Layer
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
import function.text as text


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
            ("path_to_corpus" in parameters and is_file(parameters["path_to_corpus"]))
        ), "build(): missing mandatory elements %s in the parameters\n%s" \
           % ((_NAME, _NUM_NODES, "path_to_corpus"), parameters)

        name = parameters[_NAME]
        num_nodes = parameters[_NUM_NODES]
        with open(parameters["path_to_corpus"], 'r') as file:
            corpus = file.read()

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

        _vocabulary, _index_to_word, self._word_to_index, self._probabilities = \
            text.Function.word_indexing(corpus)
        self._vocabulary = np.ndarray(_vocabulary)
        del _vocabulary, _index_to_word

    def words_to_indices(self, words) -> List[int]:
        assert len(words) > 0
        return [self.word_to_index[word] for word in words]

    def indices_to_words(self, indices) -> List[str]:
        assert len(indices) > 0
        return self.vocabulary[indices]

    def function(self, X) -> np.ndarray:
        return np.array(self.words_to_indices(X))
