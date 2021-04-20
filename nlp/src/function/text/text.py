from typing import(
    Dict,
    List
)
import logging
import re
import string
import collections
import numpy as np
from common.constant import (
    TYPE_FLOAT,
    TYPE_INT,
    DELIMITER,
    NIL,
    UNK,
    SPACE,
    META_WORDS,
    META_WORD_TO_INDEX
)
PAD_MODE_PREPEND = 'prepend'
PAD_MODE_APPEND = 'append'
PAD_MODE_SQUEEZE = 'squeeze'

Logger = logging.getLogger(__name__)


class Function:
    @staticmethod
    def pad_text(
        corpus: str,
        mode: str = PAD_MODE_PREPEND,
        delimiter: str = DELIMITER,
        padding: str = NIL,
        length: TYPE_INT = 1
    ) -> str:
        """Prepend and append size times of the padding word to the corpus
        Args:
            corpus:
            mode: how to pad. PREPEND|APPEND|SQUEEZE
            delimiter: delimiter between words
            padding: a word to pad
            length: length of words to pad
        """
        assert corpus and len(corpus) > 0 and isinstance(corpus, str)

        if mode == PAD_MODE_SQUEEZE:
            padded = delimiter.join(
                [padding] * length + [corpus] + [padding] * length
            )
        elif mode == PAD_MODE_PREPEND:
            padded = delimiter.join(
                [padding] * length + [corpus]
            )
        elif mode == PAD_MODE_APPEND:
            padded = delimiter.join(
                [corpus] + [padding] * length
            )
        else:
            raise ValueError("unexpected mode {}".format(mode))
        """
        padded = sum(
            [ 
                [ NIL ] * STRIDE, 
                corpus.split(' '),
                [ NIL ] * STRIDE
            ],
            start=[]
        )
        """
        return padded

    @staticmethod
    def replace_punctuations(corpus: str, replacement: str) -> str:
        """Replace punctuations in the corpus
        Args:
            corpus: sequence of words
            replacement: char to replace
        Returns
            corpus: corpus with punctuation being replaced with replacement.
        """
        assert len(replacement) == 1, \
            f"replacement {replacement} must be length 1 character."

        table = str.maketrans(string.punctuation, replacement * len(string.punctuation))
        corpus.translate(table)

        assert len(corpus) > 0, "corpus needs words other than punctuations."
        return corpus

    @staticmethod
    def standardize(text: str) -> str:
        """Standardize the text
        1. Lower the string
        2. Remove punctuation
        3. Replace white space, new lines, and carriage returns with a space
        Args:
            text: sequence of words
        Returns
            standardized: standardized text
        """
        assert isinstance(text, str) and len(text) > 0
        # --------------------------------------------------------------------------------
        # https://stackoverflow.com/a/67165082/4281353
        # pattern: str = '(?<!%s)[%s%s]+(?!unk>)' % ('<UNK', re.escape(string.punctuation), r"\s")
        # pattern: str = rf'(?:(?!{UNK.lower()})[\W_](?<!{UNK.lower()}))+'
        # --------------------------------------------------------------------------------
        pattern: str = rf'(?:(?!{UNK.lower()})(?!{NIL.lower()})[\W_](?<!{UNK.lower()})(?<!{NIL.lower()}))+'
        replacement = SPACE
        standardized: str = re.sub(
            pattern=pattern,
            repl=replacement,
            string=text,
            flags=re.IGNORECASE
        ).lower().strip()
        assert len(standardized) > 0, f"Text [{text}] needs words other than punctuations."
        return standardized

    @staticmethod
    def word_occurrence_probability(words: List[str], power: TYPE_FLOAT = TYPE_FLOAT(1.0)):
        """Calculate the probabilities of word occurrences
        Args:
            words: list of standardized words
            power:
                parameter to adjust the probability by p = p**power/sum(p**power)
                This is to balance the contributions of less frequent and more frequent words.
                Default 1.0. In word2vec negative sampling, power=0.75 is used.
        """
        assert (len(words) > 0)

        total = len(words)
        counts = collections.Counter(words)
        probabilities = {word: np.power((count / total), power) for (word, count) in counts.items()}

        sigma = np.sum(list(probabilities.values()))
        probabilities = {word: TYPE_FLOAT(p / sigma) for (word, p) in probabilities.items()}
        assert (1.0 - np.sum(list(probabilities.values()))) < 1e-5

        del total, counts, sigma
        return probabilities

    @staticmethod
    def word_indexing(corpus: str):
        """Generate word indices
        Add meta-words NIL at 0 and UNK at 1.
        Words are all lower-cased.

        Assumptions:
            Meta word NIL is NOT included in the corpus

        Args:
            corpus: A string including sentences to process.

        Returns:
            word_to_index: word to word index mapping
            index_to_word: word index to word mapping
            vocabulary: unique words in the corpus
            probabilities: word occurrence probabilities
        """
        words = Function.standardize(corpus).split()
        probabilities: Dict[TYPE_INT, TYPE_FLOAT] = Function.word_occurrence_probability(words)
        assert probabilities.get(NIL.lower(), None) is None, \
            f"NIL {NIL.lower()} should not be included in the corpus. Change NIL"
        probabilities[NIL.lower()] = TYPE_FLOAT(0)
        probabilities[UNK.lower()] = probabilities.get(UNK.lower(), TYPE_FLOAT(0))

        vocabulary: List[str] = META_WORDS + list(set(words) - set(META_WORDS))
        index_to_word: Dict[TYPE_INT, str] = dict(enumerate(vocabulary))
        word_to_index: Dict[str, TYPE_INT] = dict(zip(index_to_word.values(), index_to_word.keys()))

        del words
        return word_to_index, index_to_word, vocabulary, probabilities

    @staticmethod
    def sentence_to_sequence(
            sentences: str,
            word_to_index: Dict[str, TYPE_INT]
    ) -> List[List[TYPE_INT]]:
        """Generate sequence of word indices from a text sentences
        1. Skip an empty line or

        Args:
            sentences:
                A string including one ore more sentences to process.
                A sentence is delimited by EOL('\n').
            word_to_index: word to integer index mapping dictionary

        Returns: List of integer sequence per sentence
        """
        assert isinstance(sentences, str)

        sequences = []
        max_sequence_length = 0
        for line in sentences.split("\n"):
            if len(line.strip()) > 0:   # Skip empty line
                sequence = [
                    word_to_index.get(w, META_WORD_TO_INDEX[UNK.lower()])
                    for w in Function.standardize(line).split()
                ]
                # A line may have punctuations only which results in null sequence
                if len(sequence) > 0:
                    max_sequence_length = max(max_sequence_length, len(sequence))
                    sequences.append(sequence)
            else:
                Logger.warning("Sentence is empty. Skipping...")
        assert len(sequences) > 0, f"No valid sentences in the input \n[{sentences}]\n"

        padded: List[List[TYPE_INT]] = [
            np.pad(
                array=seq,
                pad_width=(0, max_sequence_length - len(seq)),
                constant_values=META_WORD_TO_INDEX[NIL.lower()]
            ).astype(TYPE_INT).tolist()
            for seq in sequences
        ]

        Logger.debug("Sequences generated for \n%s\n%s", sentences, padded)
        return padded
