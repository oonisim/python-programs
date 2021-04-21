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
    EVENT_NIL,
    EVENT_UNK,
    SPACE,
    EVENT_META_ENTITIES,
    EVENT_META_ENTITY_TO_INDEX
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
        padding: str = EVENT_NIL,
        length: TYPE_INT = 1
    ) -> str:
        """Prepend and append size times of the padding event to the corpus
        Args:
            corpus:
            mode: how to pad. PREPEND|APPEND|SQUEEZE
            delimiter: delimiter between events
            padding: a event to pad
            length: length of events to pad
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
                [ EVENT_NIL ] * STRIDE, 
                corpus.split(' '),
                [ EVENT_NIL ] * STRIDE
            ],
            start=[]
        )
        """
        return padded

    @staticmethod
    def replace_punctuations(corpus: str, replacement: str) -> str:
        """Replace punctuations in the corpus
        Args:
            corpus: sequence of events
            replacement: char to replace
        Returns
            corpus: corpus with punctuation being replaced with replacement.
        """
        assert len(replacement) == 1, \
            f"replacement {replacement} must be length 1 character."

        table = str.maketrans(string.punctuation, replacement * len(string.punctuation))
        corpus.translate(table)

        assert len(corpus) > 0, "corpus needs events other than punctuations."
        return corpus

    @staticmethod
    def standardize(text: str) -> str:
        """Standardize the text
        1. Lower the string
        2. Remove punctuation
        3. Replace white space, new lines, and carriage returns with a space
        Args:
            text: sequence of events
        Returns
            standardized: standardized text
        """
        assert isinstance(text, str) and len(text) > 0
        # --------------------------------------------------------------------------------
        # https://stackoverflow.com/a/67165082/4281353
        # pattern: str = '(?<!%s)[%s%s]+(?!unk>)' % ('<EVENT_UNK', re.escape(string.punctuation), r"\s")
        # pattern: str = rf'(?:(?!{EVENT_UNK.lower()})[\W_](?<!{EVENT_UNK.lower()}))+'
        # --------------------------------------------------------------------------------
        pattern: str = rf'(?:(?!{EVENT_UNK.lower()})(?!{EVENT_NIL.lower()})[\W_](?<!{EVENT_UNK.lower()})(?<!{EVENT_NIL.lower()}))+'
        replacement = SPACE
        standardized: str = re.sub(
            pattern=pattern,
            repl=replacement,
            string=text,
            flags=re.IGNORECASE
        ).lower().strip()
        assert len(standardized) > 0, f"Text [{text}] needs events other than punctuations."
        return standardized

    @staticmethod
    def event_occurrence_probability(
            events: List[str], power: TYPE_FLOAT = TYPE_FLOAT(1.0)
    ):
        """Calculate the probabilities of event occurrences
        Args:
            events: list of standardized events
            power:
                parameter to adjust the probability by p = p**power/sum(p**power)
                This is to balance the contributions of less frequent and more frequent events.
                Default 1.0. In word2vec negative sampling, power=0.75 is used.
        """
        assert (len(events) > 0)

        total: int = len(events)
        counts = collections.Counter(events)
        powered = {event: np.power((count / total), power) for (event, count) in counts.items()}

        integral = np.sum(list(powered.values()))
        probabilities = {event: TYPE_FLOAT(p / integral) for (event, p) in powered.items()}
        assert (1.0 - np.sum(list(probabilities.values()))) < 1e-5

        del total, counts, powered, integral
        return probabilities

    @staticmethod
    def event_indexing(corpus: str, power: TYPE_FLOAT=TYPE_FLOAT(1)):
        """Generate event indices
        Add meta-events EVENT_NIL at 0 and EVENT_UNK at 1.
        events are all lower-cased.

        Assumptions:
            Meta event EVENT_NIL is NOT included in the corpus

        Args:
            corpus: A string including sentences to process.
            power: parameter to adjust the event probability

        Returns:
            event_to_index: event to index mapping
            index_to_event: index to event mapping
            vocabulary: unique events in the corpus
            probabilities: event occurrence probabilities
        """
        events = Function.standardize(corpus).split()
        probabilities: Dict[str, TYPE_FLOAT] = Function.event_occurrence_probability(events=events, power=power)
        assert probabilities.get(EVENT_NIL.lower(), None) is None, \
            f"EVENT_NIL {EVENT_NIL.lower()} should not be included in the corpus. Change EVENT_NIL"
        probabilities[EVENT_NIL.lower()] = TYPE_FLOAT(0)
        probabilities[EVENT_UNK.lower()] = probabilities.get(EVENT_UNK.lower(), TYPE_FLOAT(0))

        vocabulary: List[str] = EVENT_META_ENTITIES + list(set(events) - set(EVENT_META_ENTITIES))
        index_to_event: Dict[TYPE_INT, str] = dict(enumerate(vocabulary))
        event_to_index: Dict[str, TYPE_INT] = dict(zip(index_to_event.values(), index_to_event.keys()))

        del events
        return event_to_index, index_to_event, vocabulary, probabilities

    @staticmethod
    def sentence_to_sequence(
            sentences: str,
            event_to_index: Dict[str, TYPE_INT]
    ) -> List[List[TYPE_INT]]:
        """Generate sequence of event indices from a text sentences
        1. Skip an empty line or

        Args:
            sentences:
                A string including one ore more sentences to process.
                A sentence is delimited by EOL('\n').
            event_to_index: event to integer index mapping dictionary

        Returns: List of integer sequence per sentence
        """
        assert isinstance(sentences, str)

        sequences = []
        max_sequence_length = 0
        for line in sentences.split("\n"):
            if len(line.strip()) > 0:   # Skip empty line
                sequence = [
                    event_to_index.get(w, EVENT_META_ENTITY_TO_INDEX[EVENT_UNK.lower()])
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
                constant_values=EVENT_META_ENTITY_TO_INDEX[EVENT_NIL.lower()]
            ).astype(TYPE_INT).tolist()
            for seq in sequences
        ]

        Logger.debug("Sequences generated for \n%s\n%s", sentences, padded)
        return padded
