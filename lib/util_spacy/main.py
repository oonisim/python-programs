"""Module for SpaCy operations
References:
    https://www.nltk.org/data.html
"""
import os
import glob
import logging
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Any,
    Optional,
    Union
)

from util_logging import (
    get_logger
)
from util_python.string import (
    string_similarity_score,
    remove_special_characters_from_text,
)

# import nltk
import textacy
import spacy
from spacy.tokens import Doc
from spacy.language import Language


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)
_logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
LANGUAGES: Dict[str, str] = {
    "en": "english",
    "es": "spanish"
}


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def get_language_from_code(language_code: str):
    """Get full language name from its code e.g. 'english' from 'en'
    TODO: move to a common library
    Args:
        language_code: language code
    Return: language
    Raises: KeyError for unknown language codes
    """
    name: str = "get_language_from_code()"
    try:
        return LANGUAGES[language_code.strip().lower()]
    except KeyError:
        _logger.error("%s: unknown language code [%s].", name, language_code)
        raise


# --------------------------------------------------------------------------------
# SpaCy class
# --------------------------------------------------------------------------------
class Pipeline:
    """Class for SpaCy Language Pretrained Pipeline
    SpaCy call the pretrained language pipeline as "nlp" or "Language" or "Pipeline".
    Use "Pipeline" to represent the Spacy Language class instance that consists of
    components to provide a pipeline (tokenizer -> tagger -> parser -> NER -> ...)
    to produce processed "Document" instance from a text.
    """
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------
    @property
    def space(self):
        """Provide a space character for the language"""
        # TODO: use space character of the target language which may not be ASCII space.
        return ' '

    @property
    def model(self) -> spacy.language.Language:
        """Provide the pretrained pipeline model for the language"""
        return self._nlp

    @property
    def stopwords(self) -> List[str]:
        """Provide the stop words for the language"""
        return self._stopwords

    @property
    def named_entity_labels(self) -> List[str]:
        """List of the supported NER labels of the current language
        Returns: List of named entity labels
        """
        return self._named_entity_labels

    @property
    def pos_tags(self) -> List[str]:
        """List of the supported PoS tags of the current language
        Returns: List of PoS tags
        """
        return self._part_of_speech_tags

    def process(self, text) -> spacy.tokens.Doc:
        """Process the text with the pretrained language pipeline
        Args:
            text: text to get the entities from
        Returns: spacy.tokens.Doc instance
        """
        return self.model(text)

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            model_name: str = "en_core_web_lg",
            download_dir: Optional[str] = None
    ):
        # --------------------------------------------------------------------------------
        # Language pipeline
        # --------------------------------------------------------------------------------
        self._language: str = get_language_from_code(model_name.split('_')[0])
        self._model_name = model_name
        try:
            self._nlp: Language = spacy.load(model_name)

        except OSError as error:
            # --------------------------------------------------------------------------------
            # Spacy downloads pretrained pipelines under the user's home directory, which can
            # cause an issue e.g. inside the AWS runtime as it is read-only causing the error:
            # ERROR: Could not install packages due to an OSError:
            # [Errno 30] Read-only file system: '/home/sbx_user1051'
            #
            # Hence, need to download to e.g. /tmp/. However, loading the downloaded ones
            # appears to require the version name, which is not a clean way.
            # --------------------------------------------------------------------------------
            if download_dir is not None:
                spacy.cli.download(model_name, False, False, "--target", download_dir)
                _logger.info(
                    "downloaded spacy pipeline [%s] under [%s]:\n%s",
                    model_name, download_dir, glob.glob(os.path.join(download_dir, model_name, "*"))
                )
                # spacy.__version__ can be 3.5.2, but the name still is -3.5.0
                # spacy.load(f"/tmp/spacy/en_core_web_sm/{model_name}-{spacy.__version__}/")
                # self._nlp = spacy.load(f'/tmp/spacy/{model_name}/{model_name}-3.5.0')
                raise NotImplementedError("need to find a way to load the downloaded.") from error

            spacy.cli.download(model_name)
            self._nlp = spacy.load(model_name)

        # --------------------------------------------------------------------------------
        # Language stop words
        # --------------------------------------------------------------------------------
        # try:
        #     # NLTK raises LookupError if not yet downloaded, instead of OSError.
        #     self._stopwords: List[str] = nltk.corpus.stopwords.words(self._language)
        # except (LookupError, OSError) as error:
        #     _logger.debug("downloading nltk stopwords because of [%s].", error)
        #     nltk.download('stopwords')
        #     self._stopwords: List[str] = nltk.corpus.stopwords.words(self._language)
        self._stopwords: List[str] = self._nlp.Defaults.stop_words

        # --------------------------------------------------------------------------------
        # PoS Tags (https://github.com/explosion/spaCy/blob/master/spacy/glossary.py)
        # https://universaldependencies.org/u/pos/
        # ADJ: adjective, e.g. big, old, green, incomprehensible, first
        # ADP: adposition, e.g. in, to, during
        # ADV: adverb, e.g. very, tomorrow, down, where, there
        # AUX: auxiliary, e.g. is, has (done), will (do), should (do)
        # CONJ: conjunction, e.g. and, or, but
        # CCONJ: coordinating conjunction, e.g. and, or, but
        # DET: determiner, e.g. a, an, the
        # INTJ: interjection, e.g. psst, ouch, bravo, hello
        # NOUN: noun, e.g. girl, cat, tree, air, beauty
        # NUM: numeral, e.g. 1, 2017, one, seventy-seven, IV, MMXIV
        # PART: particle, e.g. ’s, not,
        # PRON: pronoun, e.g I, you, he, she, myself, themselves, somebody
        # PROPN: proper noun, e.g. Mary, John, London, NATO, HBO
        # PUNCT: punctuation, e.g. ., (, ), ?
        # SCONJ: subordinating conjunction, e.g. if, while, that
        # SYM: symbol, e.g. $, %, §, ©, +, −, ×, ÷, =, :), 😝
        # VERB: verb, e.g. run, runs, running, eat, ate, eating
        # --------------------------------------------------------------------------------
        self._part_of_speech_tags: List[str] = self._nlp.get_pipe('tagger').labels

        # --------------------------------------------------------------------------------
        # Named Entity Labels
        # --------------------------------------------------------------------------------
        self._named_entity_labels: List[str] = self._nlp.get_pipe('ner').labels

        # --------------------------------------------------------------------------------
        # PoS Tags (https://github.com/explosion/spaCy/blob/master/spacy/glossary.py)
        # --------------------------------------------------------------------------------
        self._dependency_tags: List[str] = self._nlp.get_pipe("parser").labels

    # --------------------------------------------------------------------------------
    # Functions
    # --------------------------------------------------------------------------------
    @staticmethod
    def exists_similar_entity(
            similarity_threshold: float,
            text: str,
            entities: Set[str]
    ) -> bool:
        """Check if similar entity to 'text' is in already in 'entities'.
        Args:
            similarity_threshold: threshold to decide if similar or not
            text: text to find similarity
            entities: list of existing entities to check against
        Return: True if exists else False
        """
        if similarity_threshold > 0.0:
            for entity in entities:
                if string_similarity_score(text.lower(), entity.lower()) > similarity_threshold:
                    return True

        return False

    def remove_stopwords_from_text(self, text: str) -> str:
        """Remove stop words from the text
        Returns: text with stop words being removed.
        """
        return self.space.join([
            word for word in text.split()
            if word not in self.stopwords
        ])

    def clean_text(
            self,
            text: str,
            remove_special_characters: bool = True,
            remove_stopwords: bool = True,
    ) -> str:
        """
        Args:
            text: text to clean
            remove_special_characters: remove special characters
            remove_stopwords: remove stop characters
        """
        if remove_special_characters:
            text = remove_special_characters_from_text(text=text)
        if remove_stopwords:
            text = self.remove_stopwords_from_text(text=text)

        return text

    def get_named_entities_from_document(
            self,
            doc: spacy.tokens.Doc,
            excludes: Optional[List[str]] = None,
            remove_special_characters: bool = True,
            remove_stopwords: bool = True,
            return_value_only: bool = False,
            remove_similarity_threshold: float = 0.0
    ) -> Dict[str, List[Any]]:
        """Get named entities from the text
        Args:
            doc: document instance to get the entities from
            excludes: entity labels to exclude, e.g. ["ORDINAL", "CARDINAL", "PERCENT", "DATE"]
            remove_special_characters: remove special characters
            remove_stopwords: remove stop characters
            return_value_only: return entity value only for each label
            remove_similarity_threshold:
                not add entity if an entity with similarity > * remove_similarity_threshold exists.
                only valid when return_value_only is True

        Returns: {
                "<label>": [values*],
                ...
            } when return_value_only is True, or {
                "<label>": [
                    "start": <character offset start position of the entity in the text>,
                    "end": <character offset end position of the entity in the tex>,
                    "value": <entity value>
                ],
                ...
            }
        """
        excludes = [
            _entity.upper() for _entity in excludes
            if isinstance(_entity, str)
        ] if excludes else []

        if return_value_only:
            # First use set to remove duplicates
            entity_label_to_value_map: Dict[str, Union[Set, List]] = {
                _label: set()
                for _label in self._named_entity_labels
                if _label not in excludes
            }
            for entity in doc.ents:
                if entity.label_ not in excludes:
                    value: str = self.clean_text(
                        text=entity.text.strip(),
                        remove_special_characters=remove_special_characters,
                        remove_stopwords=remove_stopwords,
                    )
                    if not self.exists_similar_entity(
                            similarity_threshold=remove_similarity_threshold,
                            text=value,
                            entities=entity_label_to_value_map[entity.label_]
                    ):
                        entity_label_to_value_map[entity.label_].add(value)

            # Then convert to list
            entity_label_to_value_map = {
                _label: list(entity_label_to_value_map[_label])
                for _label in entity_label_to_value_map
            }

        else:
            entity_label_to_value_map: Dict[str, List] = {
                _label: []
                for _label in self._named_entity_labels
                if _label not in excludes
            }
            for entity in doc.ents:
                if entity.label_ not in excludes:
                    value: str = self.clean_text(
                        text=entity.text.strip(),
                        remove_special_characters=remove_special_characters,
                        remove_stopwords=remove_stopwords,
                    )
                    entity_label_to_value_map[entity.label_].append({
                        "start": entity.start_char,     # character offset start position in the text
                        "end": entity.end_char,         # character offset end position in the text
                        "value": value
                    })

        return entity_label_to_value_map

    def get_noun_phrases_from_document(
            self,
            doc: spacy.tokens.Doc,
            remove_special_characters: bool = True,
            remove_stopwords: bool = False,
    ) -> List[str]:
        """
        Args:
            doc: document instance to get the entities from
            remove_special_characters: remove special characters
            remove_stopwords: remove stop characters

        Returns: list of noun phrases
        """
        return list({
            self.clean_text(
                text=chunk.text,
                remove_special_characters=remove_special_characters,
                remove_stopwords=remove_stopwords,
            )
            for chunk in doc.noun_chunks
            # --------------------------------------------------------------------------------
            # Skip the phrase if it is the same with its root because it can be a single word,
            # hence, not a phrase noun with multiple words.
            #
            # Skip single unit phrase like 85% which is a phrase (85, %).
            # --------------------------------------------------------------------------------
            if chunk.text.lower() != chunk.root.text.lower() and len(chunk.text.split()) > 1
        })

    @staticmethod
    def get_keywords_from_document(
            doc: Doc,
            windows_size: int = 20,
            top_n: int = 10
    ) -> List[str]:
        """Get keywords form text using textrank
        https://textacy.readthedocs.io/en/latest/api_reference/root.html
        """
        # [kps for kps, weights in textacy.extract.keyterms.sgrank(doc=doc, ngrams=[1,2,3,4,5], topn=top_n)]
        return [
            key for key, weights in textacy.extract.keyterms.textrank(
                doc=doc, window_size=windows_size, topn=top_n
            )
        ]

    def get_named_entities_from_text(
            self,
            text: str,
            excludes: Optional[List[str]] = None,
            remove_special_characters: bool = True,
            remove_stopwords: bool = True,
            return_value_only: bool = False,
            remove_similarity_threshold: float = 0.0,
            include_noun_phrases: bool = True,
            include_entities: Tuple = ("PERSON", "FAC"),
            include_keywords: bool = True,
            top_n: int = 10
    ) -> Dict[str, List[Any]]:
        """Get named entities from the text
        Args:
            text: text to get the entities from
            excludes: entity labels to exclude, e.g. ["ORDINAL", "CARDINAL", "PERCENT", "DATE"]
            remove_special_characters: remove special characters
            remove_stopwords: remove stop characters
            return_value_only: return entity value only for each label
            remove_similarity_threshold:
                not add entity if an entity with similarity > * remove_similarity_threshold exists.
                only valid when return_value_only is True
            include_noun_phrases: include noun phrase as entity if True
            include_entities: entity types to include the noun phrase when include_noun_phrases is True
            include_keywords: include keywords from text if True
            top_n: number of keywords to return if include_keywords is True

        Returns: {
                "<label>": [values*],
                ...
            } when return_value_only is True, or {
                "<label>": [
                    "start": <character offset start position of the entity in the text>,
                    "end": <character offset end position of the entity in the tex>,
                    "value": <entity value>
                ],
                ...
            }
        """
        name: str = "get_named_entities_from_text()"
        assert isinstance(text, str) and len(text.strip()) > 0, f"invalid text:[{text}]"

        doc: Doc = self.process(text=text)

        # --------------------------------------------------------------------------------
        # Named Entities
        # --------------------------------------------------------------------------------
        entities: Dict[str, List] = self.get_named_entities_from_document(
            doc=doc,
            excludes=excludes,
            remove_special_characters=remove_special_characters,
            remove_stopwords=remove_stopwords,
            return_value_only=return_value_only,
            remove_similarity_threshold=remove_similarity_threshold
        )

        # --------------------------------------------------------------------------------
        # Include noun phrases as entities if required.
        # --------------------------------------------------------------------------------
        if include_noun_phrases:
            phrases = self.get_noun_phrases_from_document(
                doc=doc,
                remove_special_characters=False,
                remove_stopwords=False,
            )

            phrase_index_to_phrase_words_mapping: Dict[int, Set[Any]] = {
                index: set(phrase.lower().split())
                for index, phrase in enumerate(phrases)
            }

            for _label in entities:
                if _label not in include_entities:
                    continue

                for index, words_from_phrase in phrase_index_to_phrase_words_mapping.items():
                    match = None
                    # --------------------------------------------------------------------------------
                    # Matching a phrase with entity:
                    # If the words in an entity is the subset of the words of the phrase, include it.
                    # A noun phrase "Australian Melissa Georgiou" will be included as an entity
                    # if "Melissa Georgiou" is in the already-identified entities.
                    #
                    # Match is when at least two words from the entity is included in the phrase
                    # to reduce unexpected inclusion. Otherwise "Australian Melissa Georgiou" can
                    # be included as an organization with a match with  "Australian government".
                    #
                    # One match is enough to break the loop for each entity type.
                    #
                    # TODO:
                    #   Use embedding vector similarity (e.g. sentence transformer embedding or SpaCy
                    #   Transformer embedding using _trf model.
                    #
                    # TODO:
                    #   Incorrect matching detection. e.g. 'Fair Work Commission president Adam Hatcher'
                    #   can be matched with "Fair Work Commission" and it can be added to the ORG
                    #   although it is a person. For now, exclude 'ORG' from 'include_entities' arg.
                    # --------------------------------------------------------------------------------
                    for _entity in entities[_label]:
                        words_from_entity = set(_entity.lower().split())
                        if (
                                words_from_entity.issubset(words_from_phrase)
                                and 1 < len(words_from_entity) < len(words_from_phrase)
                        ):
                            match = phrases[index].strip()
                            # Break because once the phrase is added, no need to check with the rest.
                            break

                    if match:
                        _logger.info(
                            "%s: adding the noun phrase [%s] as an entity for label:[%s]",
                            name, match, _label
                        )
                        entities[_label] += [match]

        # --------------------------------------------------------------------------------
        # Include keywords if required
        # --------------------------------------------------------------------------------
        if include_keywords:
            entities["KEYWORDS"] = self.get_keywords_from_document(doc=doc, top_n=top_n)

        return entities

    def is_pos_tag(self, pos_tag: str) -> bool:
        """Check if the pos tag is valid
        Args:
            pos_tag: tag to check
        Returns: True if valid, else False
        """
        return isinstance(pos_tag, str) and pos_tag in self.pos_tags

    def get_tokens_of_pos_from_text(
            self,
            text: str,
            pos_tags: Optional[Union[List[str], Set[str]]] = None
    ) -> List[spacy.tokens.Token]:
        """Extract tokens of the PoS (e.g. [NOUN, ADJ]) from the text
        Args:
            text: text to extract the tokens
            pos_tags: case-sensitive Part of Speech tags e.g. ['NOUN', "ADJ"]

        Returns: List of tokens
        """
        assert isinstance(text, str) and len(text.strip()) > 0, f"invalid text:[{text}]."

        if pos_tags is not None:
            assert isinstance(pos_tags, list) and len(pos_tags) > 0, f"invalid pos tags {pos_tags}"

            pos_tags = set(pos_tags)
            assert pos_tags.issubset(set(self.pos_tags)), \
                f"invalid pos tag included in {pos_tags}, expected tags:{self.pos_tags}."

            result: List[spacy.tokens.Token] = [
                token for token in self.process(text)
                if token.pos_ in pos_tags
            ]

        else:
            result: List[spacy.tokens.Token] = [
                token for token in self.process(text)
            ]

        return result
