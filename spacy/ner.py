"""
Lambda function to extract entities from text using AWS Comprehend and SpaCy NER.
"""
import json
import logging
from http import (
    HTTPStatus
)
from typing import (
    List,
    Dict,
    Any,
    Optional
)

# pylint: disable=import-error
from util_spacy import (
    Pipeline
)
from util_aws.boto3.comprehend import (
    ComprehendDetect
)
from util_aws.boto3.translate import (
    Translate
)

import boto3


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
SPACY_MODEL_NAME: str = "en_core_web_sm"


# --------------------------------------------------------------------------------
# Global instances
# --------------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

pipeline: Pipeline = Pipeline(model_name=SPACY_MODEL_NAME)
comprehend: ComprehendDetect = ComprehendDetect(
    comprehend_client=boto3.client('comprehend')
)
translate: Translate = Translate(
    translate_client=boto3.client(
        service_name='translate'
    )
)


def get_en_translation(text: str, language_code: Optional[str]):
    """Translate the text from the language_code into English.
    Args:
        text: text to translate
        language_code: language code e.g. 'ar' for Arabic.
    Returns: translated English text
    """
    name: str = "get_en_translation()"
    translation: Optional[str] = None

    # --------------------------------------------------------------------------------
    # Language code detection if language_code is not specified
    # --------------------------------------------------------------------------------
    if not language_code:
        logger.debug(
            "%s: language_code:[%s], hence detecting the language code from text[20:]=[%s].",
            name, language_code, text[:20]
        )
        language_code = comprehend.detect_dominant_language(
            text=text, return_language_code_only=True
        )[0]
        logger.debug("%s: detected language code is [%s].", name, language_code)

    # --------------------------------------------------------------------------------
    # Translation to en if detected or specified language_code is not en
    # --------------------------------------------------------------------------------
    if language_code != "en":
        logger.debug("%s: translating text:[%s...] into English", name, text[:20])
        translation = translate.translate_text(
            text=text, source_language_code=language_code, target_language_code="en"
        )
        logger.debug("%s: translation is [%s...]", name, translation[:20])
    else:
        logger.debug("%s: detected language is en, skipping translation...", name)
        translation = text

    return translation


def get_named_entities(text, entity_type):
    """Get entities from the text in the language using SpaCy.
    Args:
        text: text to extract the entities from
        entity_type:
            type of the entity, e.g. location which AWS Comprehend supports
    Returns: list of entities
    """
    name: str = "get_named_entities()"
    logger.debug(
        "%s: extracting entity_type:[%s] from [%s...]",
        name, entity_type, text[:20]
    )
    extraction: Dict[str, Any] = pipeline.get_named_entities_from_text(
        text=text,
        excludes=["ORDINAL", "CARDINAL", "PERCENT", "DATE", "TIME", "QUANTITY", "MONEY"],
        remove_special_characters=True,
        return_value_only=True,
        remove_similarity_threshold=0.9,
        include_noun_phrases=True,
        include_keywords=True
    )
    if not extraction or all((len(value) == 0 for key, value in extraction.items())):
        msg: str = f"no entity detected by for text:[\n{text}\n]"
        logger.error("%s: %s", name, msg)
        raise RuntimeError(msg)

    # --------------------------------------------------------------------------------
    # Map entity_type to that of SpaCy NER label
    # --------------------------------------------------------------------------------
    if not entity_type:
        entities = extraction
    elif entity_type.lower() == "location":
        entities = extraction['GPE'] + extraction['LOC']
    elif entity_type.lower() == "organization":
        entities = extraction['ORG']
    elif entity_type.lower() == "title":
        entities = extraction['WORK_OF_ART']
    else:
        try:
            entities = extraction[entity_type.upper()]
        except KeyError as error:
            msg: str = f"invalid entity_type:[{entity_type}]."
            logger.error("%s: %s error:[%s]", name, msg, error)
            raise ValueError(msg) from error

    logger.debug("%s: extracted entities are %s.", name, entities)
    return entities


