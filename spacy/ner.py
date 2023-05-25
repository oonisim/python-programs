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
from util_aws.boto3.translate import (
    Translate
)
from util_spacy import (
    Pipeline
)

import boto3


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
SPACY_MODEL_NAME: str = "en_core_web_sm"


# --------------------------------------------------------------------------------
# Global instances to avoid re-instantiations
# --------------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

pipeline: Pipeline = Pipeline(model_name=SPACY_MODEL_NAME)


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
        "%s: extracting entity_type:[%s] with language_code:[%s] from [%s...]",
        name, entity_type, text[:20]
    )
    extraction: Dict[str, Any] = pipeline.get_named_entities_from_text(
        text=text,
        excludes=["ORDINAL", "CARDINAL", "PERCENT", "DATE", "TIME", "QUANTITY", "MONEY"],
        remove_special_characters=True,
        return_value_only=True,
        remove_similarity_threshold=0.9,
        include_noun_phrases=True,
        include_keywords=False
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


def ner(data: Dict[str, Any]):
    """NER"""
    name: str = "ner()"
    logger.debug(
        "%s: event:%s",
        name, json.dumps(data, indent=4, default=str, ensure_ascii=False)
    )

    # --------------------------------------------------------------------------------
    # Entity detection
    # --------------------------------------------------------------------------------
    try:
        text: str = data['text']
        entity_type: str = data['entity_type']

        entities = get_named_entities(
            text=text,
            entity_type=entity_type
        )
        if len(entities) == 0:
            msg: str = "no entity detected in the text"
            msg = msg + (f" for the entity_type [{entity_type}]." if entity_type else ".")
            logger.warning("%s", msg)
        else:
            logger.debug("%s: lambda done. returning entities:%s", name, entities)

        return entities

    except (RuntimeError, ValueError, KeyError) as error:
        logger.error("%s: failed due to [%s].", name, error)
        raise


if __name__ == "__main__":
    # with open("example.txt", "r", encoding='utf-8') as example_text:
    #    example: str = example_text.read()
    example: str = """
Australian Melissa Georgiou (Melissa Georgiou) moved to Finland over a decade ago 
to seek happiness in one of the coldest and darkest places on Earth. 
“One of my favorite things about living here is that it's easy to get close to 
nature whether you're in a residential area or in the middle of the city,” Melissa said. 
Originally a teacher, 12 years ago, she switched from the beaches of Sydney to the dark 
winters and cold lakes of Finland, and has never looked back since. Melissa said, “For Finns, 
the concept of happiness is very different from the Australian concept of happiness. 
Finns, she said, are happy to accept portrayals of themselves as melancholy and stubborn
 — a popular local saying is, “People who have happiness must hide it.” 
 “The first thing I noticed here is that you don't go to dinners or barbecues, 
 and you don't talk about real estate. No one asks you where you live, what suburb 
 do you live in, where your kids go to school.” The Finns seem quite happy with the status quo, 
 and they don't always seem to want more. Melissa Georgio's Dark Night in Northern Europe 
 Finland was named the happiest country in the world for the sixth year in a row in 
 the “World Happiness Report” released by the United Nations. “The Nordic countries are 
 often countries with (good) unemployment benefits, pensions, and other benefits,” 
 explains happiness expert and researcher Frank Martela (Frank Martela). 
 However, Frank said that Finland's position in the rankings often surprised its own people.
  “Finns, they're almost outraged because they don't think this can be true. We listen to 
  sad music and hard rock.” “Therefore, happiness is not part of the Finnish self-image.” 
  The other side of Finnish melancholy is a cultural focus on perseverance. 
  Frank said it redefines the way Finns view happiness — a concept known as “sisu” — 
  which is part of Finnish culture and is hard to translate directly, but can be understood as will, 
  determination, perseverance, and reason to face adversity. This, he said, is best reflected in Finns' 
  favorite pastime — getting warm in a sauna after taking a bath in freezing temperatures. 
  “It's about this paradox — from one extreme to the other, and it's a pretty fun experience... 
  because you need perseverance.” Melissa said, but Finland has many things that are great 
  and can provide happiness for people in this country. Finland is one of the 
  European countries least affected by the COVID-19 pandemic, and experts attribute 
  this to a high level of trust in the government and little resistance to complying 
  with restrictions. Trust in government, on the other hand, stems from a country's investment in its citizens. 
"""

    extracted = ner({
        "text": example,
        "entity_type": "location",
    })
    print(json.dumps(extracted, indent=4, default=str, ensure_ascii=False))
