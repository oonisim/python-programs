"""Module for tagging using OpenAI GPT"""
import os
import json
import logging
import traceback
from typing import (
    List,
    Dict,
    Any,
    Optional
)

from util_python._time import (     # pylint: disable=import-error
    timer
)
from .ner import (                  # pylint: disable=import-error,relative-beyond-top-level
    get_en_translation,
    get_named_entities
)
from .gpt import (                  # pylint: disable=import-error,relative-beyond-top-level
    ChatTaskForTextTagging
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# --------------------------------------------------------------------------------
# GPT
# --------------------------------------------------------------------------------
chat = ChatTaskForTextTagging(
    # TODO: Replace the API key acquisition via Secret Manager
    path_to_api_key=f"{os.path.expanduser('~')}/.openai/api_key"
)


# --------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------
def validate_gpt_extraction_with_spacy_entities(candidates, entities):
    """Validate GPT extraction with Spacy NER
    TODO:
        Implement proper validation logic. Currently crude/naive comparison where
        if one of the word in GPT tag is included in the words of Spacy identified entity,
        it is regarded as match. Hence GPT extracted "North Korea" as a location and
        SpaCy identified entity include "North America", it is regarded as validated,
        although "North Korea" is NOT "North America".

    Args:
        candidates: GPT extracted tags
        entities: SpaCy identified entities
    """
    result = []
    if entities:
        all_words_in_entities = set(" ".join(entities).strip().lower().split())
        words_of_entities = [                       # pylint: disable=unused-variable
            set(entity.strip().lower().split())
            for entity in entities
        ]

        for candidate in candidates:
            words_in_candidate = set(candidate.strip().lower().split())
            if words_in_candidate.intersection(all_words_in_entities):
                # for words_of_entity in words_of_entities:
                #    intersection = words_in_candidate.intersection(words_of_entity)
                #    if intersection and len(intersection) >= len(words_of_entity) -1:
                #     result.append(candidate)
                #     break
                result.append(candidate)

    return result


@timer
def get_theme(text: str) -> str:
    """Extract theme of the text"""
    return chat.get_theme(text=text)


@timer
def get_people(text: str, entities: List[str]) -> List[str]:
    """Get people from text using GPT"""
    people_titles_mapping: dict[str, str] = chat.get_people(text=text)
    candidates = list(people_titles_mapping.keys())
    if entities:
        validated: List[str] = validate_gpt_extraction_with_spacy_entities(
            candidates=candidates, entities=entities
        )
    else:
        validated: List[str] = candidates

    result: List[str] = []
    for name, title in people_titles_mapping.items():
        if name in validated:
            if title:
                result.append(f"{title} {name}")
            else:
                # Add person without title (e.g. president) not to miss key figures
                # without title (e.g. michael jordan may appear without title).
                result.append(name)

    return result


def get_organizations(text: str, theme: str, entities: Optional[List[str]] = None) -> List[str]:
    """Get organization from text using GPT"""
    organization_description_mapping: dict[str, str] = chat.get_organizations(text=text, theme=theme)
    candidates = list(organization_description_mapping.keys())
    if not entities:
        return candidates

    validated: List[str] = validate_gpt_extraction_with_spacy_entities(
        candidates=candidates, entities=entities
    )
    result: List[str] = []
    for name, _ in organization_description_mapping.items():
        if name in validated:
            result.append(name)

    return result


@timer
def distill(text: str, theme: str) -> Dict[str, Any]:
    """Distilled version of tagging
    Example:
    {
        "KEYWORD": [
            "Finland",
            "happiest country",
            "World Happiness Report",
            "Nordic countries",
            "sisu",
            "COVID-19 pandemic"
        ],
        "PERSON": [
            {
                "name": "Melissa Georgio",
                "title": "Australian"
            },
            {
                "name": "Frank Martela",
                "title": "happiness expert and researcher"
            }
        ],
        "ORGANIZATION": [
            "United Nations",
            "Ipsos Group"
        ],
        "LOCATION": [
            "Finland",
            "Sydney",
            "Nordic countries",
            "Europe",
            "COVID-19 pandemic",
            "Asia"
        ]
    }
    """
    _name: str = "distill()"
    distilled: Dict[str, Any] = chat.distill(text=text, theme=theme)
    if not isinstance(distilled, dict):
        msg: str = f"expected JSON response, got {distilled}."
        logger.error("%s: %s. verify the prompt and the GPT response format is JSON.", _name, msg)
        raise RuntimeError(msg)

    logger.debug(
        "%s: response from GPT is %s",
        _name, json.dumps(distilled, indent=4, default=str, ensure_ascii=False)
    )

    # --------------------------------------------------------------------------------
    # PERSON
    # --------------------------------------------------------------------------------
    try:
        people = distilled.get('PERSON', None)
        if people is None or not isinstance(people, list):
            msg: str = f"expected PERSON element as list in {distilled}."
            logger.error("%s: %s. verify the prompt and the GPT response format has PERSON.", _name, msg)
            raise RuntimeError(msg)

        list_title_name = []
        for _person in people:
            if not isinstance(_person, dict):
                msg: str = f"internal error: expected person as dictionary, got {_person}."
                logger.error("%s: %s. verify the prompt and the GPT response format.", _name, msg)
                raise RuntimeError(msg)

            # --------------------------------------------------------------------------------
            # GPT may return {
            #    "name": "Elisabeth Borne",
            #    "title": "Prime Minister"
            # }
            # or
            # {'Elisabeth Borne': 'Prime Minister'}
            # --------------------------------------------------------------------------------
            name = title = None
            if 'name' in _person:
                title = _person.get('title', "")    # title may not exist, hence ""
                name = _person['name']
                if not name:
                    raise RuntimeError(f"invalid name in person:{_person}")
            else:
                for key, value in _person.items():
                    name = key
                    title = value if value else ""  # title may not exist, hence ""
                    break

            list_title_name.append(f"{title} {name}".strip())

        distilled['PERSON'] = list_title_name

    except KeyError as error:
        msg: str = f"internal error of expected key not exist in GPT response: {error}."
        logger.error("%s: %s verify the prompt and the GPT response format for PERSON.", _name, msg)
        raise RuntimeError(msg) from error

    return distilled


def validate_gpt_extraction(gpt_distilled, spacy_entities, entity_type: Optional[str]):
    """Validate GPT extracted entities with Spacy identified entities
    Args:
        gpt_distilled: GPT extracted entities
        spacy_entities: SpaCy identified entities
        entity_type: type of entity e.g. location.
    """
    if entity_type is None or len(entity_type.strip()) == 0:
        # --------------------------------------------------------------------------------
        # PERSON
        # --------------------------------------------------------------------------------
        gpt_distilled[ChatTaskForTextTagging.TAG_ENTITY_TYPE_PERSON] = \
            validate_gpt_extraction_with_spacy_entities(
                candidates=gpt_distilled[ChatTaskForTextTagging.TAG_ENTITY_TYPE_PERSON],
                entities=spacy_entities['PERSON']
            )
        # --------------------------------------------------------------------------------
        # Organization
        # --------------------------------------------------------------------------------
        gpt_distilled[ChatTaskForTextTagging.TAG_ENTITY_TYPE_ORGANIZATION] = \
            validate_gpt_extraction_with_spacy_entities(
                candidates=gpt_distilled[ChatTaskForTextTagging.TAG_ENTITY_TYPE_ORGANIZATION],
                entities=spacy_entities['ORG']
            )
        # --------------------------------------------------------------------------------
        # Location
        # --------------------------------------------------------------------------------
        gpt_distilled[ChatTaskForTextTagging.TAG_ENTITY_TYPE_LOCATION] = \
            validate_gpt_extraction_with_spacy_entities(
                candidates=gpt_distilled[ChatTaskForTextTagging.TAG_ENTITY_TYPE_LOCATION],
                entities=spacy_entities['GPE']+spacy_entities['LOC']
            )

        return gpt_distilled

    if entity_type.upper() in ChatTaskForTextTagging.tag_entity_types():
        return validate_gpt_extraction_with_spacy_entities(
            candidates=gpt_distilled[entity_type.upper()],
            entities=spacy_entities
        )

    return []


# --------------------------------------------------------------------------------
# Tagging
# --------------------------------------------------------------------------------
def extract(data: Dict[str, Any]):
    """Extract tags"""
    name: str = "extract()"
    logger.debug(
        "%s: event:%s",
        name, json.dumps(data, indent=4, default=str, ensure_ascii=False)
    )

    # --------------------------------------------------------------------------------
    # Tagging
    # --------------------------------------------------------------------------------
    result: Dict[str, Any] = {}
    try:
        # --------------------------------------------------------------------------------
        # payload
        # --------------------------------------------------------------------------------
        entity_type: str = data.get('entity_type', "")
        use_gpt: bool = data.get('use_gpt', True)
        title: str = data.get('title', None)            # pylint: disable=unused-variable
        text: str = data.get('text', None)
        if text is None:
            raise RuntimeError("expected 'text' in the request, got none.")

        # --------------------------------------------------------------------------------
        # Auto detect language and translate to English to use with the SpaCy EN model.
        # GPT can handle multi language, however SpaCy non EN models do not perform well.
        # Hence, use English.
        # --------------------------------------------------------------------------------
        text = get_en_translation(text=text, language_code=None)

        # --------------------------------------------------------------------------------
        # Spacy NER
        # --------------------------------------------------------------------------------
        entities = get_named_entities(
            text=text,
            entity_type=entity_type
        )

        if entities:
            logger.debug("%s: entities extracted:%s", name, entities)
        else:
            msg: str = "no entity detected in the text"
            msg = msg + (f" for the entity_type [{entity_type}]." if entity_type else ".")
            logger.warning("%s: %s", name, msg)
            raise RuntimeError(msg)

        # --------------------------------------------------------------------------------
        # 'entities' is list if entity_type is specified e.g. "location", otherwise dictionary.
        # --------------------------------------------------------------------------------
        if entity_type:
            result[entity_type.upper()] = entities
        else:
            result['NER'] = entities

        # --------------------------------------------------------------------------------
        # GPT
        # --------------------------------------------------------------------------------
        if use_gpt:
            # --------------------------------------------------------------------------------
            # Tagging with GPT
            # --------------------------------------------------------------------------------
            # theme: str = title if title else get_theme(text=text)
            # distilled = distill(text=text, theme=theme)
            distilled = distill(text=text, theme="not applicable")

            # --------------------------------------------------------------------------------
            # Validate extracted tag e.g. location is correct with Spacy Named Entity.
            # --------------------------------------------------------------------------------
            validated = validate_gpt_extraction(
                gpt_distilled=distilled,
                spacy_entities=entities,
                entity_type=entity_type
            )

            result['GPT'] = validated

        return result

    # Intentional catch all
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.error("%s: failed due to [%s].", name, error)
        logger.error("%s", traceback.format_exc())
        return {
            "error": str(error)
        }


if __name__ == "__main__":
    # with open("example.txt", "r", encoding='utf-8') as example_text:
    #    example: str = example_text.read()
    example: str = """
Australian Melissa Georgiou moved to Finland over a decade ago 
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
    import re
    extracted = extract({
        "text": re.sub(r'[\s]+', ' ', example),
        "entity_type": "",
        "use_gpt": True
    })
    print(json.dumps(extracted, indent=4, default=str, ensure_ascii=False))
