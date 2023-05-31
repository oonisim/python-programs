import os
import json
import logging
from typing import (
    List,
    Dict,
    Any,
    Optional
)

from .gpt import (
    TextTask
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --------------------------------------------------------------------------------
# GPT instances
# --------------------------------------------------------------------------------
gpt = TextTask(
    path_to_api_key=f"{os.path.expanduser('~')}/.openai/api_key"
)


def get_theme(text: str) -> str:
    """Get theme of the text
    Args:
        text: text to extract the theme

    Return: theme
    """
    theme = gpt.get_theme(text=text)
    return theme


def get_people(text: str) -> List[str]:
    """Get people from text"""
    people_titles_mapping: dict[str, str] = gpt.get_people(text=text)
    people: List[str] = []
    for name, title in people_titles_mapping.items():
        if title:
            people.append(f"{title} {name}")
        else:
            people.append(name)
    return people


def extract(data: Dict[str, Any]):
    """Extract entities"""
    name: str = "extract()"
    logger.debug(
        "%s: event:%s",
        name, json.dumps(data, indent=4, default=str, ensure_ascii=False)
    )

    entities: Dict[str, Any] = {}

    # --------------------------------------------------------------------------------
    # Entity detection
    # --------------------------------------------------------------------------------
    try:
        text: str = data['text']
        entity_type: str = data['entity_type']

        # theme: str = get_theme(text=text)
        people: List[str] = get_people(text=text)
        entities['PEOPLE'] = people

        logger.debug("%s: returning entities:%s", name, entities)

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

    extracted = extract({
        "text": example,
        "entity_type": "location",
    })
    print(json.dumps(extracted, indent=4, default=str, ensure_ascii=False))
