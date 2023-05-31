import os
import json
import logging
from typing import (
    List,
    Dict,
    Any,
    Optional
)
from util_python._time import (
    timer
)

from .ner import (
    get_en_translation,
    get_named_entities
)
from .gpt import (
    ChatTask
)


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
chat = ChatTask(
    path_to_api_key=f"{os.path.expanduser('~')}/.openai/api_key"
)


def validate_extraction_with_entities(candidates, entities):
    """Validate GPT extraction with Spacy NER"""
    result = set()

    words_of_entities = [
        set(entity.strip().lower().split())
        for entity in entities
    ]

    for candidate in candidates:
        words_in_candidate = set(candidate.strip().lower().split())
        for words_of_entity in words_of_entities:
            intersection = words_in_candidate.intersection(words_of_entity)
            if intersection and len(intersection) >= len(words_of_entity) -1:
                result.add(candidate)
                break

    return sorted(result)


@timer
def get_theme(text: str) -> str:
    return chat.get_theme(text=text)


@timer
def get_people(text: str, entities: List[str]) -> List[str]:
    """Get people from text using GPT"""
    people_titles_mapping: dict[str, str] = chat.get_people(text=text)
    candidates = list(people_titles_mapping.keys())
    if entities:
        validated: List[str] = validate_extraction_with_entities(
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
                result.append(name)

    return result


def get_organizations(text: str, theme: str, entities: Optional[List[str]] = None) -> List[str]:
    """Get organization from text using GPT"""
    organization_description_mapping: dict[str, str] = chat.get_organizations(text=text, theme=theme)
    candidates = list(organization_description_mapping.keys())
    if not entities:
        return candidates

    validated: List[str] = validate_extraction_with_entities(
        candidates=candidates, entities=entities
    )
    result: List[str] = []
    for name, description in organization_description_mapping.items():
        if name in validated:
            result.append(name)

    return result


@timer
def distill(text: str, theme: str) -> Dict[str, Any]:
    return chat.distill(text=text, theme=theme)


def extract(data: Dict[str, Any]):
    """Extract entities"""
    name: str = "extract()"
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
            text=get_en_translation(text=text, language_code=None),
            # entity_type=entity_type
            entity_type=None
        )
        if len(entities) == 0:
            msg: str = "no entity detected in the text"
            msg = msg + (f" for the entity_type [{entity_type}]." if entity_type else ".")
            logger.warning("%s", msg)
        else:
            logger.debug("%s: returning entities:%s", name, entities)

        # --------------------------------------------------------------------------------
        # theme
        # --------------------------------------------------------------------------------
        theme: str = get_theme(text=text)
        entities['Z_GPT_TAGGING'] = distill(text=text, theme=theme)

        # --------------------------------------------------------------------------------
        # Person
        # --------------------------------------------------------------------------------
        # if entities['PERSON']:
        #     people: List[str] = get_people(text=text, entities=entities['PERSON'])
        #     entities['PEOPLE'] = people
        # else:
        #     entities['PEOPLE'] = []

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
    example: str = "The French government under President Emmanuel Macron on Monday survived two no-confidence motions in parliament, but still faces intense pressure over its handling of a controversial pensions reform. Prime Minister Elisabeth Borne incensed the opposition last week by announcing the government would impose a controversial pension reform without a vote in parliament, sparking accusations of anti-democratic behaviour. Its use of an article in the constitution allowing such a move also gave the opposition the right to call motions of no confidence in the government and two such demands were filed. Advertisement READ MORE Pension reform has been imposed in France without a vote. How did it happen? The 577-seat National Assembly lower rejected a motion brought by the centrist LIOT coalition that is also supported by the left, by a margin of just nine votes, much narrower than expected. It then overwhelmingly rejected a motion brought by the far-right National Rally (RN) with just 94 votes in favour. The rejection of the motions means that the reform to raise the pensions age from 62 to 64 has now been adopted by the legislature. It still needs to be signed into law by Mr Macron and may also face legal challenges. Anthony Albanese to reset relationship with France on upcoming visit 24 Jun 2022, 7:38 pm Anthony Albanese to reset relationship with France on upcoming visit It far from represents the end of the biggest domestic crisis of the second mandate in office of Mr Macron, who has yet to make any public comment on the controversy. We never went so far in building a compromise as we did with this reform, Ms Borne told parliament ahead of the vote, saying her use of the article to bypass a vote was profoundly democratic under France s constitution set up by postwar leader Charles de Gaulle. Garbage piles up in Paris following strikes Garbage cans overflowing with trash on the streets as collectors go on strike in Paris, France. Garbage collectors have joined the massive strikes throughout France against pension reform plans. Source: Getty Anadolu Agency Anadolu Agency via Getty Images Trouble ahead A new round of strikes and protests have been called on Thursday and are expected to again bring public transport to a standstill in several areas. There has been a rolling strike by rubbish collectors in Paris, leading to unsightly and unhygienic piles of trash accumulating in the French capital. The future of Ms Borne, appointed as France s second woman premier by Mr Macron after his election victory over the far right for a second mandate, remains in doubt after she failed to secure a parliamentary majority for the reform. READ MORE France wants to raise its retirement age by two years. Why are thousands protesting? Meanwhile, it remains unclear when Mr Macron will finally make public comments over the events, amid reports he is considering an address to the nation. Since Ms Borne invoked article 49.3 of the constitution, there have also been daily protests in Paris and other cities that have on occasion turned violent. A total of 169 people were arrested nationwide on Saturday during spontaneous protests, including one that assembled 4,000 in the capital. People in the street during clashes and protests in Paris. A demonstrator holds a red flare in the middle of a crowd gathered near a fire as several thousand demonstrators gathered at Place de la Concorde, opposite the National Assembly, in Paris on 16 March, 2023 to protest against pension reform. Source: Getty Samuel Boivin Government insiders and observers have raised fears that France is again heading for another bout of violent anti-government protests, only a few years after the Yellow Vest movement shook the country from 2018-2019. In order to pass, the main multi-party no confidence motion needed support from around half the 61 MPs of the traditional right-wing party The Republicans. Even after its leadership insisted they should reject the motions, 19 renegade Republicans MPs voted in favour. I think it s the only way out. We need to move on to something else, said one of the Republicans who voted for the ousting of the government, Aurelien Pradie. Ejecting PM least risky A survey on Sunday showed the head of state s personal rating at its lowest level since the height of the Yellow Vest crisis in 2019, with only 28 per cent of respondents having a positive view of him. Mr Macron has argued that the pension changes are needed to avoid crippling deficits in the coming decades linked to France s ageing population. Those among us who are able will gradually need to work more to finance our social model, which is one of the most generous in the world, Finance Minister Bruno Le Maire said Sunday. Opponents of the reform say it places an unfair burden on low earners, women and people doing physically wearing jobs. Opinion polls have consistently shown that two thirds of French people oppose the changes. As for Mr Macron s options now, replacing Ms Borne would be the least risky and the most likely to give him new momentum, Bruno Cautres of the Centre for Political Research told AFP. Calling new elections is seen as unlikely. When you re in this much of a cycle of unpopularity and rejection over a major reform, it s basically suicidal to go to the polls, Brice Teinturier, head of the polling firm Ipsos, told AFP. A Harris Interactive survey of over 2,000 people this month suggested that the only winner from a new general election would be the far right, with all other major parties losing ground."
    import re
    extracted = extract({
        "text": re.sub(r'[\s]+', ' ', example),
        "entity_type": "location",
    })
    print(json.dumps(extracted, indent=4, default=str, ensure_ascii=False))
