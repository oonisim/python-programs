"""Module for Chat completion"""
import json
import logging
from typing import (
    List,
    Dict,
    Set,
    Any,
    Optional,
)

from util_logging import (  # pylint: disable=import-error
    get_logger
)
from util_openai import (   # pylint: disable=import-error
    OpenAI
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)
_logger.setLevel(logging.DEBUG)


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
CATEGORY_EXAMPLES: List[str] = [
    'Culture of Finland', 'Australian sports', 'War Crime', 'French Politics',
    'Quantum Technology', 'Asian Food', 'Lifestyle', 'Life Science', "Energy Policy",
    "Cost of Living", "Immigration Policy", "Legal System", "Gender Equality", "Civil Rights"
                                                                               'Financial Business', "Financial Market", "British Society", "Political Philosophy",
    "Diplomatic Relationship with China", "Australian Economy", "Solar Energy",
    "Innovation", "Relationship",  "Roman History"
]

KEYWORD_EXAMPLES: List[str] = [
    "Defamation Trial", "Inflation", "Criminal Investigation", "No Confidence Vote",
    "Artificial Intelligence", "Sexual Abuse", "Global Happiness Index", "Global Warming",
    "Happiness in Life"
]

SENTIMENTS = {
    "Uplifting": "story covering inspiring people, making people feel sense of hope.",
    "Light": "story that may leave the audience feeling bemused or entertained "
             "which is suitable for ‘Divert Me’ stories.",
    "Neutral": "story covering topics the audience will find interesting and important, "
               "but unlikely to stir great emotion. "
               "Eg, announcements on interest rate rises, cost of living advice",
    "Serious": "story covering a serious issue eg: "
               "people struggling with interest rate rises, "
               "alcohol addiction, crime, policies relating to extremism, "
               "domestic violence policy, visa/work exploitation, politician talking about war",
    "Heavy": "story covering topics likely to elicit strong emotion or story covering heavy violence, "
             "suicide, deaths in custody, incidents of domestic violence, severe impacts of mental health, "
             "impacts of war. Anything requiring a content warning."
}


def _to_json(text: str) -> Dict[str, Any]:
    _func_name: str = "_to_json()"
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as _error:
        msg: str = f"cannot decode to JSON from the GPT response [{text}]"
        _logger.error("%s: %s due to [%s].", _func_name, msg, _error)
        raise RuntimeError(msg) from _error


# --------------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------------
class ChatCompletion(OpenAI):
    """Class for Open AI chat completion task operations"""
    LABEL_PERSON: str = "PERSON"
    LABEL_LOCATION: str = "LOCATION"
    LABEL_ORGANIZATION: str = "ORGANIZATION"
    LABEL_FACILITY: str = "FACILITY"
    LABELS = {
        LABEL_PERSON,
        LABEL_LOCATION,
        LABEL_ORGANIZATION,
        LABEL_FACILITY
    }
    LABEL_COUNTRY: str = "COUNTRY"
    LABEL_INSTITUTION: str = "INSTITUTION"
    LABEL_GOVERNMENT: str = "GOVERNMENT"
    LABEL_BUSINESS: str = "BUSINESS"

    @staticmethod
    def tag_entity_types():
        """List of text tag entity types"""
        return ChatCompletion.LABELS

    def __init__(self, path_to_api_key: Optional[str] = None):
        super().__init__(path_to_api_key=path_to_api_key)

    def get_summary(self, text: str, max_words: int = 25) -> str:
        """Get summary of the text
        Args:
              text: text to summarize.
              max_words: number of max words in the summary
        Returns: summary of the text
        Raises: RuntimeError: Failed to get summary.
        """
        prompt = f"Summarize the TEXT in one sentence less than {max_words+1} words. " \
                 f"Response must be less than {max_words+1} words. " \
                 f"TEXT={text}"

        summary = self.get_chat_completion_by_prompt(prompt=prompt)
        return summary

    def get_theme(self, text: str, max_words: int = 10) -> str:
        """Get summary of the text
        Args:
              text: text to summarize.
              max_words: number of max words in the summary
        Returns: summary of the text
        Raises: RuntimeError: Failed to get summary.
        """
        # --------------------------------------------------------------------------------
        # Theme of the text
        # --------------------------------------------------------------------------------
        prompt = f"Focus of the text less than {max_words+1} words. TEXT={text}"
        theme: str = self.get_chat_completion_by_prompt(prompt=prompt)
        _logger.debug("focus is [%s].", theme)
        return theme

    def get_key_events(self, text, max_words: int = 10) -> Dict[str, Any]:
        """Get key events or items in the form of noun phrase from the text.
        Args:
            text: text to extract the key events from.
            max_words: number of max words for an event noun phrase
        Returns: JSON/dictionary in the form of {
            <event noun phrase>: <description>
        }
        """
        # prompt = "Critical events in the TEXT. " \
        #          f"Return as JSON where key is event as noun phrase within {max_words} words " \
        #          f"and value is explanation. TEXT={text}"

        prompt = f"""Critical EVENTS in the TEXT as JSON in the format:
{{
    "EVENT": "EXPLANATION",
    ...
}}

where EVENT is the event within {max_words} words and 
EXPLANATION is the explanation of the EVENT.

TEXT={text}.
"""

        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

    def get_people(self, text: str) -> Dict[str, Any]:
        """Get people from the text.
        Args:
            text: text to extract the key events from.
        Returns: people as JSON/dictionary in the form of {
            <name>: <title>
        }
        """
        prompt = f"Notable figures in the TEXT as a JSON where key is name and value is its title. " \
                 f"Exclude organizations or groups. " \
                 f"Return JSON null if there is no organization.  TEXT={text}."
        # prompt = f'Generate a JSON with the names and titles of individuals in the TEXT. TEXT={text}.'
        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

    def get_organizations(self, text: str, theme: str) -> Dict[str, Any]:
        """Get organizations from the text.
        Args:
            text: text to extract the key events from.
            theme: theme of the text to which the organizations are related with
        Returns: organizations as JSON/dictionary in the form of {
            <name>: <description>
        }
        """
        prompt = f"Organizations in the TEXT that directly induce the THEME '{theme}' " \
                 "as JSON where the key is organization and value is its description. " \
                 f"Do not include human individuals. " \
                 f"Return JSON null if there is no organization.  TEXT={text}."
        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

    def get_geographic_locations(
            self,
            text: str,
            theme: Optional[str],   # pylint: disable=unused-argument
            top_n: int = 6
    ) -> Dict[str, Any]:
        """Get geographical locations from the text.
        Args:
            text: text to extract the key events from.
            theme: theme of the text to which the organizations are related with
            top_n: number of organizations to return
        Returns: organizations as JSON/dictionary in the form of {
            <name>: <description>
        }
        """
        # prompt = f"{top_n} geographic locations in the TEXT where the THEME '{theme}' occurred. " \
        #          f"The locations must exist in the Google map. " \
        #          "Return as a JSON object where the key is geographic location and the value is explanation " \
        #          "why the location is important to the THEME. " \
        #          f"Return JSON null if there is no geographic locations. TEXT={text}"

        prompt = f"Max {top_n} geographic locations where the TEXT occurred. " \
                 "Return as a JSON object where the key is geographic location and the value is explanation " \
                 "why the location is important to the THEME. " \
                 f"Return JSON null if there is no geographic locations. TEXT={text}"

        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

    def categorize_locations(
            self,
            locations: List[str],
            text: Optional[str] = None  # pylint: disable=unused-argument
    ) -> Dict[str, Any]:
        """categorize the locations extracted from text
        GPT regards government (e.g. parliament), institution (e.g. university) as
        locations besides the place names. Hence, classify them into
        1. government
        2. organization
        3. institution
        4. business
        5. facility
        6. geographic territory place name
        7. other

        Args:
            locations: locations to validate
            text: text from where the locations are extracted
        Returns: dictionary of categorized locations in the format of {
            "ORGANIZATION": [
                "Microsoft",
                "Tesla Giga Factory",
                "SBS News",
                "The Sydney Morning Herald"
            ],
            "INSTITUTION": [
                "Sydney University",
                "NSW Prison",
                "Clarence Correctional Centre"
            ],
            "COUNTRY": [
                "Afghanistan",
                "Australia",
                "Singapore"
            ],
            "PLACE": [
                {
                    "name": "Bali",
                    "country": "Indonesia"
                },
                {
                    "name": "Seattle",
                    "country": "United States",
                    "state": "Washington",
                    "latitude": "47.6062° N",
                    "longitude": "122.3321° W"
                },
                {
                    "name": "New South Wales",
                    "country": "Australia"
                }
            ],
            "GOVERNMENT": [
                "Parliament"
            ],
            "OTHER": []
        }
        OR
        {
            "GOVERNMENT": [],
            "ORGANIZATION": [],
            "INSTITUTION": [
                {
                    "name": "Clarence Correctional Centre",
                    "country": "Australia",
                    "state": "New South Wales",
                    "latitude": -29.7072,
                    "longitude": 152.9322
                }
            ],
            "PLACE": [
                {
                    "name": "Grafton",
                    "country": "Australia",
                    "state": "New South Wales",
                    "latitude": -29.6903,
                    "longitude": 152.9333
                }
            ],
            "OTHER": []
        }
        """
        prompt = f"""
Categorize {locations} into
country or place or organization or institution or government other.
as JSON in the format:
{{
    "{self.LABEL_ORGANIZATION}": [organizations names],
    "{self.LABEL_INSTITUTION}": [institution names],
    "{self.LABEL_COUNTRY}": [ country names ],
    "PLACE": [
        {{
            "name": place name,
            "country": country name,
            "state": state name,
            "latitude": latitude,
            "longitude: longitude
        }}
    ],
    "{self.LABEL_GOVERNMENT}": [government names],
    "OTHER": [other names]
}}
"""

        _func_name: str = "categorize_locations()"
        _logger.debug(
            "%s: locations to categorize: %s", _func_name, locations
        )

        categorized: Dict[str, Any] = _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))
        _logger.debug(
            "%s: categorized locations: %s",
            _func_name, json.dumps(categorized, indent=4, default=str, ensure_ascii=False)
        )
        return categorized

    def get_keywords(self, text: str, theme: str, top_n: int = 5) -> Dict[str, Any]:
        """Get keywords from the text that directly induce the theme.
        Args:
            text: text to extract the key events from.
            theme: theme of the text to which the organizations are related with
            top_n: number of organizations to return
        Returns: keywords as JSON/dictionary
        """
        # prompt = f"Top {top_n} critical keywords from the TEXT that directly induce '{theme}'. " \
        #          f"Return as a JSON list. " \
        #          f"Order by importance. " \
        #          f"TEXT={text}."

        #         prompt = f"""Top {top_n} key noun phrases from the TEXT that directly induce '{theme}'.
        # Let's do step by step.
        # 1. Identify {top_n} KEY EVENT of the TEXT that occurred.
        # 2. Generate a NOUN PHRASE for each event within 5 words.
        # 3. Order by importance.
        # 4. Return as JSON.
        #
        # JSON format:
        # {{
        #     "NOUN PHRASE": "KEY EVENT"
        # }}
        #
        # TEXT={text}.
        # """
        prompt = f"""Top {top_n} KEYWORD and REASON from the TEXT that directly induce '{theme}'
as JSON in the format:
{{
    "KEYWORD": "REASON"
}}

TEXT={text}.
"""

        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get the sentiment of the text
        Args:
            text: text to get the sentiment from.
        """
        prompt = f"""With Sentiment={json.dumps(SENTIMENTS, indent=2, ensure_ascii=False, default=str)},
Select one SENTIMENT about the TEXT and its REASON as JSON in the format {{
    "sentiment": "SENTIMENT",
    "reason": "REASON"
}}

Example:
{{
    "sentiment": "Serious",
    "reason": "The text discusses the case of Kathleen Folbigg, who spent 20 years in jail"
}}

TEXT={text}
"""
        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

    def get_entities(
            self,
            text: str,
            theme: Optional[str] = None,    # pylint: disable=unused-argument
            max_words: int = 25,            # pylint: disable=unused-argument
            top_n: int = 6,
            do_categorize: bool = True
    ) -> Dict[str, Any]:
        """
        Get theme, summary, keywords, key phrases, people, locations from the text
        that directly induce the theme.

        Args:
            text: text to extract the key events from.
            theme: theme of the text to which the organizations are related with
            max_words: max words for summary
            top_n: number of entities to return
            do_categorize:

        Returns: organizations as JSON/dictionary in the form of {
            "keywords": KEYWORDS,
            "phrases": NOUN PHRASES,
            "people": KEY FIGURES,
            "locations": GEOGRAPHIC LOCATIONS
        }
        """
        _func_name: str = "get_entities()"
        #         prompt_that_takes_longer = f"""
        # THEME is '{theme}'.
        #
        # Top {top_n} KEYWORDS from the TEXT that induces the THEME.
        # Top {top_n} PERSON as title and name from the TEXT who participated to the THEME.
        # Top {top_n} ORGANIZATIONS as name and explanation that induced the THEME in the the TEXT.
        # Top {top_n} GEOGRAPHIC LOCATIONS where the THEME occurs in the TEXT.
        #
        # Return a JSON in the following format that the python json.loads method can handle.
        # {{
        #     "KEYWORD": [{{keyword:explanation}}] or [],
        #     "PERSON": [{{name:title}}] or [],
        #     "ORGANIZATION": [{{name:explanation}}] or [],
        #     "LOCATION": [{{location:explanation}}] or []
        # }}
        #
        # TEXT={text}
        # """
        #         prompt_replaced = f"""
        # THEME is '{theme}'.
        #
        # Top {top_n} KEYWORDS from the TEXT that induces the THEME.
        # Top {top_n} PERSON as title and name from the TEXT who participated to the THEME.
        # Max {top_n} ORGANIZATIONS that induced the THEME in the the TEXT. Must be less than {top_n+1}.
        # Top {top_n} GEOGRAPHIC LOCATIONS where the THEME does occur.
        #
        # Return a JSON in the following format that the python json.loads method can handle.
        # {{
        #     "KEYWORD": KEYWORDS  or [],
        #     "{self.LABEL_PERSON}": [{{name:title}}] or [],
        #     "{self.LABEL_ORGANIZATION}": ORGANIZATION or [],
        #     "{self.LABEL_LOCATION}": GEOGRAPHIC LOCATIONS or []
        # }}
        #
        # TEXT={text}
        #
        # """

        prompt = f"""
Top {top_n} news categories or topics as truecased KEYWORDS about the NEWS such as {', '.join(KEYWORD_EXAMPLES)}.
Top {top_n} PERSON as title and name who are world well known and participated in the key events of the NEWS. Must be known figures.
Top {top_n} ORGANIZATIONS that participated in the key events of the NEWS.
Maximum 3 GEOGRAPHIC COUNTRY or LOCATION where the key events of the NEWS happened. Must be Maximum 3.
Maximum 3 CATEGORIES of the NEWS such as {', '.join(CATEGORY_EXAMPLES)}.

Return a JSON in the following format that the python json.loads method can handle.
{{
    "KEYWORD": KEYWORDS  or [],
    "{self.LABEL_PERSON}": [{{name:title}}] or [],
    "{self.LABEL_ORGANIZATION}": ORGANIZATION or [],
    "{self.LABEL_LOCATION}": GEOGRAPHIC COUNTRIES or LOCATION or []
    "CATEGORY": CATEGORIES  or []
}}

NEWS={text}
"""
        entities: Dict[str, Any] = \
            _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

        assert self.LABEL_PERSON in entities
        assert self.LABEL_ORGANIZATION in entities
        assert self.LABEL_LOCATION in entities

        if do_categorize:
            categorized_locations: Dict[str, Any] = \
                self.categorize_locations(locations=entities[self.LABEL_LOCATION])
            assert isinstance(categorized_locations, dict), \
                f"expected JSON response from categorize_locations(), got {categorized_locations}"

            # --------------------------------------------------------------------------------
            # Empty the LOCATION to replace with PLACE from the categorized
            # --------------------------------------------------------------------------------
            originals: Set[str] = {
                _location.lower() for _location in entities.pop(self.LABEL_LOCATION)
            }
            entities[self.LABEL_LOCATION] = []
            _logger.debug(
                "%s: location (lowered) before categorize:%s", _func_name, originals
            )

            locations: Set[str] = set()

            # --------------------------------------------------------------------------------
            # {
            #     "GOVERNMENT": [
            #         "Parliament",
            #         "NSW prison"
            #     ],
            #     "ORGANIZATION": [
            #         "Ipsos Group",
            #         "Apple Inc"
            #     ],,
            #     "INSTITUTION": [
            #         "Sydney University"
            #     ],
            #     "PLACE": {
            #         "Sydney": {
            #             "country": "Australia",
            #             "state": "New South Wales",
            #             "latitude": -33.8688,
            #             "longitude": 151.2093
            #         },
            #         "Bathurst": {
            #             "country": "Australia",
            #             "state": "New South Wales",
            #             "latitude": -33.4193,
            #             "longitude": 149.5775
            #         },
            #         "Australia": {
            #             "country": "Australia",
            #             "state": null,
            #             "latitude": -25.2744,
            #             "longitude": 133.7751
            #         }
            #     },
            #     "OTHER": [
            #         "December quarter"
            #     ]
            # }
            # --------------------------------------------------------------------------------
            for key, values in categorized_locations.items():
                # --------------------------------------------------------------------------------
                # ORGANIZATION
                # --------------------------------------------------------------------------------
                if key.strip().lower() in [
                    self.LABEL_GOVERNMENT.lower(),
                    self.LABEL_ORGANIZATION.lower(),
                    self.LABEL_INSTITUTION.lower()
                ]:
                    assert isinstance(values, list), f"expected LIST, got {values}."
                    if self.LABEL_ORGANIZATION in entities:
                        entities[self.LABEL_ORGANIZATION] += values
                    else:
                        entities[self.LABEL_ORGANIZATION] = values

                # --------------------------------------------------------------------------------
                # COUNTRY
                # --------------------------------------------------------------------------------
                if key.strip().lower() in [
                    self.LABEL_COUNTRY.lower(),
                ]:
                    assert isinstance(values, list), \
                        f"expected LIST for COUNTRY, got {values}."

                    for _country in values:
                        if isinstance(_country, str):
                            _lowered = _country.lower()
                            if _lowered not in locations and _lowered in originals:
                                locations.add(_lowered)
                                entities[self.LABEL_LOCATION].append(_country)
                        elif isinstance(_country, dict) and 'name' in _country:
                            _lowered = _country['name'].lower()
                            if _lowered not in locations and _lowered in originals:
                                locations.add(_lowered)
                                entities[self.LABEL_LOCATION].append(_country['name'])
                        else:
                            assert False, f"unexpected country {_country}."
                # end COUNTRY

                # --------------------------------------------------------------------------------
                # PLACE
                # --------------------------------------------------------------------------------
                if key.strip().lower() in [
                    "place"
                ]:
                    assert isinstance(values, (list, dict)), \
                        f"expected JSON for PLACE, got {values}."

                    if isinstance(values, dict):
                        for name, information in values.items():
                            assert isinstance(information, dict), \
                                f"expected dict for PLACE element, got {information}"

                            _lowered = name.lower()

                            if _lowered not in originals:
                                # GPT can make up a location that is not in original locations.
                                # For instance, categorize may generate 'Helsinki, Finland' for 'Finland'
                                # although 'Helsinki' does not exist in locations before categorize.
                                # Those locations returned from categorize must exist in original.
                                del _lowered
                                continue

                            # categorize may generate {
                            #     "name": "Finland",
                            #     "country": "Finland"
                            # }
                            # Avoid generating "Finland, Finland" as a location.
                            if (
                                    'state' in information
                                    and information['state']
                                    and information['state'] not in name
                            ):
                                name += f", {information['state']}"

                            if (
                                    'country' in information
                                    and information['country']
                                    and information['country'] not in name
                            ):
                                name += f", {information['country']}"

                            if name.lower() not in locations:
                                locations.add(name.lower())
                                entities[self.LABEL_LOCATION].append(name)

                    elif isinstance(values, list):
                        for information in values:
                            assert isinstance(information, dict), \
                                f"expected dictionary as location, got {information}."
                            assert 'name' in information, f"expected 'name' element in {information}."

                            name: str = information['name']
                            _lowered = name.lower()
                            if _lowered not in originals:
                                # GPT can make up a location that not in original locations.
                                # For instance, 'Helsinki, Finland' for 'Finland' at categorize.
                                # Those locations returned from categorize must exist in original.
                                del _lowered
                                continue

                            # categorize may generate {
                            #     "name": "Finland",
                            #     "country": "Finland"
                            # }
                            # Avoid generating "Finland, Finland" as a location.
                            if (
                                    'state' in information
                                    and information['state']
                                    and information['state'] not in name
                            ):
                                name += f", {information['state']}"

                            if (
                                    'country' in information
                                    and information['country']
                                    and information['country'] not in name
                            ):
                                name += f", {information['country']}"

                            if name.lower() not in locations:
                                locations.add(name.lower())
                                entities[self.LABEL_LOCATION].append(name)
                # end PLACE
            # end for key, values in categorized_locations.items()

            _logger.debug(
                "%s: location after categorize:%s", _func_name, entities[self.LABEL_LOCATION]
            )
        # end do_categorize

        return entities
