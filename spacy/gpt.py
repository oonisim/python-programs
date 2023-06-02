"""Module for Chat completion"""
import json
import logging
from typing import (
    List,
    Dict,
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
    'culture', 'sports', 'war', 'crime', 'politics', 'technology', 'food',
    'lifestyle', 'science', 'business', "society", "philosophy", "innovation",
    "relationship", "economy", "energy", "history"
]


def _to_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as _error:
        msg: str = f"cannot decode to JSON from the GPT response [{text}]"
        _logger.error("%s due to [%s].", msg, _error)
        raise RuntimeError from _error


# --------------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------------
class ChatTaskForTextTagging(OpenAI):
    """Class for text tagging using Open AI chat task operations"""
    TAG_ENTITY_TYPE_PERSON: str = "PERSON"
    TAG_ENTITY_TYPE_LOCATION: str = "LOCATION"
    TAG_ENTITY_TYPE_ORGANIZATION: str = "ORGANIZATION"
    TAG_ENTITY_TYPES = {
        TAG_ENTITY_TYPE_PERSON,
        TAG_ENTITY_TYPE_LOCATION,
        TAG_ENTITY_TYPE_ORGANIZATION
    }

    @staticmethod
    def tag_entity_types():
        """List of text tag entity types"""
        return ChatTaskForTextTagging.TAG_ENTITY_TYPES

    def __init__(self, path_to_api_key: str):
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
        prompt = "Critical events in the TEXT. " \
                 f"Return as JSON where key is event as noun phrase within {max_words} words " \
                 f"and value is explanation. TEXT={text}"

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

    def get_geographic_locations(self, text: str, theme: Optional[str], top_n: int = 6) -> Dict[str, Any]:
        """Get geographical locations from the text.
        Args:
            text: text to extract the key events from.
            theme: theme of the text to which the organizations are related with
            top_n: number of organizations to return
        Returns: organizations as JSON/dictionary in the form of {
            <name>: <description>
        }
        """
        prompt = f"{top_n} geographic locations in the TEXT where the THEME '{theme}' is related. " \
                 f"The locations must exist in the Google map. " \
                 "Return as a JSON object where the key is geographic location and the value is explanation " \
                 "why the location is important to the THEME. " \
                 f"Return JSON null if there is no geographic locations. TEXT={text}"

        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

    def get_keywords(self, text: str, theme: str, top_n: int = 5) -> Dict[str, Any]:
        """Get keywords from the text that directly induce the theme.
        Args:
            text: text to extract the key events from.
            theme: theme of the text to which the organizations are related with
            top_n: number of organizations to return
        Returns: organizations as JSON/dictionary in the form of {
            <name>: <description>
        }
        """
        # prompt = f"Top {top_n} most important keywords in the TEXT as a JSON list. TEXT={text}."
        # prompt = f"Order top {top_n} important keywords from the TEXT that directly induce '{theme}'. " \
        #          f"Return as a JSON list. " \
        #          f"TEXT={text}."
        prompt = f"Top {top_n} critical keywords from the TEXT that directly induce '{theme}'. " \
                 f"Return as a JSON list. " \
                 f"Order by importance. " \
                 f"TEXT={text}."

        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))

    def distill(
            self,
            text: str,
            theme: str,
            max_words: int = 25,
            top_n: int = 6
    ) -> Dict[str, Any]:
        """
        Get theme, summary, keywords, key phrases, people, locations from the text
        that directly induce the theme.

        Args:
            text: text to extract the key events from.
            theme: theme of the text to which the organizations are related with
            max_words: max words for summary
            top_n:

        Returns: organizations as JSON/dictionary in the form of {
            "keywords": KEYWORDS,
            "phrases": NOUN PHRASES,
            "people": KEY FIGURES,
            "locations": GEOGRAPHIC LOCATIONS
        }
        """
        prompt_that_takes_longer = f"""
THEME is '{theme}'.

Top {top_n} KEYWORDS from the TEXT that induces the THEME.
Top {top_n} PERSON as title and name from the TEXT who participated to the THEME.
Top {top_n} ORGANIZATIONS as name and explanation that induced the THEME in the the TEXT. 
Top {top_n} GEOGRAPHIC LOCATIONS where the THEME occurs in the TEXT.

Return a JSON in the following format that the python json.loads method can handle.
{{
    "KEYWORD": [{{keyword:explanation}}] or [],
    "PERSON": [{{name:title}}] or [],
    "ORGANIZATION": [{{name:explanation}}] or [],
    "LOCATION": [{{location:explanation}}] or []
}}

TEXT={text}
"""
        prompt_replaced = f"""
THEME is '{theme}'.

Top {top_n} KEYWORDS from the TEXT that induces the THEME.
Top {top_n} PERSON as title and name from the TEXT who participated to the THEME. 
Max {top_n} ORGANIZATIONS that induced the THEME in the the TEXT. Must be less than {top_n+1}. 
Top {top_n} GEOGRAPHIC LOCATIONS where the THEME does occur.

Return a JSON in the following format that the python json.loads method can handle.
{{
    "KEYWORD": KEYWORDS  or [],
    "{self.TAG_ENTITY_TYPE_PERSON}": [{{name:title}}] or [],
    "{self.TAG_ENTITY_TYPE_ORGANIZATION}": ORGANIZATION or [],
    "{self.TAG_ENTITY_TYPE_LOCATION}": GEOGRAPHIC LOCATIONS or []
}}

TEXT={text}

"""
        prompt = f"""
Top {top_n} news categories or topics as KEYWORDS about the NEWS.
Top {top_n} PERSON as title and name who is connected to the incident of the NEWS.
Top {top_n} ORGANIZATIONS that participated in the incident of the NEWS.
Maximum {top_n} GEOGRAPHIC COUNTRY and its LOCATIONS where the incident of the NEWS happened.
Maximum three CATEGORIES of the NEWS such as {', '.join(CATEGORY_EXAMPLES)}.

Return a JSON in the following format that the python json.loads method can handle.
{{
    "KEYWORD": KEYWORDS  or [],
    "{self.TAG_ENTITY_TYPE_PERSON}": [{{name:title}}] or [],
    "{self.TAG_ENTITY_TYPE_ORGANIZATION}": ORGANIZATION or [],
    "{self.TAG_ENTITY_TYPE_LOCATION}": GEOGRAPHIC COUNTRIES and LOCATIONS or []
    "CATEGORY": CATEGORIES  or []
}}

NEWS={text}
"""
        return _to_json(text=self.get_chat_completion_by_prompt(prompt=prompt))
