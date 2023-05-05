"""Module for AWS Comprehend operation with Boto3
"""
import logging
from typing import (
    List,
    Dict,
    Set,
    Any,
    Optional,
)

import botocore

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
DEFAULT_LOG_LEVEL_NAME: str = "ERROR"
SUPPORTED_LANGUAGES_CODES: List[str] = [  # As of 04MAY2023
    "de",
    "en",
    "es",
    "it",
    "pt",
    "fr",
    "ja",
    "ko",
    "hi",
    "ar",
    "zh",
    "zh-TW"
]


# --------------------------------------------------------------------------------
# AWS service class
# --------------------------------------------------------------------------------
class ComprehendDetect:
    """Class to provide the AWS Comprehend detection functions.
    """
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------
    @staticmethod
    def is_language_code_supported(language_code: str):
        """Check if the language code is supported by Comprehend
        """
        return language_code in SUPPORTED_LANGUAGES_CODES

    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    def __init__(self, comprehend_client):
        """
        Args:
             comprehend_client: A Boto3 Comprehend client.
        """
        self.comprehend_client = comprehend_client

    def detect_dominant_language(
            self,
            text: str,
            return_language_code_only: bool = False
    ) -> List[Any]:
        """Detect dominant languages in the text.
        Returns:
            If return_language_code_only is True, single language code that has the
            highest score, or sorted list of {
                'LanguageCode': str,
                'Score': float
            }

        Raises: RuntimeError when API call fails
        """
        try:
            response = self.comprehend_client.detect_dominant_language(
                Text=text
            )
            detections: List[Dict[str, Any]] = sorted(
                response['Languages'],
                key=lambda detection: detection['Score']
            )
            _logger.debug("detected languages %s", detections)

            if return_language_code_only:
                detections = [
                    detection['LanguageCode']
                    for detection in detections
                ]

        except botocore.exceptions.ClientError as error:
            msg: str = f"Comprehend.detect_dominant_language() failed due to {error}\n" \
                       f"text=[{text}]."
            _logger.error(msg)
            raise RuntimeError(msg) from error

        return detections

    def detect_entities(
            self,
            text: str,
            language_code: Optional[str],
            auto_detect_language: bool = False
    ) -> List[Any]:
        """
        Detects entities in a document. Entities can be things like people and places
        or other common terms.

        Args:
            text: text to extract entities from.
            language_code: language code of the text. It can be omitted if  auto_detect_language is True
            auto_detect_language: use language auto detection

        Returns:
            The list of entities along with their confidence scores.
            Example
            [
                {
                    "BeginOffset": 2,
                    "EndOffset": 5,
                    "Score": 0.5963479280471802,
                    "Text": "Tako",
                    "Type": "TITLE"
                },
                {
                    "BeginOffset": 22,
                    "EndOffset": 31,
                    "Score": 0.9983485341072083,
                    "Text": "Ika",
                    "Type": "TITLE"
                },
                ...
            ]

        Raises: RuntimeError when Boto3 failed e.g. language_code is not supported.
        """
        name: str = "detect_entities()"
        assert (
            (auto_detect_language is False and isinstance(language_code, str)) or
            (auto_detect_language is True and language_code is None)
        ), \
            f"invalid combination of auto_detect_language:[{auto_detect_language}] " \
            f"and language_code:[{language_code}]"

        # --------------------------------------------------------------------------------
        # Auto language detection if specified
        # --------------------------------------------------------------------------------
        if auto_detect_language:
            language_code = self.detect_dominant_language(text=text, return_language_code_only=True)[0]
            _logger.debug("%s: auto detected language code is [%s]", name, language_code)

        # --------------------------------------------------------------------------------
        # Entity detections
        # --------------------------------------------------------------------------------
        try:
            response: Dict[str, Any] = self.comprehend_client.detect_entities(
                Text=text, LanguageCode=language_code
            )
            entities: List[Any] = response['Entities']
            _logger.debug("%s: number of entities detected is [%s].", name, len(entities))

        except botocore.exceptions.ClientError as error:
            msg: str = f"Comprehend.detect_entities() failed due to {error}\n" \
                       f"text=[{text}]\nlanguage_code=[{language_code}]."
            _logger.error(msg)
            raise RuntimeError(msg) from error

        return entities

    def detect_entities_by_type(
            self,
            text: str,
            language_code: str,
            auto_detect_language: bool = False,
            entity_types: List[str] = None,
            sort_by_score: bool = True,
            return_entity_value_only: bool = False,
    ) -> List[Any]:
        """Detect entities of the entity types specified.
        Args:
            text: text to extract entities from.
            language_code: language code of the text
            entity_types:
                types of entities that AWS Comprehend identifies.
                See https://docs.aws.amazon.com/comprehend/latest/dg/how-entities.html
            sort_by_score:
                return entities sorted by entity score in descending order.
                Entities with higher score come first.
            return_entity_value_only:
                When True, return entity values only.
                For instance, return ["Tako", "Ika"] when entities = [
                    {
                        "BeginOffset": 2,
                        "EndOffset": 5,
                        "Score": 0.5963479280471802,
                        "Text": "Tako",
                        "Type": "TITLE"
                    },
                    {
                        "BeginOffset": 22,
                        "EndOffset": 31,
                        "Score": 0.9983485341072083,
                        "Text": "Ika",
                        "Type": "PERSON"
                    },
                    ...
                ]

        """
        result = None
        detected_entities: List[Any] = self.detect_entities(
            text=text, language_code=language_code, auto_detect_language=auto_detect_language
        )

        if not entity_types:
            return detected_entities

        assert len(entity_types) > 0 and isinstance(entity_types[0], str), \
            f"invalid entity_types:[{entity_types}]."

        entity_type_set: Set[str] = {
            _type.lower()
            for _type in entity_types
        }

        if return_entity_value_only:
            # Extract Entity['Text']. There will be multiple entities with the same text
            # because the same text can exist at multiple locations in the text document.
            # When find the same text, take the one with the higher score. If there are
            # multiple with same score, then the first one.
            entities = {}
            for entity in detected_entities:
                if entity['Type'].lower() in entity_type_set:
                    text: str = entity['Text'].lower()
                    if text not in entities or entities[text] < entity['Score']:
                        # Create {text: score} pair and sort by score later in sorted().
                        entities[text] = entity['Score']

            # reverse=True to sort descending (0.98, 0.82, 0.77, ...) in number.
            result = sorted(entities, key=entities.get, reverse=True) if sort_by_score else entities

        else:
            entities = []
            for entity in detected_entities:
                if entity['Type'].lower() in entity_type_set:
                    entities.append(entity)

            # reverse=True to sort descending (0.98, 0.82, 0.77, ...) in number.
            result = sorted(
                entities, reverse=True, key=lambda _entity: _entity['Score']
            ) if sort_by_score else entities

        assert result is not None
        return result
