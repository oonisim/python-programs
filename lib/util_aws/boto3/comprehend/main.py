"""Module for AWS Comprehend operation with Boto3
"""
import logging
from typing import (
    List,
    Dict,
    Set,
    Any,
    Optional,
    Callable,
)
from functools import wraps

from util_aws.boto3.common import (     # pylint: disable=import-error
    Base
)

import botocore


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
# https://docs.aws.amazon.com/comprehend/latest/dg/how-languages.html
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
    "zh-TW".lower()
]
# https://docs.aws.amazon.com/comprehend/latest/dg/how-entities.html
SUPPORTED_ENTITY_TYPES: List[str] = [
    "COMMERCIAL_ITEM",
    "DATE",
    "EVENT",
    "LOCATION",
    "ORGANIZATION",
    "OTHER",
    "PERSON",
    "QUANTITY",
    "TITLE"
]


# --------------------------------------------------------------------------------
# AWS service class
# --------------------------------------------------------------------------------
class ComprehendDetect(Base):
    """Class to provide the AWS Comprehend detection functions.
    """
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------
    @staticmethod
    def is_language_code_supported(language_code: str):
        """Check if the language code is supported by Comprehend
        """
        assert isinstance(language_code, str)
        return language_code.lower() in SUPPORTED_LANGUAGES_CODES
    
    @staticmethod
    def is_entity_type_supported(entity_type: str):
        """Check if the entity type is supported by Comprehand"""
        assert isinstance(entity_type, str)
        return entity_type.upper() in SUPPORTED_ENTITY_TYPES

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

        Raises:
            ValueError: argument values are invalid
            RuntimeError: AWS API call failed
        """
        name: str = "detect_dominant_language()"

        # --------------------------------------------------------------------------------
        # Validate and clean text
        # --------------------------------------------------------------------------------
        text = self.validate_text(text=text)

        # --------------------------------------------------------------------------------
        # Detect language(s)
        # --------------------------------------------------------------------------------
        try:
            response = self.comprehend_client.detect_dominant_language(
                Text=text
            )
            detections: List[Dict[str, Any]] = sorted(
                response['Languages'],
                key=lambda detection: detection['Score']
            )
            _logger.debug("%s: detected languages %s", name, detections)

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
            auto_detect_language: use language auto-detection

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

        Raises:
            ValueError: argument values are invalid
            RuntimeError: AWS API call failed
        """
        name: str = "detect_entities()"
        assert (
            (auto_detect_language is False and isinstance(language_code, str)) or
            (auto_detect_language is True and language_code is None)
        ), \
            f"invalid combination of auto_detect_language:[{auto_detect_language}] " \
            f"and language_code:[{language_code}]"

        # --------------------------------------------------------------------------------
        # Validate and clean text
        # --------------------------------------------------------------------------------
        text = self.validate_text(text=text)

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
            return entities
        
        except self.comprehend_client.exceptions.TextSizeLimitExceededException as error:
            msg: str = f"length of the text [{len(text)}] exceeded the max size.\ncause:[{error}]"
            _logger.error("%s: %s error: %s", name, msg, error)
            raise ValueError(msg) from error

        except botocore.exceptions.ParamValidationError as error:
            msg: str = f"invalid parameter. check if language_code:[{language_code}] is correct." \
                       f"\ncause:[{error}]"
            _logger.error("%s: %s", name, msg)
            raise ValueError(msg) from error

        except botocore.exceptions.ClientError as error:
            msg: str = f"Comprehend.detect_entities() failed due to {error}\n" \
                       f"text=[{text}]\nlanguage_code=[{language_code}]."
            _logger.error("%s", msg)
            raise RuntimeError(msg) from error

    def detect_entities_by_type(
            self,
            text: str,
            language_code: str,
            auto_detect_language: bool = False,
            entity_types: List[str] = None,
            score_threshold: float = 0.0,
            sort_by_score: bool = True,
            return_entity_value_only: bool = False
    ) -> List[Any]:
        """Detect entities of the entity types specified.
        Args:
            text: text to extract entities from.
            language_code: language code of the text
            auto_detect_language: flat to use language code auto-detection
            entity_types:
                types of entities that AWS Comprehend identifies.
                See https://docs.aws.amazon.com/comprehend/latest/dg/how-entities.html
            score_threshold: minimum score below which to omit the entity
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
        Returns:
            list of values of the entity_type if return_entity_value_only is True.
            Otherwise, list of entities
        """
        name: str = "detect_entities_by_type()"
        result: Optional[List[Any]] = None

        # --------------------------------------------------------------------------------
        # Validate and clean text
        # --------------------------------------------------------------------------------
        text = self.validate_text(text=text)

        # --------------------------------------------------------------------------------
        # Detect entities
        # --------------------------------------------------------------------------------
        detected_entities: List[Any] = self.detect_entities(
            text=text, language_code=language_code, auto_detect_language=auto_detect_language
        )

        # --------------------------------------------------------------------------------
        # If no entity types, return the detected entities as is. ignore score_threshold.
        # --------------------------------------------------------------------------------
        if not entity_types:
            return sorted(
                detected_entities, reverse=True, key=lambda _entity: _entity['Score']
            ) if sort_by_score else detected_entities

        # --------------------------------------------------------------------------------
        # Extract entity of the specified entity_types
        # --------------------------------------------------------------------------------
        assert len(entity_types) > 0 and isinstance(entity_types[0], str), \
            f"invalid entity_types:[{entity_types}]."

        entity_type_set: Set[str] = {
            _type.lower()
            for _type in entity_types
            if self.is_entity_type_supported(_type)
        }
        if len(entity_type_set) == 0:
            msg: str = f"the entity types provided {entity_types} are not in "\
                       f"valid types {SUPPORTED_ENTITY_TYPES}."
            _logger.error("%s %s", name, msg)
            raise ValueError(msg)

        if return_entity_value_only:
            # Extract Entity['Text']. There will be multiple entities with the same text
            # because the same text can exist at multiple locations in the text document.
            # When find the same text, take the one with the higher score. If there are
            # multiple with same score, then the first one.
            entities: Dict[str, float] = {}
            for entity in detected_entities:
                if entity['Type'].lower() in entity_type_set:
                    if entity['Score'] >= score_threshold:
                        text: str = entity['Text']
                        if text not in entities or entities[text] < entity['Score']:
                            # Create {text: score} pair and sort by score later in sorted().
                            entities[text] = entity['Score']

            # reverse=True to sort descending (0.98, 0.82, 0.77, ...) in number.
            result = sorted(entities, key=entities.get, reverse=True) if sort_by_score else entities

        else:
            entities: List[Dict[str, Any]] = []
            for entity in detected_entities:
                if entity['Type'].lower() in entity_type_set:
                    if entity['Score'] >= score_threshold:
                        entities.append(entity)

            # reverse=True to sort descending (0.98, 0.82, 0.77, ...) in number.
            result = sorted(
                entities, reverse=True, key=lambda _entity: _entity['Score']
            ) if sort_by_score else entities

        return result

    def detect_pii_entities(
            self,
            text: str,
            language_code: Optional[str],
            auto_detect_language: bool = False
    ) -> List[Any]:
        """
        Detects PII entities in a document.
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehend/client/detect_pii_entities.html

        Args:
            text: text to extract entities from.
            language_code: language code of the text. It can be omitted if  auto_detect_language is True
            auto_detect_language: use language auto-detection

        Returns:
            The list of entities along with their confidence scores.
            Example
            [
                {
                    'Score': ...,
                    'Type': BANK_ACCOUNT_NUMBER,
                    'BeginOffset': 123,
                    'EndOffset': 123
                },
                ...
            ]

        Raises:
            ValueError: argument values are invalid
            RuntimeError: AWS API call failed
        """
        name: str = "detect_pii_entities()"
        assert (
            (auto_detect_language is False and isinstance(language_code, str)) or
            (auto_detect_language is True and language_code is None)
        ), \
            f"invalid combination of auto_detect_language:[{auto_detect_language}] " \
            f"and language_code:[{language_code}]"

        # --------------------------------------------------------------------------------
        # Validate and clean text
        # --------------------------------------------------------------------------------
        text = self.validate_text(text=text)

        # --------------------------------------------------------------------------------
        # Auto language detection if specified
        # --------------------------------------------------------------------------------
        if auto_detect_language:
            language_code = self.detect_dominant_language(text=text, return_language_code_only=True)[0]
            _logger.debug("%s: auto detected language code is [%s]", name, language_code)

        language_code = self.validate_text(language_code)
        if not self.is_language_code_supported(language_code):
            msg: str = f"language code detected or specified [{language_code}] is not supported."
            _logger.error("%s: %s", name, msg)
            raise ValueError(msg)
            
        # --------------------------------------------------------------------------------
        # Entity detections
        # --------------------------------------------------------------------------------
        # TODO: Handle errors by a decorator
        try:
            response: Dict[str, Any] = self.comprehend_client.detect_pii_entities(
                Text=text, LanguageCode=language_code
            )
            entities: List[Any] = response['Entities']
            for _entity in entities:
                start: int = int(_entity['BeginOffset'])
                end: int = int(_entity['EndOffset'])
                _entity['Text'] = text[start:end]
            
            _logger.debug("%s: number of entities detected is [%s].", name, len(entities))

        except self.comprehend_client.exceptions.TextSizeLimitExceededException as error:
            msg: str = f"length of the text [{len(text)}] exceeded the max size.\ncause:[{error}]"
            _logger.error("%s: %s error: %s", name, msg, error)
            raise ValueError(msg) from error

        except botocore.exceptions.ParamValidationError as error:
            msg: str = f"invalid parameter. check if language_code:[{language_code}] is correct." \
                       f"\ncause:[{error}]"
            _logger.error("%s: %s", name, msg)
            raise ValueError(msg) from error

        except botocore.exceptions.ClientError as error:
            msg: str = f"Comprehend.detect_entities() failed due to {error}\n" \
                       f"text=[{text}]\nlanguage_code=[{language_code}]."
            _logger.error("%s", msg)
            raise RuntimeError(msg) from error

        return entities