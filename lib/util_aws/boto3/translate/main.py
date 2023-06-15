"""Module for AWS Translate operations using Boto3"""
import json
import logging
from typing import (
    List,
    Any,
)

# pylint: disable=import-error
from util_aws.boto3.common import (
    Base
)

from util_logging import (              # pylint: disable=import-error
    get_logger
)

import botocore


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# AWS Translate
# --------------------------------------------------------------------------------
class Translate(Base):
    """Class to provide AWS Translate functions."""
    # pylint: disable=too-few-public-methods
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    def __init__(self, translate_client):
        """
        Args:
            translate_client: A Boto3 Translate client.
        """
        super().__init__()
        self._client = translate_client

    def list_languages(
            self,
            return_language_code_only: bool = True
    ) -> List[str]:
        """Provides a list of languages (RFC-5646 codes and names) that Amazon Translate supports.
        https://docs.aws.amazon.com/translate/latest/APIReference/API_ListLanguages.html
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/translate/client/list_languages.html
        """
        response = self._client.list_languages(
            DisplayLanguageCode='en',
            MaxResults=200,
        )

        results: List[Any] = response['Languages']
        while "NextToken" in response:
            response = self._client.list_languages(
                DisplayLanguageCode='en',
                NextToken=response["NextToken"]
            )
            results.extend(response["Languages"])

        return [
            language['LanguageCode'] for language in results
        ] if return_language_code_only else results

    def translate_text(self, text: str, source_language_code: str, target_language_code: str):
        """
        Translate text in source language to the target language
        Args:
            text: text in UTF-8 encoding to translate
            source_language_code: language code of the text to translate
            target_language_code: target language code to translate to.
        Returns: translated text
        Raises:
            RuntimeError: AWS API call failure
            ValueError: Invalid values identified
        """
        name: str = "translate_text()"

        # --------------------------------------------------------------------------------
        # Validate and clean text strings
        # --------------------------------------------------------------------------------
        text = self.validate_text(text=text)
        source_language_code = self.validate_text(source_language_code)
        target_language_code = self.validate_text(target_language_code)

        # --------------------------------------------------------------------------------
        # Translation
        # --------------------------------------------------------------------------------
        try:
            _logger.debug(
                "%s: source language:[%s] target language:[%s] text:\n[%s]",
                name,
                source_language_code,
                target_language_code,
                text
            )
            if source_language_code == target_language_code:
                _logger.debug(
                    "source and target languages are the same [%s]. returning text as is.",
                    source_language_code
                )
                return text

            response = self._client.translate_text(
                Text=text,
                SourceLanguageCode=source_language_code,
                TargetLanguageCode=target_language_code
            )
            _logger.debug("translated:%s", json.dumps(response, indent=4, default=str))

        except self._client.exceptions.TextSizeLimitExceededException as error:
            msg: str = f"length of the text [{len(text)}] exceeded the max size.\ncause:[{error}]"
            _logger.error("%s: %s error: %s", name, msg, error)
            raise ValueError(msg)

        except botocore.exceptions.ParamValidationError as error:
            msg: str = f"invalid parameter. check if source_language_code:[{source_language_code}] " \
                       f"and target_language_code:[{target_language_code}] are correct.\ncause:[{error}]"
            _logger.error("%s: %s", name, msg)
            raise ValueError(msg) from error

        except botocore.exceptions.ClientError as error:
            msg: str = f"translation from source_language_code:[{source_language_code}] to " \
                       f"target_language_code:[{target_language_code}].\ncause:[{error}]."
            _logger.error("%s: %s", name, msg)
            raise RuntimeError(msg) from error

        return response['TranslatedText']
