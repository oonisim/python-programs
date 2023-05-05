"""Module for AWS Translate operations using Boto3"""
import json
import logging

import botocore

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# AWS Translate
# --------------------------------------------------------------------------------
class Translate:
    """Class to provide AWS Translate functions."""
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
        self._client = translate_client

    def translate_text(self, text, source_language_code, target_language_code):
        """
        Translate text in source language to the target language
        Args:
            text: text in UTF-8 encoding to translate
            source_language_code: language code of the text to translate
            target_language_code: target language code to translate to.
        Returns: translated text
        Raises: RuntimeError when Boto3 caused an error.
        """
        try:
            _logger.debug(
                "source language:[%s] target language:[%s] text:\n[%s]",
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

        except botocore.exceptions.ClientError as error:
            msg: str = f"translate_text() from source_language_code:[{source_language_code}] to " \
                       f"target_language_code:[{target_language_code}] failed due to [{error}]."
            _logger.error("%s", msg)
            raise RuntimeError(msg) from error

        return response['TranslatedText']
