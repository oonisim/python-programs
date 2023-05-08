"""Module to test ComprehendDetect class"""
import logging

# pylint: disable=import-error
from util_aws.boto3.translate import (
    Translate
)

import boto3


translate: Translate = Translate(translate_client=boto3.client('translate'))


# --------------------------------------------------------------------------------
# Dominant language detection
# --------------------------------------------------------------------------------
def test_translate_text__fail_with_invalid_language_code():
    """Test translate fails when the language is not valid.
    Test conditions:
    1. Fails when source_language_code is not valid.
    2. Fails when target_language_code is not valid.
    """
    try:
        translate.translate_text(
            text="i love sushi", source_language_code="invalid", target_language_code="es"
        )
        # Test condition #1
        assert False, "expected RuntimeError for the invalid source_language_code."
    except RuntimeError as error:
        logging.debug(error)

    try:
        translate.translate_text(
            text="i love sushi", source_language_code="en", target_language_code="invalid"
        )
        # Test condition #2
        assert False, "expected RuntimeError for the invalid target_language_code."
    except RuntimeError as error:
        logging.debug(error)


def test_translate_text__fail_with_invalid_text():
    """Test translate fails when the text is not valid.
    """
    try:
        translate.translate_text(
            text=None, source_language_code="en", target_language_code="es"
        )
        assert False, "expected RuntimeError for the invalid text."
    except RuntimeError as error:
        logging.debug(error)

    try:
        translate.translate_text(
            text="   ", source_language_code="en", target_language_code="es"
        )
        assert False, "expected RuntimeError for the invalid text."
    except RuntimeError as error:
        logging.debug(error)


def test_translate_text__same_source_and_target_language_codes():
    """Test translate_text skips translation.
    """
    text: str = "I love sushi."
    translation: str = translate.translate_text(
        text=text, source_language_code="en", target_language_code="en"
    )
    assert text == translation, f"expected [{text} got [{translation}]."


def test_traslate_text__en_es():
    """Test translate EN to ES.
    """
    text: str = "English"
    expected: str = "Inglés".lower()

    translation: str = translate.translate_text(
        text=text, source_language_code="en", target_language_code="es"
    )
    assert translation.lower() == expected, f"expected [{expected} got [{translation}]."
