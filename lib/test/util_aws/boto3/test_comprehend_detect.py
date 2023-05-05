"""Module to test ComprehendDetect class"""
import json
from typing import (
    List,
    Dict,
    Any,
)

import pytest

# pylint: disable=import-error
from util_aws.boto3.comprehend import (
    ComprehendDetect
)
import boto3

print(pytest.__version__)
comprehend: ComprehendDetect = ComprehendDetect(comprehend_client=boto3.client('comprehend'))


# --------------------------------------------------------------------------------
# Supported language codes
# --------------------------------------------------------------------------------
def test_comprehend_detect_supported_language_codes():
    """Test AWS comprehend detect_entities"""
    assert comprehend.is_language_code_supported(language_code="en")
    assert not comprehend.is_language_code_supported(language_code="hoge")


# --------------------------------------------------------------------------------
# Dominant language detection
# --------------------------------------------------------------------------------
def test_comprehend_detect_dominant_language__return_language_code_only():
    """Test language detection
    """
    detected_language_code_for_en: List[str] = comprehend.detect_dominant_language(
        text="I love sushi", return_language_code_only=True
    )
    assert detected_language_code_for_en[0] == "en", \
        f"expected en, got {detected_language_code_for_en[0]}."

    detected_language_code_for_ja: List[str] = comprehend.detect_dominant_language(
        text="すし大好き", return_language_code_only=True
    )
    assert detected_language_code_for_ja[0] == "ja", \
        f"expected ja, got {detected_language_code_for_ja[0]}."

    detected_language_code_for_es: List[str] = comprehend.detect_dominant_language(
        text="Me encanta el sushi", return_language_code_only=True
    )
    assert detected_language_code_for_es[0] == "es", \
        f"expected es, got {detected_language_code_for_es[0]}."


def test_comprehend_detect_dominant_language():
    """Test language detection
    """
    detected_language_code_for_en: List[Dict[str, Any]] = comprehend.detect_dominant_language(
        text="I love sushi", return_language_code_only=False
    )
    assert detected_language_code_for_en[0]['LanguageCode'] == "en", \
        f"expected language code is en, got {detected_language_code_for_en[0]}."
    assert isinstance(detected_language_code_for_en[0]['Score'] , float), \
        f"expected score as float, got {type(detected_language_code_for_en[0]['Score'])}."

    detected_language_code_for_ja: List[Dict[str, Any]] = comprehend.detect_dominant_language(
        text="すし大好き", return_language_code_only=False
    )
    assert detected_language_code_for_ja[0]['LanguageCode'] == "ja", \
        f"expected language code is ja, got {detected_language_code_for_ja[0]}."
    assert isinstance(detected_language_code_for_ja[0]['Score'] , float), \
        f"expected score as float, got {type(detected_language_code_for_ja[0]['Score'])}."


# --------------------------------------------------------------------------------
# Entity detections
# --------------------------------------------------------------------------------
def test_comprehend_detect_entities__fail_with_no_language_code():
    """Test if entity detection fails there is no language code.
    Test conditions:
        1. detect_entities() raise Exception when language_code is not specified
           and auto_detect_language is False.
    """
    try:
        entities = comprehend.detect_entities(
            text="During the civil war from 12 Apr 1861",
            language_code=None,
            auto_detect_language=False
        )
        assert False, f"expected exception, got {entities}."
    except (AssertionError, RuntimeError):
        pass


def test_comprehend_detect_entities__fail_with_non_supported_language_code():
    """Test entity detection fails with non-supported language code.
    Test conditions:
        1. detect_entities() raise Exception when language_code is not supported.
    """
    try:
        entities = comprehend.detect_entities(
            text="During the civil war from 12 Apr 1861",
            language_code="invalid",
            auto_detect_language=False
        )
        assert False, f"expected exception, got {entities}."
    except (AssertionError, RuntimeError):
        pass


def test_comprehend_detect_entities__en():
    """Test entity detections in English text
    Test conditions:
        From the text including PERSON, LOCATION, DATE more than threshold certainty.
        1. Detect PERSON, e.g. 'Abraham Lincoln'
        2. Detect LOCATION, e.g. 'Kentucky'
        3. Detect DATE, e.g. '9 Apr 1865'
    """
    person: str = "Abraham Lincoln"
    location: str = "Kentucky"
    date: str = "9 Apr 1865"

    person_certainty: float = 0.
    location_certainty: float = 0.
    date_certainty: float = 0.
    threshold: float = 0.9

    entities: List[Dict[str, Any]] = comprehend.detect_entities(
        text=f"""During the civil war from 12 Apr 1861 – {date},
        {person} was the president and leading The Union.
        He was born in {location}.
        """,
        language_code="en"
    )
    listing = json.dumps(entities, indent=4, default=str, ensure_ascii=False)
    # print(listing)
    assert len(entities) > 0, f"expected detections, got [{entities}]"
    for entity in entities:
        # Test condition #1
        if entity['Text'] == person and entity['Type'] == "PERSON":
            person_certainty = float(entity['Score'])

        # Test condition #2
        if entity['Text'] == location and entity['Type'] == "LOCATION":
            location_certainty = float(entity['Score'])

        # Test condition #3
        if entity['Text'] == date and entity['Type'] == "DATE":
            date_certainty = float(entity['Score'])

    # Test condition #1
    assert person_certainty > threshold, \
        f"expected detecting {person} as DATE with {person_certainty} > {threshold} in\n{listing}"

    # Test condition #2
    assert location_certainty > threshold, \
        f"expected detecting {location} as DATE with {location_certainty} > {threshold} in\n{listing}"

    # Test condition #3
    assert date_certainty > threshold, \
        f"expected detecting {date} as DATE with {date_certainty} > {threshold} in\n{listing}"


def test_comprehend_detect_entities__ja():
    """Test entity detections in Japanese text
    Test conditions:
        From the text including PERSON, LOCATION, DATE more than threshold certainty.
        1. Detect PERSON, e.g. "エイブラハム　リンカーン"
        2. Detect LOCATION, e.g. "ケンタッキー州"
        3. Detect DATE, e.g. "1865年4月９日"
    """
    person: str = "エイブラハム　リンカーン"
    location: str = "ケンタッキー州"
    date: str = "1865年4月９日"

    person_certainty: float = 0.
    location_certainty: float = 0.
    date_certainty: float = 0.
    threshold: float = 0.9

    text: str = f"１８６１年４月12日から{date}の南北戦争中、" \
                f"{person}は大統領として合衆国を率いた。" \
                f"彼は{location}の生まれであった。"

    entities: List[Dict[str, Any]] = comprehend.detect_entities(
        text=text,
        language_code="ja"
    )
    listing = json.dumps(entities, indent=4, default=str, ensure_ascii=False)
    # print(listing)
    assert len(entities) > 0, f"expected detections, got [{entities}]"
    for entity in entities:
        # Test condition #1
        if entity['Text'] == person and entity['Type'] == "PERSON":
            person_certainty = float(entity['Score'])

        # Test condition #2
        if entity['Text'] == location and entity['Type'] == "LOCATION":
            location_certainty = float(entity['Score'])

        # Test condition #3
        if entity['Text'] == date and entity['Type'] == "DATE":
            date_certainty = float(entity['Score'])

    # Test condition #1
    assert person_certainty > threshold, \
        f"expected detecting {person} as DATE with {person_certainty} > {threshold} in\n{listing}"

    # Test condition #2
    assert location_certainty > threshold, \
        f"expected detecting {location} as DATE with {location_certainty} > {threshold} in\n{listing}"

    # Test condition #3
    assert date_certainty > threshold, \
        f"expected detecting {date} as DATE with {date_certainty} > {threshold} in\n{listing}"


# --------------------------------------------------------------------------------
# Detect entities with auto language detection
# --------------------------------------------------------------------------------
def test_comprehend_detect_entities__auto_detect_language_ja():
    """Test entity detections with auto-detect language
    Test conditions:
        From the text including PERSON, LOCATION, DATE more than threshold certainty.
        1. Detect PERSON, e.g. "エイブラハム　リンカーン"
        2. Detect LOCATION, e.g. "ケンタッキー州"
        3. Detect DATE, e.g. "1865年4月９日"
    """
    person: str = "エイブラハム　リンカーン"
    location: str = "ケンタッキー州"
    date: str = "1865年4月９日"

    person_certainty: float = 0.
    location_certainty: float = 0.
    date_certainty: float = 0.
    threshold: float = 0.9

    text: str = f"１８６１年４月12日から{date}の南北戦争中、" \
                f"{person}は大統領として合衆国を率いた。" \
                f"彼は{location}の生まれであった。"

    entities: List[Dict[str, Any]] = comprehend.detect_entities(
        text=text,
        language_code=None,
        auto_detect_language=True
    )
    listing = json.dumps(entities, indent=4, default=str, ensure_ascii=False)
    assert len(entities) > 0, f"expected detections, got [{entities}]"
    for entity in entities:
        # Test condition #1
        if entity['Text'] == person and entity['Type'] == "PERSON":
            person_certainty = float(entity['Score'])

        # Test condition #2
        if entity['Text'] == location and entity['Type'] == "LOCATION":
            location_certainty = float(entity['Score'])

        # Test condition #3
        if entity['Text'] == date and entity['Type'] == "DATE":
            date_certainty = float(entity['Score'])

    # Test condition #1
    assert person_certainty > threshold, \
        f"expected detecting {person} as DATE with {person_certainty} > {threshold} in\n{listing}"

    # Test condition #2
    assert location_certainty > threshold, \
        f"expected detecting {location} as DATE with {location_certainty} > {threshold} in\n{listing}"

    # Test condition #3
    assert date_certainty > threshold, \
        f"expected detecting {date} as DATE with {date_certainty} > {threshold} in\n{listing}"
