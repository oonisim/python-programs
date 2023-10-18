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
        f"expected detecting {person} as PERSON with {person_certainty} > {threshold} in\n{listing}"

    # Test condition #2
    assert location_certainty > threshold, \
        f"expected detecting {location} as LOCATION with {location_certainty} > {threshold} in\n{listing}"

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
    person: str = "エイブラハム リンカーン"    # space is ASCII not multi-byote JP space
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
        f"expected detecting {person} as PERSON with {person_certainty} > {threshold} in\n{listing}"

    # Test condition #2
    assert location_certainty > threshold, \
        f"expected detecting {location} as LOCATION with {location_certainty} > {threshold} in\n{listing}"

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
        1. Detect PERSON, e.g. "エイブラハム リンカーン"
        2. Detect LOCATION, e.g. "ケンタッキー州"
        3. Detect DATE, e.g. "1865年4月９日"
    """
    person: str = "エイブラハム リンカーン"    # space is ASCII not multi-byote JP space
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


# --------------------------------------------------------------------------------
# Entity detections by entity type
# --------------------------------------------------------------------------------
def test_detect_entities_by_type():
    """Test entity detections of a certain entity type e.g. PERSON, LOCATION, DATE
    in English text

    Test conditions:
        From the text including PERSON, LOCATION, DATE,
        1. Extract PERSON entities only with descending order by score.
        2. Extract LOCATION entities only with descending order by score
        3. Extract DATE entities only with descending order by score
    """
    person1: str = "Abraham Lincoln"
    person2: str = "Ulysses S. Grant"
    person3: str = "Robert E. Lee"
    location: str = "Gettysburg"
    date1: str = "12 Apr 1861"
    date2: str = "9 Apr 1865"

    text: str = f"""During the civil war from {date1} – {date2},
                {person1} was the president and leading The Union. 
                {person2} was some of his generals. 
                {person3} fought the last fight at {location}."""

    # --------------------------------------------------------------------------------
    # All entities
    # --------------------------------------------------------------------------------
    entities: List[Dict[str, Any]] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=None
    )
    listing = json.dumps(entities, indent=4, default=str, ensure_ascii=False)
    # print(f"total entities: {listing}")

    # --------------------------------------------------------------------------------
    # entity_type=PERSON
    # [
    #     {
    #         "Score": 0.9997009038925171,
    #         "Type": "PERSON",
    #         "Text": "Abraham Lincoln",
    #         "BeginOffset": 52,
    #         "EndOffset": 67
    #     },
    #     {
    #         "Score": 0.9993404746055603,
    #         "Type": "PERSON",
    #         "Text": "Robert E. Lee",
    #         "BeginOffset": 152,
    #         "EndOffset": 165
    #     },
    #     {
    #         "Score": 0.9573029279708862,
    #         "Type": "PERSON",
    #         "Text": "Ulysses S. Grant",
    #         "BeginOffset": 109,
    #         "EndOffset": 125
    #     }
    # ]
    # --------------------------------------------------------------------------------
    people: List[Dict[str, Any]] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["PERSON"],
        sort_by_score=True
    )
    people_listing = json.dumps(people, indent=4, default=str, ensure_ascii=False)
    # print(f"people_listing: {people_listing}")
    assert len(people) == 3, f"expected 3 people entities, got [{len(people)}]."
    assert people[0]['Score'] >= people[1]['Score'] >= people[2]['Score'], \
        f"expected descending sorted by score, got {people_listing}"
    assert {people[0]['Text'], people[1]['Text'], people[2]['Text']} == {person1, person2, person3}, \
        f"expected people {{person1, person2, person3}} in {people_listing}"

    # --------------------------------------------------------------------------------
    # entity_type=LOCATION
    # [
    #     {
    #         "Score": 0.9948418736457825,
    #         "Type": "LOCATION",
    #         "Text": "Gettysburg",
    #         "BeginOffset": 191,
    #         "EndOffset": 201
    #     }
    # ]
    # --------------------------------------------------------------------------------
    locations: List[Dict[str, Any]] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["LOCATION"],
        sort_by_score=True
    )
    location_listing = json.dumps(locations, indent=4, default=str, ensure_ascii=False)
    # print(f"location_listing: {location_listing}")
    assert len(locations) == 1, f"expected 1 locations entity, got [{len(locations)}]."
    assert locations[0]['Text'] == location, \
        f"expected locations {location}, got {locations[0]['Text']}.\n" \
        f"Entities found:{location_listing}"

    # --------------------------------------------------------------------------------
    # entity_type=DATE
    # [
    #     {
    #         "Score": 0.9979767799377441,
    #         "Type": "DATE",
    #         "Text": "9 Apr 1865",
    #         "BeginOffset": 40,
    #         "EndOffset": 50
    #     },
    #     {
    #         "Score": 0.9972290396690369,
    #         "Type": "DATE",
    #         "Text": "12 Apr 1861",
    #         "BeginOffset": 26,
    #         "EndOffset": 37
    #     }
    # ]
    # --------------------------------------------------------------------------------
    dates: List[Dict[str, Any]] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["DATE"],
        sort_by_score=True
    )
    dates_listing = json.dumps(dates, indent=4, default=str, ensure_ascii=False)
    # print(f"dates_listing: {dates_listing}")
    assert len(dates) == 2, f"expected 2 dates entities, got [{len(dates)}]."
    assert dates[0]['Score'] >= dates[1]['Score'], \
        f"expected descending sorted by score, got {dates_listing}"
    assert {dates[0]['Text'], dates[1]['Text']} == {date1, date2}, \
        f"expected dates {{date1, date2}} in {dates_listing}"


def test_detect_entities_by_type__return_entity_value_only():
    """Test entity detections of a certain entity type e.g. PERSON, LOCATION, DATE
    in English text with only values returned e.g. ["Abraham Lincoln", "Robert E. Lee"]
    instead of list of dictionaries.

    Test conditions:
        From the text including PERSON, LOCATION, DATE,
        1. Extract PERSON values only with descending order by score.
        2. Extract LOCATION values only with descending order by score
        3. Extract DATE values only with descending order by score
    """
    person1: str = "Abraham Lincoln"
    person2: str = "Ulysses S. Grant"
    person3: str = "Robert E. Lee"
    location: str = "Gettysburg"
    date1: str = "12 Apr 1861"
    date2: str = "9 Apr 1865"

    text: str = f"""During the civil war from {date1} – {date2},
                {person1} was the president and leading The Union. 
                {person2} was some of his generals. 
                {person3} fought the last fight at {location}."""

    # --------------------------------------------------------------------------------
    # entity_type=PERSON
    # [
    #     {
    #         "Score": 0.9997009038925171,
    #         "Type": "PERSON",
    #         "Text": "Abraham Lincoln",
    #         "BeginOffset": 52,
    #         "EndOffset": 67
    #     },
    #     {
    #         "Score": 0.9993404746055603,
    #         "Type": "PERSON",
    #         "Text": "Robert E. Lee",
    #         "BeginOffset": 152,
    #         "EndOffset": 165
    #     },
    #     {
    #         "Score": 0.9573029279708862,
    #         "Type": "PERSON",
    #         "Text": "Ulysses S. Grant",
    #         "BeginOffset": 109,
    #         "EndOffset": 125
    #     }
    # ]
    # --------------------------------------------------------------------------------
    # First, get the list of dictionaries
    people: List[Dict[str, Any]] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["PERSON"],
        sort_by_score=True,
        return_entity_value_only=False
    )
    people_listing = json.dumps(people, indent=4, default=str, ensure_ascii=False)
    # print(f"people_listing: {people_listing}")
    assert len(people) == 3, f"expected 3 people entities, got [{len(people)}]."

    people_values_expected: List[str] = [
        people[0]['Text'], people[1]['Text'], people[2]['Text']
    ]

    # Then, get the values only.
    people_values_actual: List[str] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["PERSON"],
        sort_by_score=True,
        return_entity_value_only=True
    )
    # print(f"people_values_actual: {people_values_actual}")
    assert len(people_values_expected) == len(people_values_actual), \
        f"expected {len(people_values_expected)} people, got {len(people_values_actual)}\n " \
        f"as people_values_actual={people_values_actual} collected from {people_listing}"

    assert all([
        people_values_actual[index] == expected
        for index, expected in enumerate(people_values_expected)
    ]), f"expected {people_values_expected}, got {people_values_actual}\n" \
        f"collected from {people_listing}"

    # --------------------------------------------------------------------------------
    # entity_type=LOCATION
    # [
    #     {
    #         "Score": 0.9948418736457825,
    #         "Type": "LOCATION",
    #         "Text": "Gettysburg",
    #         "BeginOffset": 191,
    #         "EndOffset": 201
    #     }
    # ]
    # --------------------------------------------------------------------------------
    locations: List[Dict[str, Any]] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["LOCATION"],
        sort_by_score=True,
        return_entity_value_only=False
    )
    location_listing = json.dumps(locations, indent=4, default=str, ensure_ascii=False)
    # print(f"location_listing: {location_listing}")
    assert len(locations) == 1, f"expected 1 locations entity, got [{len(locations)}]."
    assert locations[0]['Text'] == location, \
        f"expected locations {location}, got {locations[0]['Text']}.\n" \
        f"Entities found:{location_listing}"

    location_values_expected: List[str] = [location]
    location_values_actual: List[str] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["LOCATION"],
        sort_by_score=True,
        return_entity_value_only=True
    )
    # print(f"location_values_actual: {location_values_actual}")
    assert location_values_expected == location_values_actual, \
        f"expected {location_values_expected} entities, got {location_values_actual}\n" \
        f"collected from {location_listing}"

    # --------------------------------------------------------------------------------
    # entity_type=DATE
    # [
    #     {
    #         "Score": 0.9979767799377441,
    #         "Type": "DATE",
    #         "Text": "9 Apr 1865",
    #         "BeginOffset": 40,
    #         "EndOffset": 50
    #     },
    #     {
    #         "Score": 0.9972290396690369,
    #         "Type": "DATE",
    #         "Text": "12 Apr 1861",
    #         "BeginOffset": 26,
    #         "EndOffset": 37
    #     }
    # ]
    # --------------------------------------------------------------------------------
    dates: List[Dict[str, Any]] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["DATE"],
        sort_by_score=True,
        return_entity_value_only=False
    )
    dates_listing = json.dumps(dates, indent=4, default=str, ensure_ascii=False)
    # print(f"dates_listing: {dates_listing}")
    assert len(dates) == 2, f"expected 2 dates entities, got [{len(dates)}]."
    assert dates[0]['Score'] >= dates[1]['Score'], \
        f"expected descending sorted by score, got {dates_listing}"

    date_values_expected: List[str] = [dates[0]['Text'], dates[1]['Text']]
    date_values_actual: List[str] = comprehend.detect_entities_by_type(
        text=text,
        language_code="en",
        entity_types=["DATE"],
        sort_by_score=True,
        return_entity_value_only=True
    )
    # print(f"date_values_actual: {date_values_actual}")
    assert len(date_values_expected) == len(date_values_actual), \
        f"expected {len(date_values_expected)} date, got {len(date_values_actual)}\n " \
        f"as date_values_actual={date_values_actual} collected from {dates_listing}"

    # Verify the values are descending order by score as in
    assert all([
        date_values_actual[index] == expected
        for index, expected in enumerate(date_values_expected)
    ]), f"expected {date_values_expected}, got {date_values_actual}\n" \
        f"collected from {dates_listing}"

