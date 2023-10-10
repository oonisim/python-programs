"""Module to test util_nlp.test
"""
from typing import (
    List
)

from lib.util_nlp.text import (
    redact_phone_numbers,
    TAG_AUSTRALIAN_PHONE_NUMBER,
    TAG_AUSTRALIAN_BUSINESS_NUMBER,
)


def test_redact_phone_numbers__australia_success():
    """
    Verify the phone number redaction.
    expected: Australian phone number redacted.
    """
    for number in [
        '+(61) 455 562 400',
        '+(61)455562400',
        '+61-455-562-400',
        '+61 455 562 400',
        '+(61)-455-562-400',
        '+(61)-455-562400',
        '+(61) 455 562 400',
        '(02) 4371 3164',
        '(02) 4371-3164',
        '02 80268989',
        '03 80268989',
        '04 80268989',
        '05 80268989',
        '0433245898',
        '0433 245 898',
        '433 245 898',
        '433-245-898',
        '0433-245-898',
        '08 54 587 456',
        '0854587456',
        '+61854587456'
    ]:
        assert redact_phone_numbers(
            text=number,
            country="AU"
        ) == TAG_AUSTRALIAN_PHONE_NUMBER, \
            f"expected phone number redaction for [{number}]."


def test_redact_phone_numbers__australia_fail():
    """
    Verify the phone number redaction.
    expected: Non-Australian phone number reduction fails
    """
    for number in [
        # '+(610) 455 562 400',  # <--- cannot handle this one yet.
        '+(61)1455562400',
        '+61-455-62-400',
        '() 4371 3164',
        '020268989',
        '030268989',
        '040268989',
        '05 68989',
        '0433245x898',
        '0433 245 89',
        '18 54 587 456',
        '9854587456',
        '+6185458745'
    ]:
        assert redact_phone_numbers(
            text=number,
            country="AU"
        ) == number, \
            f"expected phone number not redacted for [{number}]."
