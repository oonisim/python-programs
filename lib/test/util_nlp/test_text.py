"""Module to test util_nlp.test
"""
from typing import (
    List
)

import regex as re

from lib.util_nlp.text import (
    normalize_typographical_unicode_characters,
    restore_contracted,
    redact_abn,
    redact_phone_numbers,
    normalize,
    TAG_AUSTRALIAN_PHONE_NUMBER,
    TAG_AUSTRALIAN_BUSINESS_NUMBER,
)


# --------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------
def test_normalize_typographical_unicode_characters():
    """Verify typographical non-ascii unicode e.g. EM-dash
    Test conditions:
    1. EM-dash is decoded into two hyphen-minus
    2. Right single quote is decoded into single quote.
    """
    # Test condition #1
    normalized: str = normalize_typographical_unicode_characters('\u2014')
    assert normalized == '-', \
        f"expected EM-dash (\u2014) normalized as '-' got {normalized}"

    # Test condition #2
    normalized: str = normalize_typographical_unicode_characters('\u2019')
    assert normalized == "'", \
        f"expected right single quotation (\u2019) normalized as ''' got {normalized}"

    normalized: str = normalize_typographical_unicode_characters("ÔÇÿ\u2014Ö\u2019")
    assert normalized == "ÔÇÿ-Ö'", \
        f"expected (ÔÇÿ\u2014Ö\u2019) normalized as ÔÇÿ-Ö' got {normalized}"


def test_restore_contracted():
    """ Verify the de-contraction.
    Expected: contraction e.g. won't -> will not
    """
    assert restore_contracted("I'm not into it.") == "I am not into it."
    assert restore_contracted("we won't make it.") == "we will not make it."
    assert restore_contracted("they're well prepared.") == "they are well prepared."
    assert restore_contracted("Here's the coffee.") == "Here is the coffee."


def test_redact_abn__success():
    """Verify the valid ABN redaction.
    Test conditions:
    Valid ABN is redacted (see https://abr.business.gov.au/Help/AbnFormat).
    1. ABN: 861-4924-1458
    2. ABN : 861-4924-1458
    3. ABN #861-4924-1458
    4. ABN 861-4924-1458
    5. ABN 86 149 241 458
    6. ABN 86149 241 458
    7. ABN 86149 241 458.
    8. ABN 86149 241 458,
    """
    for abn in [
        "ABN: 861-4924-1458",
        "ABN : 861-4924-1458",
        "ABN #861-4924-1458",
        "ABN 861-4924-1458",
        "ABN 86 149 241 458",
        "ABN 86149 241 458",
    ]:
        assert redact_abn(abn) == TAG_AUSTRALIAN_BUSINESS_NUMBER, \
            f"expected ABN redaction for [{abn}], got [{redact_abn(abn)}]."

    abn: str = "ABN 86149 241 458."
    expected: str = f"{TAG_AUSTRALIAN_BUSINESS_NUMBER}."
    assert redact_abn(abn) == expected, \
        f"expected ABN redaction {expected} for [{abn}], got [{redact_abn(abn)}]."

    abn: str = "(abN 86149 241 458),"
    expected: str = f"({TAG_AUSTRALIAN_BUSINESS_NUMBER}),"
    assert redact_abn(abn) == expected, \
        f"expected ABN redaction {expected} for [{abn}], got [{redact_abn(abn)}]."

    abn: str = "St. Vincent Hospital (abN 86149 241 458), is a private hospital in NSW."
    expected: str = f"St. Vincent Hospital ({TAG_AUSTRALIAN_BUSINESS_NUMBER}), "\
                    "is a private hospital in NSW."
    assert redact_abn(abn) == expected, \
        f"expected ABN redaction {expected} for [{abn}], got [{redact_abn(abn)}]."


def test_redact_abn__fail():
    """Verify the valid ABN redaction.
    Test conditions:
    Invalid ABN not following the rule is not redacted.
    (see https://abr.business.gov.au/Help/AbnFormat)
    1. ABN 16149 241 458 is returned as is.
    2. ABN 241 458 is returned as is.
    3. BBN 86149 241 458 is returned as is.
    """
    for abn in [
        "ABN 16149 241 458",
        "ABN 241 458",
        "BBN 86149 241 458"
    ]:
        redacted: str = redact_abn(abn)
        assert redacted == abn, \
            f"expected no redaction for [{abn}], got [{redacted}]."

    abn: str = "St. Vincent Hospital ,abn 6149 241 458, is a private hospital in NSW."
    expected: str = f"St. Vincent Hospital ({TAG_AUSTRALIAN_BUSINESS_NUMBER}), "\
                    "is a private hospital in NSW."
    assert redact_abn(abn) == abn, \
        f"expected no redaction for [{abn}], got [{redacted}]."


def test_redact_phone_numbers__australia_success():
    """
    Verify the phone number redaction.
    expected: Australian phone number redacted.
    """
    for number in [
        '+(61) 455 562 400',
        '+61-02-8088-3433',
        '+(61)455562400',
        '+61-455-562-400',
        '+61 455 562 400',
        '+(61)-455-562-400',
        '+(61)-455-562400',
        '+(61) 455 562 400',
        '(02) 4371 3164',
        '(02) 4371-3164',
        '0416117205',
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

    number = '0416117205.'
    expected = f"{TAG_AUSTRALIAN_PHONE_NUMBER}."
    assert redact_phone_numbers(
        text=number,
        country="AU"
    ) == expected, \
        f"expected phone number redaction [{expected}] for [{number}]."

    number = '#0416117205'
    expected = f"#{TAG_AUSTRALIAN_PHONE_NUMBER}"
    assert redact_phone_numbers(
        text=number,
        country="AU"
    ) == expected, \
        f"expected phone number redaction [{expected}] for [{number}]."

    number = '#0416117205.'
    expected = f"#{TAG_AUSTRALIAN_PHONE_NUMBER}."
    assert redact_phone_numbers(
        text=number,
        country="AU"
    ) == expected, \
        f"expected phone number redaction [{expected}] for [{number}]."

    number = '@0416117205'
    expected = f"@{TAG_AUSTRALIAN_PHONE_NUMBER}"
    assert redact_phone_numbers(
        text=number,
        country="AU"
    ) == expected, \
        f"expected phone number redaction [{expected}] for [{number}]."

    number = '@0416117205.'
    expected = f"@{TAG_AUSTRALIAN_PHONE_NUMBER}."
    assert redact_phone_numbers(
        text=number,
        country="AU"
    ) == expected, \
        f"expected phone number redaction [{expected}] for [{number}]."

    number = 'at 0416117205'
    expected = f"at{TAG_AUSTRALIAN_PHONE_NUMBER}"
    assert redact_phone_numbers(
        text=number,
        country="AU"
    ) == expected, \
        f"expected phone number redaction [{expected}] for [{number}]."

    number = 'on 0416117205'
    expected = f"on{TAG_AUSTRALIAN_PHONE_NUMBER}"
    assert redact_phone_numbers(
        text=number,
        country="AU"
    ) == expected, \
        f"expected phone number redaction [{expected}] for [{number}]."

    number = 'on 0416117205.'
    expected = f"on{TAG_AUSTRALIAN_PHONE_NUMBER}."
    assert redact_phone_numbers(
        text=number,
        country="AU"
    ) == expected, \
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


def test_normalize():
    """Verify text normalization
    """
    text: str = """
    Dear Dr... Ian FlemingÔÇÿ.
    Re: ÔÇÿpatient James Bond:
    After the injury at the Sky Never Fall, he made a complete recovery 
    ((To my surprise 😉... TBCH 🤣😁😄)).
    Email ian.fleming@mi6.gov.uk
    Phone: +61 (02)\u20148088\u20143433
    URL: https://www.sis.gov.uk/mi6
    """

    expected: str = """
    Dear Dr. Ian Fleming. Re: patient James Bond: 
    After the injury at the Sky Never Fall, he made a complete recovery
    (To my surprise . TBCH ).
    Email TAG_EMAIL Phone: 
    TAG_PHONE_NUMBER 
    URL: TAG_URL
    """

    expected = re.sub(pattern=r'\s+', repl=' ', string=expected).strip()
    actual: str = normalize(text)
    assert actual == expected, \
        f"expected [{text}]\nnormalized as \n[{expected}],\ngot[{actual}]."
