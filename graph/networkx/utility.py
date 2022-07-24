import os
import sys
import string
from itertools import (
    product
)
from typing import (
    List,
    Tuple,
    Any,
    Optional,
)
import contextlib

CSV_FIELD_DELIMITER: str = ","
OS_EOL_CHARACTER = os.linesep


@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'w', encoding='utf-8')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def generate_alphabet_combinations(length: int = 2) -> List[str]:
    """Generate cartesian product of alphabet characters of up to the length
    If n is 2, then ['a', 'b', 'c', ..., 'aa', ..., 'zz']
    Args:
        length: length of the string generated
    Returns: Combination of alphabet characters up to length.
    """
    assert length > 0
    alphabets = string.ascii_lowercase

    return [
        ''.join(combination)
        for n in range(1, length+1)
        for combination in product(alphabets, repeat=n)
    ]


def is_number(entity: Any) -> Tuple[bool, Optional[float]]:
    """Check if the input is a number
    Args:
        entity: input variable to test
    Returns: (result, value) where value is float if the input is the number or None.
    """
    try:
        number: float = float(entity)
        return True, number
    except ValueError:
        return False, None


def test_is_number():
    assert is_number("-.1")[0]
    assert not is_number("hoge")[0]
