from typing import (
    TypeVar,
)
import string
import random


T = TypeVar('T')


def random_string(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
