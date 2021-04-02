from typing import (
    TypeVar,
    Generator
)
import string
import random


T = TypeVar('T')


def random_string(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def generator(func):
    def start(*args, **kwargs):
        g = func(*args, **kwargs)
        # next(g) is the same but be clear intention of advancing the execution to the yield line.
        g.send(None)
        return g

    return start
