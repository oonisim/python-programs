from typing import (
    Final
)
import logging
import random
import string
MAX_BOARD_SIZE: Final[int] = 999
MAX_TEST_TIMES: Final[int] = 100


def random_string(n: int) -> str:
    """Generate random string of length N"""
    s = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))
    return s
