"""Module to test the Board class"""
import random
import sys
from typing import (
    List,
)

from implementation import (
    Board
)


# Number of random test executions
MAX_RANDOM_TRIAL: int = 100


def test_board_initialization_fail():
    """
    Test Case ID: B.CASE.011
    Verify creating a board of a shape (w,h) fails when the w or h is invalid.
    """
    valid_numbers: List[int] = [1, random.randint(2, sys.maxsize), sys.maxsize]
    invalid_numbers: List[int] = [num - sys.maxsize for num in valid_numbers]

    for _ in range(MAX_RANDOM_TRIAL):
        # 50,50 chance that width or height is invalid
        if random.uniform(0, 1) > 0.5:
            # B.CND.011 (invalid width)
            width: int = random.choice(valid_numbers)
            height: int = random.choice(invalid_numbers)
        else:
            # B.CND.012 (invalid height)
            width: int = random.choice(invalid_numbers)
            height: int = random.choice(valid_numbers)

        try:
            Board(width=width, height=height)
            assert False, \
                f"expected ValueError for non positive width:[{width}] or/and height:[{height}]."
        except ValueError:
            pass
