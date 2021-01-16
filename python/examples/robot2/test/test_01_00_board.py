from typing import (
    Final
)
import random
import pytest
from board import Board

MAX_BOARD_SIZE: Final[int] = 999999999
MAX_TEST_TIMES: Final[int] = 100


def test_board_setup():
    """Test (x, y) is inside the board(n, m)"""
    negative = random.randint(-MAX_BOARD_SIZE, -1)
    positive = random.randint(0, MAX_BOARD_SIZE

    # --------------------------------------------------------------------------------
    # Board creation for size (m, n) where m < 0 or n < 0
    # --------------------------------------------------------------------------------
    for i in range(MAX_TEST_TIMES):
        try:
            board: Board = Board(n=negative, m=negative)
            assert False, "Board creation needs to fail for (x, y)=({},{})".format(negative, negative)
        except Exception as e:
            pass

        try:
            board: Board = Board(n=negative, m=0)
            assert False, "Board creation needs to fail for (x, y)=({},{})".format(negative, 0)
        except Exception as e:
            pass

        try:
            board: Board = Board(n=0, m=-1)
            assert False, "Board creation needs to fail for (x, y)=({},{})".format(0, negative)
        except Exception as e:
            pass

        try:
            board: Board = Board(n=0, m=-1)
            assert False, "Board creation needs to fail for (x, y)=({},{})".format(0, 0)
        except Exception as e:
            pass

        try:
            board: Board = Board(n=positive, m=0)
            assert False, "Board creation needs to fail for (x, y)=({},{})".format(positive, 0)
        except Exception as e:
            pass

        try:
            board: Board = Board(n=positive, m=0)
            assert False, "Board creation needs to fail for (x, y)=({},{})".format(0, positive)
        except Exception as e:
            pass

        # --------------------------------------------------------------------------------
        # Location within the board boundary (0, 0) <= (x, y) < (n, m) for location (x,y)
        # --------------------------------------------------------------------------------
        n = random.randint(0, MAX_BOARD_SIZE)
        m = random.randint(0, MAX_BOARD_SIZE)

        board = Board(n, m)

        location = [random.randint(-MAX_BOARD_SIZE, -1), 0]
        assert board.is_inside(location) is False, "is_inside({}) needs to be False".format(location)

        location = [0, random.randint(-MAX_BOARD_SIZE, -1)]
        assert board.is_inside(location) is False, "is_inside({}) needs to be False".format(location)

        location = [random.randint(n, MAX_BOARD_SIZE+1), 0]
        assert board.is_inside(location) is False, "is_inside({}) needs to be False".format(location)

        location = [0, random.randint(m, MAX_BOARD_SIZE+1)]
        assert board.is_inside(location) is False, "is_inside({}) needs to be False".format(location)

        location = [random.randint(n, MAX_BOARD_SIZE+1), random.randint(m, MAX_BOARD_SIZE+1)]
        assert board.is_inside(location) is False, "is_inside({}) needs to be False".format(location)

        location = [0, random.randint(0, m-1)]
        assert board.is_inside(location) is True, "is_inside({}) needs to be False".format(location)

        location = [random.randint(0, n-1), m]
        assert board.is_inside(location) is True, "is_inside({}) needs to be False".format(location)

        location = [random.randint(0, n-1), random.randint(0, m-1)]
        assert board.is_inside(location) is True, "is_inside({}) needs to be False".format(location)
