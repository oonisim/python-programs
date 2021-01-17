from typing import (
    Final
)
import random
from . area import Board

MAX_BOARD_SIZE: Final[int] = 999999999
MAX_TEST_TIMES: Final[int] = 100


def test_board_setup():
    """Test (x, y) is inside the board(n, m)"""
    negative = random.randint(-MAX_BOARD_SIZE, -1)
    positive = random.randint(0, MAX_BOARD_SIZE)

    # --------------------------------------------------------------------------------
    # Board creation for size (m, n) where m < 0 or n < 0
    # --------------------------------------------------------------------------------
    for _ in range(MAX_TEST_TIMES):
        try:
            board: Board = Board(n=negative, m=negative)
            print()
            assert False, f"Board creation needs to fail for (x, y)=({negative},{negative})"
        except Exception:
            pass

        try:
            board: Board = Board(n=negative, m=0)
            assert False, f"Board creation needs to fail for (x, y)=({negative},0)"
        except Exception:
            pass

        try:
            board: Board = Board(n=0, m=-1)
            assert False, f"Board creation needs to fail for (x, y)=(0,{negative})"
        except Exception:
            pass

        try:
            board: Board = Board(n=0, m=-1)
            assert False, "Board creation needs to fail for (x, y)=(0,0)"
        except Exception:
            pass

        try:
            board: Board = Board(n=positive, m=0)
            assert False, f"Board creation needs to fail for (x, y)=({positive},0)"
        except Exception:
            pass

        try:
            board: Board = Board(n=positive, m=0)
            assert False, f"Board creation needs to fail for (x, y)=({positive},0)"
        except Exception:
            pass

        # --------------------------------------------------------------------------------
        # Location within the board boundary (0, 0) <= (x, y) < (n, m) for location (x,y)
        # --------------------------------------------------------------------------------
        n = random.randint(0, MAX_BOARD_SIZE)
        m = random.randint(0, MAX_BOARD_SIZE)

        board = Board(n, m)

        # False for x < 0
        location = [random.randint(-MAX_BOARD_SIZE, -1), 0]
        assert not board.contains(location), f"contains({location}) needs to be False"

        # False for y < 0
        location = [0, random.randint(-MAX_BOARD_SIZE, -1)]
        assert not board.contains(location), f"contains({location}) needs to be False"

        # False for x > n-1
        location = [random.randint(n, MAX_BOARD_SIZE+1), 0]
        assert not board.contains(location), f"contains({location}) needs to be False"

        # False for y > m-1
        location = [0, random.randint(m, MAX_BOARD_SIZE+1)]
        assert not board.contains(location), f"contains({location}) needs to be False"

        # False for x > n-1 and y > m-1
        location = [random.randint(n, MAX_BOARD_SIZE+1), random.randint(m, MAX_BOARD_SIZE+1)]
        assert not board.contains(location), f"contains({location}) needs to be False"

        # True for x = 0 and 0 <= y <= m-1
        location = [0, random.randint(0, m-1)]
        assert board.contains(location), f"contains({location}) needs to be True"

        # True for 0 <= x <= n-1 and y = 0
        location = [random.randint(0, n-1), 0]
        assert board.contains(location), f"contains({location}) needs to be True"

        # True for 0 <= x <= n-1 and 0 <= y <= m-1
        location = [random.randint(0, n-1), random.randint(0, m-1)]
        assert board.contains(location), f"contains({location}) needs to be False"
