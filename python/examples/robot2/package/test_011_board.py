"""Pytest Test case for the robot placement check on the board"""
import random
from . area import Board
from . test_00_config import *


def test_board_robot_placement():
    """Test case for board boundaries to check if a robot can be placed at a location.
    """
    # Location within the board boundary (0, 0) <= (x, y) < (n, m) for location (x,y)
    for _ in range(MAX_TEST_TIMES):
        negative = random.randint(-MAX_BOARD_SIZE, -1)
        positive = random.randint(1, MAX_BOARD_SIZE)
        n = random.randint(1, MAX_BOARD_SIZE)
        m = random.randint(1, MAX_BOARD_SIZE)

        board = Board(n, m)

        # TC 005: False for x < 0
        location = [random.randint(-MAX_BOARD_SIZE, -1), 0]
        assert not board.contains(location), f"contains({location}) needs to be False"

        # TC 006 False for y < 0
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
        assert board.contains(location), f"contains({location}) needs to be True"
