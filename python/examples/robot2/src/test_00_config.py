from typing import (
    List,
    Dict,
    Tuple,
    Final
)
import random
import string
import logging
import numpy as np
from . constant import (
    NORTH,
    SOUTH,
    EAST,
    WEST,
    REPORT,
    MOVE,
    LEFT,
    RIGHT,
    PLACE
)
from . area import Board
from . robot import (
    Robot,
    State,
    Command
)
from . operator import Operator

MAX_BOARD_SIZE: Final[int] = 999
MAX_TEST_TIMES: Final[int] = 100


def random_string(n: int) -> str:
    """Generate random string of length N"""
    s = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))
    return s


def create_robot() -> (Robot, State):
    n = random.randint(1, MAX_BOARD_SIZE)
    m = random.randint(1, MAX_BOARD_SIZE)

    board: Board = Board(n=n, m=m)
    location: List[int] = [0, 0]
    d: str = NORTH
    state: State = State(location=location, direction=d)
    robot: Robot = Robot(board=board, state=state)

    return n, m, board, robot, state


def create_operator(path: str, log_level=logging.CRITICAL) -> Operator:
    """Create an operator instance"""
    n, m, board, robot, state = create_robot()
    operator: Operator = Operator(board=board, path=path, log_level=log_level)
    return operator
