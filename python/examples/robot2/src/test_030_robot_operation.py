"""Pytest Test case for the Robot Creation"""
from typing import(
    List,
    Dict,
    Tuple
)
import random
import string
import logging
from . area import Board
from . test_10_board_config import (
    MAX_BOARD_SIZE,
    MAX_TEST_TIMES
)
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
from . robot import (
    Robot,
    State,
    Command
)


def random_string() -> str:
    n: int = 6
    ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def create_robot() -> (Robot, State):
    n = random.randint(1, MAX_BOARD_SIZE)
    m = random.randint(1, MAX_BOARD_SIZE)

    board: Board = Board(n=n, m=m)
    location: List[int] = [0, 0]
    d: str = NORTH
    state: State = State(location=location, direction=d)
    robot: Robot = Robot(board=board, state=state, log_level=logging.DEBUG)

    return n, m, board, robot, state


# ====================================================================================================
# TC 001-002
# ====================================================================================================
def tc_001(n:int, m:int, board: Board, robot: Robot, state: State):
    """TC 001: move to the north boundary
    """
    command: Command = Command(action=MOVE, state=state)

    # Move (m-1) times
    reply: State = None
    for y in range(0 + 1, (m - 1) + 1):
        reply = robot.execute(command=command)
        assert reply['location'][1] == y, \
            f"robot y location expected {y} actual {reply['location'][1]}"

    assert reply and reply['location'] == [0, m - 1], \
        f"robot y location expected [0, {m - 1}] actual {reply['location']}"
    assert reply and reply['direction'] == NORTH, \
        f"robot y location expected {NORTH} actual {reply['direction']}"


def tc_002(n:int, m:int, board: Board, robot: Robot, state: State):
    """TC 002: Push the robot further NORTH
    """
    command: Command = Command(action=MOVE, state=state)
    reply: State = None
    for _ in range(random.randint(1, 100)):
        reply = robot.execute(command=command)

    assert reply and reply['location'] == [0, m - 1], \
        f"robot y location expected [0, {m - 1}] actual {reply['location']}"
    assert reply and reply['direction'] == NORTH, \
        f"robot y location expected {NORTH} actual {reply['direction']}"


def test_operator_operate_robot_001_002():
    """Test robot operations
    Test conditions: 001-002
    """
    for _ in range(MAX_TEST_TIMES):
        n, m, board, robot, state = create_robot()

        try:
            # TC 001: move to the north boundary
            tc_001(n, m, board, robot, state)
            # TC 002 Push the robot further NORTH
            tc_002(n, m, board, robot, state)
        except Exception:
            assert False, f"Robot operations TC 001-002: move NORTH from initial need to succeed."


# ====================================================================================================
# TC 003-005
# ====================================================================================================
def tc_003(n: int, m: int, board: Board, robot: Robot, state: State):
    """TC 003: Turn left to EAST
    """
    command: Command = Command(action=LEFT, state=state)
    reply = robot.execute(command=command)
    assert reply and reply['location'] == [0, m - 1], \
        f"robot y location expected [0, {m - 1}] actual {reply['location']}"
    assert reply and reply['direction'] == EAST, \
        f"robot y location expected {EAST} actual {reply['direction']}"


def tc_004(n: int, m: int, board: Board, robot: Robot, state: State):
    """TC 004: move to the EAST boundary
    """
    command: Command = Command(action=MOVE, state=state)

    # Move (m-1) times
    reply: State = None
    for x in range(0 + 1, (n - 1) + 1):
        reply = robot.execute(command=command)
        assert reply['location'][1] == x, \
            f"robot x location expected {x} actual {reply['location'][1]}"

    assert reply and reply['location'] == [n-1, m - 1], \
        f"robot x location expected [{n-1}, {m - 1}] actual {reply['location']}"
    assert reply and reply['direction'] == EAST, \
        f"robot x location expected {EAST} actual {reply['direction']}"


def tc_005(n: int, m: int, board: Board, robot: Robot, state: State):
    """TC 005: Push the robot further EAST
    """
    command: Command = Command(action=MOVE, state=state)
    reply: State = None
    for _ in range(random.randint(1, 100)):
        reply = robot.execute(command=command)

    assert reply and reply['location'] == [n-1, m - 1], \
        f"robot x location expected [{n-1}, {m - 1}] actual {reply['location']}"
    assert reply and reply['direction'] == EAST, \
        f"robot x location expected {EAST} actual {reply['direction']}"


def test_operator_operate_robot_003_004():
    """Test robot operations
    """
    for _ in range(MAX_TEST_TIMES):
        n, m, board, robot, state = create_robot()

        try:
            tc_001(n, m, board, robot, state)
            tc_002(n, m, board, robot, state)

            # TC 003: Turn left to EAST
            tc_003(n, m, board, robot, state)
            # TC 004: Move to the EAST boundary
            tc_004(n, m, board, robot, state)
            # TC 005: Push the robot further EAST
            tc_004(n, m, board, robot, state)
        except Exception:
            assert False, f"Robot operations TC 003-005: move EAST need to succeed."
