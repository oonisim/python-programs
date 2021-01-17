"""Pytest Test case for the Robot Creation"""
from typing import (
    List,
    Dict,
    Tuple
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
from . test_00_config import *


# ====================================================================================================
# TC 001-002
# ====================================================================================================
def tc_001(n: int, m: int, board: Board, robot: Robot, state: State):
    """TC 001: move to the NORTH boundary
    """
    command: Command = Command(action=MOVE, state=state)

    # Move (m-1) times
    reply: State = None

    for y in range(0 + 1, (m - 1) + 1):
        reply = robot.execute(command=command)
        assert reply['location'][1] == min(y, m-1), \
            f"robot y location expected {y} actual {reply['location'][1]}"

    # --------------------------------------------------------------------------------
    # If m=1, no move.
    # --------------------------------------------------------------------------------
    if m > 1:
        assert reply, f"robot.execute() returned None m {m} n {n} command {command}"
        assert np.all(reply['location'] == [0, m - 1]), \
            f"robot y location expected [0, {m - 1}] actual {reply['location']}"
        assert reply['direction'] == NORTH, \
            f"robot y location expected {NORTH} actual {reply['direction']}"


def tc_002(n: int, m: int, board: Board, robot: Robot, state: State):
    """TC 002: Push the robot further NORTH
    """
    command: Command = Command(action=MOVE, state=state)
    reply: State = None
    for _ in range(random.randint(1, 100)):
        reply = robot.execute(command=command)

    assert reply and np.all(reply['location'] == [0, m - 1]), \
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
            # TC 001: move to the NORTH boundary
            tc_001(n, m, board, robot, state)
            # TC 002 Push the robot further NORTH
            tc_002(n, m, board, robot, state)
        except Exception:
            assert False, f"Robot operations TC 001-002: move NORTH from initial need to succeed."


# ====================================================================================================
# TC 003-005
# ====================================================================================================
def tc_003(n: int, m: int, board: Board, robot: Robot, state: State):
    """TC 003: Turn RIGHT to EAST
    """
    command: Command = Command(action=RIGHT, state=state)
    reply = robot.execute(command=command)
    assert reply and np.all(reply['location'] == [0, m - 1]), \
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

    # --------------------------------------------------------------------------------
    # If n=1, no move.
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # If m=1, no move.
    # --------------------------------------------------------------------------------
    if n > 1:
        assert reply, f"robot.execute() returned None m {m} n {n} command {command}"
        assert np.all(reply['location'] == [n-1, m - 1]), \
            f"robot x location expected [{n-1}, {m - 1}] actual {reply['location']}"
        assert reply['direction'] == EAST, \
            f"robot x location expected {EAST} actual {reply['direction']}"


def tc_005(n: int, m: int, board: Board, robot: Robot, state: State):
    """TC 005: Push the robot further EAST
    """
    command: Command = Command(action=MOVE, state=state)
    reply: State = None
    for _ in range(random.randint(1, 100)):
        reply = robot.execute(command=command)

    assert reply and np.all(reply['location'] == [n-1, m - 1]), \
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

            # TC 003: Turn RIGHT to EAST
            tc_003(n, m, board, robot, state)
            # TC 004: Move to the EAST boundary
            tc_004(n, m, board, robot, state)
            # TC 005: Push the robot further EAST
            tc_004(n, m, board, robot, state)
        except Exception:
            assert False, "Robot operations TC 003-005: move EAST need to succeed."
