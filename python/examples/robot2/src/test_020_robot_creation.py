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
from . test_10_board_config import *
from . constant import (
    NORTH,
    SOUTH,
    EAST,
    WEST,
    REPORT
)
from . robot import (
    Robot,
    State,
    Command
)


def test_operator_create_robot():
    """Test robot creation"""
    for _ in range(MAX_TEST_TIMES):
        n = random.randint(1, MAX_BOARD_SIZE)
        m = random.randint(1, MAX_BOARD_SIZE)

        board: Board = Board(n=n, m=m)

        # --------------------------------------------------------------------------------
        # TC 003 direction is incorrect.
        #   Robot creation with the state (x=0,y=0,d=NOT IN [NORTH, SOUTH, EAST, WEST])
        # Expected result: Creation succeed.
        # --------------------------------------------------------------------------------
        try:
            d: str = random_string(6)
            state: State = State(location=[0, 0], direction=d)
            Robot(board=board, state=state)
            assert False, f"Robot creation needs to fail for state({n}, {m}, {d})"
        except Exception:
            pass

        try:
            location: List[int] = [0, 0]
            d: str = NORTH
            state: State = State(location=location, direction=d)

            # --------------------------------------------------------------------------------
            # TC 001 robot creation:
            #   Robot creation with the state (x=0,y=0,d=NORTH)
            # Expected result: Creation succeeds.
            # --------------------------------------------------------------------------------
            robot: Robot = Robot(board=board, state=state)

            # --------------------------------------------------------------------------------
            # TC 002 robot initial state
            # Expected:
            #   state=([0, 0], NORTH)
            # --------------------------------------------------------------------------------
            command: Command = Command(action=REPORT, state=state)
            reply: State = robot.execute(command=command)
            assert reply['location'] == location, \
                f"Initial robot state is not {location} but {reply['location']}"
            assert reply['direction'] == NORTH, \
                f"Initial robot state is not NORTH but {reply['direction']}"

        except Exception:
            assert False, f"Robot creation needs to succeed for state({n}, {m}, {d})"
