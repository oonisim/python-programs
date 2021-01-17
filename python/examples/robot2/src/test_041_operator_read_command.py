"""Pytest test case for the operator read commands"""
from typing import (
    List,
    Dict,
    Tuple
)
import random
import string
import logging
import numpy as np
from . area import Board
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
from . operator import Operator
from . test_00_config import *
from . test_041_config import *


def tc_001_empty(_path: str):
    """TC 001: Operator reads an empty command file
    """
    operator: Operator
    try:
        operator = create_operator(_path)
        _stream = operator._command_line_stream(_path)
        for line in _stream:
            print(f"TC 001 Empty: Command line from {_path} is {line}")

        _stream.close()
    except Exception as e:
        assert False, \
            f"Operator needs to succeed with a file path {_path} but raised {e}"


def tc_001(_path: str):
    """TC 001: Operator reads an empty command file
    """
    try:
        operator: Operator = create_operator(_path)
        _stream = operator._command_line_stream(_path)
        for line in _stream:
            print(f"TC 001: Command line from {_path} is {line}")

        del operator
    except Exception as e:
        assert False, \
            f"Operator needs to succeed path {_path} but raised {e}"


def tc_002(_path: str):
    """TC 002: Operator reads non commands
    """
    try:
        operator: Operator = create_operator(_path)
        _stream = operator._command_line_stream(_path)
        _commands = operator._build_command(_stream)
        for command in _commands:
            print(f"TC 002: Command from {_path} is {command}")
            assert False,\
                f"Not suppose to see {command} for invalid command lines"

        del operator
    except AssertionError as e:
        raise e
    except Exception as e:
        assert False, \
            f"Operator needs to succeed path {_path} but raised {e}"


def tc_004(_path: str):
    """TC 004: Operator reads commands
    """
    try:
        operator: Operator = create_operator(_path,log_level=logging.DEBUG)
        _stream = operator._command_line_stream(_path)
        _commands = operator._build_command(_stream)
        commands = list(_commands)
        assert len(commands) > 0, "Did not build any commands for path {_path}"

        del operator
    except Exception as e:
        assert False, \
            f"Operator needs to succeed path {_path} but raised {e}"


def test_operator_handle_read_commands():
    """
    Test Case:
        Operator handles an invalid file path.
    Expected: Operator fails.
    """
    # TC: 001 Read a command file (empty)
    tc_001_empty(EMPTY_COMMAND_FILE)

    # TC: 001 Read a command file
    tc_001(EMPTY_COMMAND_FILE)

    # TC: 002 Read invalid commands
    tc_002(INVALID_COMMAND_FILE)

    # TC: 004 Read commands
    tc_004(COMMAND_FILE)

