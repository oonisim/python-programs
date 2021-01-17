"""Pytest test case for an operator handles an invalid path"""
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
from . test_00_config import *
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


def tc_001():
    """TC 001: Operator handles an invalid path.
    """
    _path = random_string(40)
    try:
        operator: Operator = create_operator(_path)
        assert False, f"Operator needs to fail for a non-existing file path {_path}"
    except FileNotFoundError:
        pass
    except Exception as e:
        assert False, \
            f"Operator needs to fail properly with a file path {_path} but raised {e}"


def test_operator_handle_invalid_path():
    """
    Test Case:
        Operator handles an invalid file path.
    Expected: Operator fails.
    """
    tc_001()
