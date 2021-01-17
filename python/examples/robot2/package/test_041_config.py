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
EMPTY_COMMAND_FILE = "./data/test_041_empty.txt"
COMMAND_FILE = "./data/commands.txt"
INVALID_COMMAND_FILE = "./data/test_041_invalid_commands.txt"