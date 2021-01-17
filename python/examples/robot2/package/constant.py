"""
OBJECTIVE:
Implements system-wide constant values
"""
from typing import (
    List,
    Dict,
    Final
)
import logging
import numpy as np

# ================================================================================
# Log
# ================================================================================
DEFAULT_LOG_LEVEL = logging.ERROR

# ================================================================================
# Cardinal directions
# ================================================================================
NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"

# Mapping from direction string to move vector. e.g. SOUTH -> (0, -1)
DIRECTION_TO_MOVE: Final[Dict[str, np.ndarray]] = {
    NORTH: [0, 1],
    EAST: [1, 0],
    WEST: [-1, 0],
    SOUTH: [0, -1]
}

# Directions in string e.g. SOUTH
DIRECTIONS: Final[List[str]] = list(DIRECTION_TO_MOVE.keys())

# ================================================================================
# Board constant
# ================================================================================
DEFAULT_N = 10      # Default board size for n
DEFAULT_M = 11      # Default board size for m

# ================================================================================
# Robot constant
# ================================================================================
# Mapping from move vector to direction string. e.g. (1, 0) -> EAST
MOVE_TO_DIRECTION: Final[Dict[np.ndarray, str]] = {
    tuple(DIRECTION_TO_MOVE[NORTH]): NORTH,
    tuple(DIRECTION_TO_MOVE[EAST]): EAST,
    tuple(DIRECTION_TO_MOVE[WEST]): WEST,
    tuple(DIRECTION_TO_MOVE[SOUTH]): SOUTH
}

# List of all available move vectors
MOVES: Final[List[np.ndarray]] = [list(_move) for _move in MOVE_TO_DIRECTION]

# Robot commands
PLACE = "PLACE"
REPORT = "REPORT"
MOVE = "MOVE"
LEFT = "LEFT"
RIGHT = "RIGHT"

COMMAND_ACTIONS = [
    PLACE,
    REPORT,
    MOVE,
    LEFT,
    RIGHT
]
