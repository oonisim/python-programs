"""
OBJECTIVE:
Implements system-wide constant values
"""
from typing import (
    List,
    Dict,
    TypedDict,
    Final
)
import numpy as np

# ================================================================================
# Cardinal directions
# ================================================================================
NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"

# Mapping from direction string to move vector. e.g. SOUTH -> (0, -1)
DIRECTION_TO_MOVE: Final[Dict[str, np.ndarray]] = {
    NORTH: np.ndarray([0, 1]),
    EAST: np.ndarray([1, 0]),
    WEST: np.ndarray([-1, 0]),
    SOUTH: np.ndarray([0, -1])
}

# Directions in string e.g. SOUTH
DIRECTIONS: Final[List[str]] = list(DIRECTION_TO_MOVE.keys())

# ================================================================================
# Robot constant
# ================================================================================
# Mapping from move vector to direction string. e.g. (1, 0) -> EAST
MOVE_TO_DIRECTION: Final[Dict[np.ndarray, str]] = {
    DIRECTION_TO_MOVE[NORTH]: NORTH,
    DIRECTION_TO_MOVE[EAST]: EAST,
    DIRECTION_TO_MOVE[WEST]: WEST,
    DIRECTION_TO_MOVE[SOUTH]: SOUTH
}

# List of all available move vectors
MOVES: Final[List[np.ndarray]] = [_move for _move in MOVE_TO_DIRECTION.keys()]

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
