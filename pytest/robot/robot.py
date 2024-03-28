from enum import Enum
from typing import (
    Tuple
)


class Direction(Enum):
    NORTH: int = 1
    EAST: int = 2
    SOUTH: int = 3
    WEST: int = 4


class Command(Enum):
    PLACE: int = 0
    LEFT: int = 1
    RIGHT: int = 2
    AHEAD: int = 3
    REPORT: int = 4


X_MIN: int = 0
X_MAX: int = 7
Y_MIN: int = 0
Y_MAX: int = 8


class BoardIF:
    @property
    def width(self) -> int:
        """Board width"""

    @property
    def height(self) -> int:
        """Board height"""

    def __init__(self, n: int, m: int):
        """Initialize the robot and its limit
        Args:
            n: board horizontal size
            m: board vertical size
        """

    def is_on_board(self, x, y) -> bool:
        """Tell if the coordinate (x, y) is on-board.
        Args:
            x: x coordinate
            y: y coordinate
        Returns: True if on-board. Otherwise, False
        """


class RobotIF:
    def __init__(self):
        """Initialize the robot and its limit
       """

    def execute(self, command: Command, **cmdargs) -> Tuple[int, int, Direction]:
        """Execute the command from the operator
        Args:
            command: command to execute
            cmdargs: command parameters {"x": x, "y": y, "direction": direction} or None

        Returns: (x, y, direction) after the command execution.
        """
