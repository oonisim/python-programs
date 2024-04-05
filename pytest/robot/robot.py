from abc import (
    ABC,
    abstractmethod,
)
from enum import Enum
from typing import Tuple


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


class BoardIF(ABC):
    @abstractmethod
    @property
    def width(self) -> int:
        """Board width"""

    @abstractmethod
    @property
    def height(self) -> int:
        """Board height"""

    def __init__(self, w: int, h: int):
        """Initialize the robot and its limit
        Args:
            w: board vertical size
            h: board horizontal size
        """

    @abstractmethod
    def is_on_board(self, x, y) -> bool:
        """Tell if the coordinate (x, y) is on-board.
        Args:
            x: x coordinate
            y: y coordinate
        Returns: True if on-board. Otherwise, False
        """


class RobotIF(ABC):
    def __init__(self):
        """Initialize the robot and its limit"""

    @abstractmethod
    def execute(self, command: Command, **cmdargs) -> Tuple[int, int, Direction]:
        """Execute the command from the operator
        Args:
            command: command to execute
            cmdargs: command parameters {"x": x, "y": y, "direction": direction} or None

        Returns: (x, y, direction) after the command execution.
        """

    def __call__(self, *args, **kwargs) -> Tuple[int, int, Direction]:
        """Reports the current coordinate and direction.
        Returns: (x, y, direction) after the command execution.
        """
        return self.execute(command=Command.REPORT)
