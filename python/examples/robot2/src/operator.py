"""
OBJECTIVE:
Implement an operator which operates a robot in a board.

RESPONSIBILITY:
Director of the Director Pattern of GoF used e.g. compiler parser.
Separate the parsing command source and from the robot
1. Operator receive the information of a board, a robot, and the command file.
2. Operator read lines from the command file and send them to the robot.
"""
from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    TypedDict,
    Final
)
import logging
import pathlib
import re
# Intentionally not following Google style for domain specific packages
from constant import (
    NORTH,
    SOUTH,
    EAST,
    WEST,
    DIRECTIONS,
    MOVES,
    PLACE,
    REPORT,
    MOVE,
    LEFT,
    RIGHT,
    COMMAND_ACTIONS
)
from board import Board
import robot
from robot import (
    Robot,
    State,
    Command
)


class Operator(object):
    """Operator to operates a robot on a board"""
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(self, board: Board, path: str, log_level: int = logging.ERROR):
        """Initialize the operator
        Args:
            board: Board
            path: command line file path
        Raises:
            ValueError if the path is invalid.
        """
        if not pathlib.Path(path).is_file():
            raise ValueError("file {} does not exist or non file".format(path))
        self._path: str = path

        # Board where the robot is placed
        self._board: Board = board
        # The robot which the operator is in charge of.
        self._robot: Robot = Robot(
            board=board,
            state=State(location=board.base, direction=NORTH),
            log_level=log_level
        )

        logging.basicConfig()
        self._log_level: int = log_level
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

    # --------------------------------------------------------------------------------
    # Functions
    # --------------------------------------------------------------------------------
    def _send_command(self, line) -> Optional[State]:
        """Parse a command line and run the corresponding command
        Args:
            line: command line
        Return:
            State after the command or None if not executed by the robot.
        """
        self._logger.debug("execute: line [{}]".format(line))
        for action in COMMAND_ACTIONS:
            if action == PLACE:
                pattern = r'[\t\s]*^PLACE[\t\s]+([0-9]+)[\t\s]+([0-9]+)[\t\s]+(NORTH|EAST|WEST|SOUTH)'
                if match := re.search(pattern, line, re.IGNORECASE):
                    self._logger.debug("execute: matched action {}".format(
                        match.group(0).upper()
                    ))

                    x = int(match.group(1))
                    y = int(match.group(2))
                    direction = match.group(3).upper()

                    anterior: State = State(location=[x, y], direction=direction)
                    posterior: State = self._robot.execute(Command(action=action, state=anterior))
                    return posterior
            else:
                pattern = r'^[\t\s]*({})[\t\s]*'.format(action)
                if match := re.search(pattern, line, re.IGNORECASE):
                    self._logger.debug("execute: matched action {}".format(
                        match.group(0).upper()
                    ))

                    anterior: State = State(location=[-1, -1], direction="NOWHERE")
                    posterior: State = self._robot.execute(Command(action=action, state=anterior))
                    return posterior

        self._logger.debug("execute: none executed.")
        return None

    def process_commands(self, commands) -> None:
        """Process commands from the command file
        Args:
            commands: generator that provides a command line at each call.
        Returns: None
        """
        while True:
            try:
                self._send_command(next(self._read_commands(self._path)))

            except StopIteration:
                self._logger.debug("process_commands(): no more command.")
                break

    def _read_commands(self, path: str) -> str:
        """Read lines from the file at path
        Args:
            path: file path
        Returns: line
        Raises: ValueError for file I/O error
        """
        try:
            _file = pathlib.Path(path)
            with _file.open() as f:
                for line in f:
                    yield line.rstrip()
        except Exception as e:
            self._logger.error(e)
