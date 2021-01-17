"""
OBJECTIVE:
Implement an operator which operates a robot in a self._board.
"""
from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    TypedDict,
    Final,
    Generator
)
import logging
import pathlib
import re
# Intentionally not following Google style for domain specific packages
from . constant import (
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
from . area import Board
from . robot import (
    Robot,
    State,
    Command
)


class Operator:
    """Operator class that controls a robot
    RESPONSIBILITY:
    1. Command source handling to read command lines to build robot commands.
    2. Interface with the robot.
    """
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(
        self, board: Board, path: str, log_level: int = logging.ERROR, blocking: bool = False
    ):
        """Initialize the operator
        Args:
            board: Board
            path: command line file path
        Raises:
            ValueError if the path is invalid.
        """
        if not pathlib.Path(path).is_file():
            raise FileNotFoundError(f"file {path} does not exist or non file")
        self._path: str = path

        # Board where the robot is placed
        self._board: Board = board
        # The robot which the operator is in charge of.
        self._robot: Robot = Robot(
            board=board,
            state=State(location=self._board.base, direction=NORTH),
            log_level=log_level
        )
        # Flag for synchronization operation.
        self._blocking = blocking

        self._log_level: int = log_level
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

    @property
    def path(self):
        """Command file path"""
        return self._path

    # --------------------------------------------------------------------------------
    # Command processing
    # --------------------------------------------------------------------------------
    def _build_command(
            self, command_lines: Generator[str, None, None]
    ) -> Generator[Command, None, None]:
        """Parse a command line to build a robot command to send to the robot.
        Args:
            command_lines: generator to stream command lines
        Returns: None
        """
        try:
            for line in command_lines:
                self._logger.debug("_build_command: command line [%s]", line)

                for action in COMMAND_ACTIONS:
                    if action == PLACE:
                        pattern = r'[\t\s]*^PLACE[\t\s]+([0-9]+)[\t\s]+([0-9]+)[\t\s]+(NORTH|EAST|WEST|SOUTH)[\t\s]*$'
                        if match := re.search(pattern, line, re.IGNORECASE):
                            self._logger.debug(
                                "_build_command: matched action %s", match.group(0).upper()
                            )
                            x = int(match.group(1))
                            y = int(match.group(2))
                            direction = match.group(3).upper()

                            # Handle only with a valid command line
                            if self._board.contains([x, y]) and direction in DIRECTIONS:
                                target: State = State(location=[x, y], direction=direction)
                                yield Command(action=action, state=target)
                    else:
                        pattern = r'^[\t\s]*({})[\t\s]*$'.format(action)
                        if match := re.search(pattern, line, re.IGNORECASE):
                            self._logger.debug(
                                "execute: matched action %s", match.group(0).upper()
                            )
                            dummy: State = State(location=[-1, -1], direction="NOWHERE")
                            yield Command(action=action, state=dummy)

        except StopIteration:
            self._logger.debug("_build_command(): no more command line left.")
            return  # End the generator

    def _command_line_stream(self, path: str) -> Generator[str, None, None]:
        """Stream command lines from the file command source.
        Responsibility:
            Encapsulate a command data source (e.g. file, socket, API, etc) to
            provide command lines as a stream.
        Args:
            path: file path
        Returns: line
        Raises: FileNotFoundError

        NOTE: Generator typing
            https://stackoverflow.com/questions/57363181/
        """
        if not pathlib.Path(path).is_file():
            raise FileNotFoundError(f"file {path} does not exist or non file")
        self._path: str = path
        try:
            _file = pathlib.Path(path)
            with _file.open() as f:
                for line in f:
                    yield line.rstrip()
        except (IOError, FileNotFoundError) as e:
            self._logger.error(e)

    # --------------------------------------------------------------------------------
    # Robot control
    # --------------------------------------------------------------------------------
    def _send_command(self, command: Command) -> Optional[State]:
        """Send a command to the robot.
        Args:
            command: robot command
        Return:
            State after the command or None if not executed by the robot.
        """
        return self._robot.execute(command)

    def execute(self):
        """Auto-run the operator"""
        # A generator to direct the operation interactive manner
        g: Generator[Command, None, None] = self.direct()

        # Auto run the director
        for _ in g:
            pass

    def direct(self) -> Generator[Command, None, None]:
        """Direct the robot to execute commands interactive manner
        OBJECTIVE:
            Be able to coordinate multiple robots as agents.
            The caller of the direct() can block this operator and unblock by
            sending a message via generator's send() I/F.

            To extend to asyncio or multiprocessing, this is the extension point.
        Returns:
            Generator of Command
        """
        self._logger.debug("direct(): path is %s", self.path)
        try:
            message: str = ''
            if self._log_level == logging.DEBUG:
                _stream = self._command_line_stream(self.path)
                for line in _stream:
                    self._logger.debug("direct(): line is %s", line)

            for command in self._build_command(self._command_line_stream(self.path)):
                self._logger.debug("direct(): command to sent to robot is %s", command)
                current: State = self._send_command(command)

                message = (yield current)
                if message == "STOP":
                    return

        except StopIteration:
            self._logger.debug("direct(): ended execution.")
