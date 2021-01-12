#!/usr/bin/env python
"""
SYSTEM REQUIREMENT:
- Python 3.8.x as using the assignment operation (https://www.python.org/dev/peps/pep-0572/)
- Numpy 1.19.x or later.

HOW TO EXECUTE:
- python3 -f <command file path> [-m m [-n n]]

OBJECTIVE:
Implement a robot which can move on a board of size n x m.
------------
1. Robot is always facing one of the following directions: NORTH, EAST,SOUTH or WEST.
2. Robot can turn left, right, and move one step at a time.
3. Robot cannot move off the board and ignores a command forcing it to do it.
4. Robot can report its current position on the board and direction.
5. A newly created robot must be placed into location x=0 y=0 and face NORTH.
6. The board is aligned with Cartesian coordinate system.
   Its bottom left corner has coordinate (0,0) and top right (n-1, m-1).

The solution will receive commands as a text file and outputs robot's reports into STDOUT.
Below are the only commands that robot understands:

PLACE x y direction     -- places robot into specified location on a board and set initial direction
MOVE                    -- moves robot one step in the current direction
LEFT                    -- turns robot 90 degrees anticlockwise
RIGHT                   -- turns robot 90 degrees clockwise
REPORT                  -- outputs current state into STDOUT
"""
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
)
from numbers import Number
import sys
import pathlib
import getopt
import logging
import re
import numpy as np

"""
Approaches:

Position is represented as a vector (x, y). Move is represented as vector (dx, dy).
Move is either (0, 1): NORTH, (1, 0): EAST, (-1, 0): WEST, or (0, -1): SOUTH.

At MOVE command, the current Position is updated by np.add(Position, Move) if the
destination is still on the board. np.zero(2) <= Position <= np.array([n-1, m-1)]).

LEFT/RIGHT is a rotation of the current Move. New Move is set as rotation.dot(Move).
```
    rotation = np.array([
        [np.cos(radian), np.sin(radian)],
        [-np.sin(radian), np.cos(radian)]
    ])
```
"""

# --------------------------------------------------------------------------------
# Global constants to be set in initialize().
# --------------------------------------------------------------------------------
(N, M) = -1, -1           # Board geometry. N is horizontal/x, M is vertical/y
BASE = None               # Base coordinate (0, 0)
TIP = None                # Top right coordinate (n-1, m-1)

DIRECTIONS = None         # Directions in string e.g. SOUTH
DIRECTION_TO_MOVE = None  # Mapping from direction string to move vector. e.g. SOUTH -> (0, -1)
MOVE_TO_DIRECTION = None  # Mapping from move vector to direction string. e.g. (1, 0) -> EAST
MOVES = None              # List of all available move vectors

# --------------------------------------------------------------------------------
# Robot state
# At MOVE command, the next position is via Position + Move vector add operation,
# so long as the position is on the board.
# --------------------------------------------------------------------------------
Position: List = None     # Current coordinate on board (x on N axis, y on M axis)
Direction: str = None     # Current direction in string
Move: List = None         # Move step vector e.g. (0, 1) to move north, (-1, 0) west

Logger = logging.getLogger(__name__)


def initialize(n=5, m=5) -> None:
    """Initialize the system"""
    global N
    global M
    global BASE
    global TIP
    global DIRECTIONS
    global DIRECTION_TO_MOVE
    global MOVE_TO_DIRECTION
    global MOVES

    global Position
    global Direction
    global Move
    global Logger

    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)
    Logger = logging.getLogger(__name__)

    N = n  # x / east direction
    M = m  # y / north direction

    BASE = np.zeros(2).astype(int)
    TIP = np.array([N - 1, M - 1])

    DIRECTION_TO_MOVE = {
        "NORTH": [0, 1],
        "EAST": [1, 0],
        "WEST": [-1, 0],
        "SOUTH": [0, -1]
    }
    DIRECTIONS = list(DIRECTION_TO_MOVE.keys())

    MOVE_TO_DIRECTION = {
        tuple(DIRECTION_TO_MOVE["NORTH"]): "NORTH",
        tuple(DIRECTION_TO_MOVE["EAST"]): "EAST",
        tuple(DIRECTION_TO_MOVE["WEST"]): "WEST",
        tuple(DIRECTION_TO_MOVE["SOUTH"]): "SOUTH"
    }
    MOVES = [list(_move) for _move in MOVE_TO_DIRECTION.keys()]

    Position = np.array([0, 0])
    Direction = DIRECTIONS[0]
    Move = MOVES[0]


def is_inside(position) -> bool:
    """Check if the position is inside the board
    Args:
        position: coordinates (x, y) to check.
    Return:
        True if inside
    """
    decision = np.all(position >= BASE) and np.all(position <= TIP)
    Logger.debug("position is {} and is inside is {}".format(
        position, decision
    ))
    return decision


def is_same_array(a, b) -> bool:
    """Verify if two arrays are th same"""
    return np.all(np.array(a) == np.array(b))


at_same_location = is_same_array


def place(_position=None, _direction=None) -> (List, str, List):
    """Set the current position, direction, and move.
    If the arguments are None, report the current robot state without change.
    The new position must be in the board.
    Args:
        _position: new position as coordinate (x, y)
        _direction: new direction in string
    Returns:
        (Position, Direction, Move): current robot state
    """
    Logger.debug("place: position {}, direction {}".format(_position, _direction))

    global Position
    global Direction
    global Move

    _x, _y = (_position[0], _position[1]) if _position is not None else (None, None)
    _direction = _direction.upper() if _direction is not None else "NOWHERE"

    if isinstance(_x, Number) and isinstance(_y, Number) and _direction in DIRECTIONS:
        if is_inside([_x, _y]):
            Position = [_x, _y]
            Direction = _direction
            Move = DIRECTION_TO_MOVE[Direction]

            Logger.debug("place: new position {} direction {} move {}".format(
                Position, Direction, Move
            ))

    return Position, Direction, Move


def report() -> None:
    """Report the current location and direction"""
    formatting: str = "X: {} Y: {} Direction: {}"
    position, direction, _ = place()
    print(formatting.format(position[0], position[1], direction))


def move() -> List:
    """Move a step in the current direction if the destination is in the board
    Returns:
        new position
    """
    global Position

    destination = np.add(Position, Move)
    if is_inside(destination):
        Position = destination

    return Position


def _rotate(vector, theta):
    """Rotate the vector with theta degree clock-wise
    Args:
        vector: vector to rotate
        theta: degrees to rotate
    Return:
        rotated vector
    """
    radian = np.radians(theta)
    rotation = np.array([
        [np.cos(radian), np.sin(radian)],
        [-np.sin(radian), np.cos(radian)]
    ])
    rotated = rotation.dot(vector).astype(int)
    Logger.debug("Current is {} rotation is {} rotated is {}".format(
        vector, theta, rotated
    ))
    return rotated


def rotate(theta):
    """Rotate the current move vector with theta degree
    Args:
        theta: rotation degrees
    Returns: new move vector
    """
    global Move
    global Direction

    Move = _rotate(Move, theta)
    Direction = MOVE_TO_DIRECTION[tuple(Move)]
    return Move


def left():
    """LEFT command handler"""
    return rotate(-90)


def right():
    """RIGHT command handler"""
    return rotate(90)


# ----------------------------------------------------------------------
# Mapping from a command string to its command function.
# When the command is RIGHT, then invoke the right function.
# ----------------------------------------------------------------------
COMMANDS = {
    "PLACE": place,
    "LEFT": left,
    "RIGHT": right,
    "MOVE": move,
    "REPORT": report
}


def execute_command(line) -> Optional[str]:
    """Parse a command line and run the corresponding command
    Args:
        line: command line
    Return:
        command executed e.g. PLACE or None if no execution
    """
    Logger.debug("execute_command: line [{}]".format(line))
    for command in COMMANDS.keys():
        if command == "PLACE":
            pattern = r'[\t\s]*^PLACE[\t\s]+([0-9]+)[\t\s]+([0-9]+)[\t\s]+(NORTH|EAST|WEST|SOUTH)'
            if match := re.search(pattern, line, re.IGNORECASE):
                x = int(match.group(1))
                y = int(match.group(2))
                direction = match.group(3).upper()

                Logger.debug("execute_command: matched command {}".format(
                    match.group(0).upper()
                ))
                COMMANDS[command]([x, y], direction)
                return command
        else:
            pattern = r'^[\t\s]*({})[\t\s]*'.format(command)
            if match := re.search(pattern, line, re.IGNORECASE):
                Logger.debug("execute_command: matched command {}".format(
                    match.group(0).upper()
                ))
                COMMANDS[command]()
                return command

    Logger.debug("execute_command: none executed.")
    return None


def process_commands(lines) -> None:
    """Process commands from the command file
    Args:
        lines: generator that provides a command line at each call.
    Returns: None
    """
    while True:
        try:
            execute_command(next(lines))

        except StopIteration:
            break


def read_commands(path: str) -> str:
    """Read lines from the file at path
    Args:
        path: file path
    Returns: line
    Raises: ValueError for file I/O error
    """
    Logger.debug("read_commands: path [{}]".format(path))
    _file = pathlib.Path(path)
    if not _file.is_file():
        raise ValueError("file {} does not exist or non file".format(path))
    else:
        with _file.open() as f:
            for line in f:
                yield line.rstrip()


def usage():
    """Program usage message"""
    print("{} -m <y max> -n <x max> -f <path>".format(
        sys.argv[0]
    ))


def get_options(argv) -> Optional[Tuple[str, int, int]]:
    """Handle command line options -f <file path> -m -n and -h
    Args:
        command line args from sys.argv[1:]
    Returns:
        (path, m, n) : command file path and (m, n) as board size
    """

    Logger.debug("get_path: argv [{}]".format(argv))
    path: str = ''
    m: int = -1
    n: int = -1

    try:
        opts, args = getopt.getopt(argv, "hf:m:n:")
        Logger.debug("get_path opts {} args {}".format(
            opts, args
        ))
    except getopt.GetoptError:
        Logger.error("Invalid command line")
        usage()
        return None

    if not opts:
        usage()
        return None

    for opt, arg in opts:
        if opt == '-h':     # Help message
            usage()
            return None
        elif opt == "-f":   # File path
            Logger.debug('command file is {}'.format(arg))
            path = arg
            _file = pathlib.Path(path)
            if not _file.is_file():
                print("invalid -f {}. The file does not exist or not a file.".format(arg))
                return None

        elif opt == "-m":   # board vertical size
            if arg.isdigit() and int(arg) > 0:
                m = int(arg)
            else:
                print("invalid m {}".format(arg))
                return None
        elif opt == "-n":   # board horizontal size
            if arg.isdigit() and int(arg) > 0:
                n = int(arg)
            else:
                print("invalid n {}".format(arg))
                return None

    return path, m, n


def main(argv):
    if options := get_options(argv[1:]):
        path = options[0]
        m = options[1]
        n = options[2]

        assert path
        assert m > 0
        assert n > 0

        initialize(n=n, m=m)
        process_commands(read_commands(path))


if __name__ == "__main__":
    main(sys.argv)
