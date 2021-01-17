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

--------------------------------------------------------------------------------
APPROACH:

Position is represented as a vector (x, y). Move is represented as vector (dx, dy).
Move is either (0, 1): NORTH, (1, 0): EAST, (-1, 0): WEST, or (0, -1): SOUTH.

At MOVE command, the current Position is updated by np.add(Position, Move) if the
destination is still on the self._board. np.zero(2) <= Position <= np.array([n-1, m-1)]).

LEFT/RIGHT is a rotation of the current Move. New Move is set as rotation.dot(Move).
```
    rotation = np.array([
        [np.cos(radian), np.sin(radian)],
        [-np.sin(radian), np.cos(radian)]
    ])
```
"""
from typing import (
    Optional,
    Tuple,
)
import sys
import pathlib
import getopt
import logging
from . area import Board
from . operator import Operator

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)


def read_commands(path: str) -> str:
    """Read lines from the file at path
    Args:
        path: file path
    Returns: line
    Raises: ValueError for file I/O error
    """
    LOGGER.debug("read_commands: path [%s]", path)
    _file = pathlib.Path(path)
    if not _file.is_file():
        raise ValueError(f"file {path} does not exist or non file")

    with _file.open() as f:
        for line in f:
            yield line.rstrip()


def usage():
    """Program usage message"""
    print("{sys.argv[0]} -m <y max> -n <x max> -f <path>")


def get_options(argv) -> Optional[Tuple[str, int, int]]:
    """Handle command line options -f <file path> -m -n and -h
    Args:
        command line args from sys.argv[1:]
    Returns:
        (path, m, n) : command file path and (m, n) as board size
    """

    LOGGER.debug("get_path: argv [%s]", argv)
    path: str = ''
    m: int = -1
    n: int = -1

    try:
        opts, args = getopt.getopt(argv, "hf:m:n:")
        LOGGER.debug("get_path opts %s args %s", opts, args)
        if not opts:
            return None

        for opt, arg in opts:
            if opt == '-h':  # Help message
                usage()
            elif opt == "-f":  # File path
                LOGGER.debug("command file is %s", arg)
                if pathlib.Path(arg).is_file():
                    path = arg
                else:
                    path = ''
                    print("invalid -f %s. The file does not exist or not a file.", arg)
            elif opt == "-m":  # board vertical size
                if arg.isdigit() and int(arg) > 0:
                    m = int(arg)
                else:
                    print(f"invalid m {arg}")
                    return None
            elif opt == "-n":  # board horizontal size
                if arg.isdigit() and int(arg) > 0:
                    n = int(arg)
                else:
                    print(f"invalid n {arg}")
            else:
                LOGGER.debug("unknown option %s", opt)

    except getopt.GetoptError:
        LOGGER.error("Invalid command line")
        usage()

    return path, m, n if (path and m > 0 and n > 0) else None


def main(argv):
    """Run the system
    Args:
        argv: command line parameters
    """
    if options := get_options(argv[1:]):
        path = options[0]
        m = options[1]
        n = options[2]

        assert path
        assert m > 0
        assert n > 0

        board: Board = Board(n, m)
        operator: Operator = Operator(
            board=board, path=path, log_level=logging.DEBUG,blocking=False
        )
        operator.direct()


if __name__ == "__main__":
    main(sys.argv)
