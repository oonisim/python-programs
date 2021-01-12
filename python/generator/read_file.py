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


COMMANDS = [
    "PLACE",
    "LEFT",
    "RIGHT",
    "MOVE",
    "REPORT"
]


def execute_command(line) -> Optional[str]:
    """Parse a command line and run the corresponding command
    Args:
        line: command line
    Return:
        command executed e.g. PLACE or None if no execution
    """
    for command in COMMANDS:
        if command == "PLACE":
            pattern = r'[\t\s]*^PLACE[\t\s]+([0-9]+)[\t\s]+([0-9]+)[\t\s]+(NORTH|EAST|WEST|SOUTH)'
            if match := re.search(pattern, line, re.IGNORECASE):
                x = int(match.group(1))
                y = int(match.group(2))
                direction = match.group(3).upper()
                return command
        else:
            pattern = r'^[\t\s]*({})[\t\s]*'.format(command)
            if match := re.search(pattern, line, re.IGNORECASE):
                match.group(0).upper()
            return command

    return None


def read_commands(path: str) -> str:
    """Read lines from the file at path
    Args:
        path: file path
    Returns: line
    Raises: ValueError for file I/O error
    """
    _file = pathlib.Path(path)
    if not _file.is_file():
        raise ValueError("file {} does not exist or non file".format(path))
    else:
        with _file.open() as f:
            for line in f:
                yield line.rstrip()


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
