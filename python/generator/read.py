from typing import (
    Optional,
    Tuple,
    List,
    Dict,
)
import sys
import os
import pathlib
import getopt
import logging
import re

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
Logger = logging.getLogger(__name__)

COMMANDS = {
    "PLACE": place,
    "LEFT": left,
    "RIGHT": right,
    "MOVE": move,
    "REPORT": report
}

def read_lines(path: str) -> str:
    _file = pathlib.Path(path)
    if not _file.is_file():
        raise ValueError("file {} does not exist or non file".format(path))
    else:
        with _file.open() as f:
            for line in f:
                yield line


def parse(lines) -> None:
    """Parse commands"""
    while True:
        try:
            line = next(lines)
            Logger.debug("parse: line {}".format(line))
            if args := parse_place_command(line):
                command = args[0]
                x = args[1]
                y = args[2]
                direction = args[3]
                Logger.debug("parse: command {} x {} y {} direction {}".format(
                    command, x, y, direction
                ))
            elif command := parse_command(line, "MOVE"):
                Logger.debug("parse: command {}".format(command))
            elif command := parse_command(line, "LEFT"):
                Logger.debug("parse: command {}".format(command))
            elif command := parse_command(line, "RIGHT"):
                Logger.debug("parse: command {}".format(command))
            elif command := parse_command(line, "REPORT"):
                Logger.debug("parse: command {}".format(command))
            else:
                Logger.debug("parse: no command found {}")
                pass

        except StopIteration:
            break


def main(argv):
    path: str = ""
    try:
        opts, args = getopt.getopt(argv[1:], "hf:")
        print(opts)
        print(args)
    except getopt.GetoptError:
        print("{} -f <path>".format(
            sys.argv[0]
        ))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("{} -f <path>".format(
                sys.argv[0]
            ))
            sys.exit()
        elif opt in ("-f") :
            path = arg

    Logger.debug('Input file is {}'.format(path))
    parse(read_lines(path))


if __name__ == "__main__":
    main(sys.argv)
