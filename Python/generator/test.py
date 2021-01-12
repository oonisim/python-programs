from typing import (
    Optional,
    Tuple
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


def main(argv):
    def test() -> Optional[Tuple[int, int, str]]:
        return 0, 1, "hoge"

    if match := test():
        print(match)


if __name__ == "__main__":
    main(sys.argv)
