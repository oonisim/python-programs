""" Module docstring
[Objective]

[Prerequisites]

[Assumptions]

[Note]

[TODO]

"""
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Union,
    Optional,
)
import os
import sys
import logging

from utility import get_logger


# ================================================================================
# Global constants
# ================================================================================
LOG_LEVEL_NAME = (
    os.environ["LOG_LEVEL_NAME"].upper()
    if "LOG_LEVEL_NAME" in os.environ
    else logging.getLevelName(logging.DEBUG)
)
assert hasattr(logging, LOG_LEVEL_NAME), f"Invalid log level name {LOG_LEVEL_NAME}."
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME)


# ================================================================================
# Base class
# ================================================================================
class Base:
    """
    Base class for [TBD]
    """
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------
    @staticmethod
    def get_python_version() -> str:
        """
        Returns: Python version running on
        """
        return sys.version_info

    @staticmethod
    def get_fully_qualified_name():
        """
        Returns: FQN of this python module"""
        return __name__

    # --------------------------------------------------------------------------------
    # Class
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    def __init__(self):
        self._log_level: int = LOG_LEVEL
        self._logger: logging.Logger = get_logger(name=__name__, level=LOG_LEVEL)

    @property
    def log_level(self) -> int:
        """
        Returns: Current log level of the module
        """
        return self._log_level

    @log_level.setter
    def log_level(self, log_level: int):
        """Set the logging level
        Args:
            log_level: Logging level to set
        """
        assert log_level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL
        ], f"invalid log level {log_level}"
        self._logger.setLevel(log_level)

    @property
    def logger(self) -> logging.Logger:
        """Get instance level logger
        Returns: Logger instance
        """
        return self._logger

    # --------------------------------------------------------------------------------
    # Logic
    # --------------------------------------------------------------------------------
    def run(self):
        """Instance run method"""
        raise NotImplemented("To be implemented")


# ================================================================================
# Tests (to be separated)
# ================================================================================
def test_get_fully_qualified_name():
    """
    Test condition:
        The function returns the fully qualified python module name in the dot format (e.g. x.y.z).
    Prerequisite:

    Assumptions:

    Expected:

    Validation:
        <How to validate if the test condition is met or the deviation (expected - actual) detected>
    """
    instance: Base = Base()
    name: str = instance.get_fully_qualified_name()
    assert name == __name__
