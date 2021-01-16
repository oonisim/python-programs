#!/usr/bin/env python
"""
OBJECTIVE:
Define the Board class.
"""
from typing import (
    Union,
    List,
    Dict,
    Tuple,
    Final,
    Optional
)
import logging
import numpy as np


class Board:
    """Board of the size (n, m) where a robot functions"""
    # --------------------------------------------------------------------------------
    # Class initialization
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # Instance initialization
    # --------------------------------------------------------------------------------
    def __init__(self, n: int, m: int, log_level=logging.ERROR) -> None:
        """Initialize the board
        Args:
            n: Horizontal/west-to-east size of the board
            m: Vertical/south-to-north sizeof the board
            log_level: logging level or logging.ERROR by default
        """
        self._n = n
        self._m = m
        # Base coordinate of the board (0,0)
        self._base = np.zeros(2).astype(int)
        # Farthest (north, east) coordinate of the self._board.
        self._tip = np.array([n - 1, m - 1])

        logging.basicConfig()
        self._log_level = log_level
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(self._log_level)

        assert n > 0, "Board size need to be positive"
        assert m > 0, "Board size need to be positive"

    @property
    def n(self):
        """Horizontal/west-to-east size of the board"""
        return self._n

    @property
    def m(self) -> int:
        """Vertical/south-to-north sizeof the board"""
        return self._n

    @property
    def base(self) -> List[int]:
        """base coordinate of the board"""
        return [self._base[0], self._base[1]]

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the board"""
        return self._n, self._m

    @property
    def log_level(self):
        """Logging level from the logging package"""
        return self._n

    @log_level.setter
    def log_level(self, level: int = logging.ERROR):
        """Set the logging level
        Args:
            level: log level from the logging package, default to ERROR
        """
        self._log_level = level if level in [
            logging.NOTSET,
            logging.DEBUG,
            logging.INFO,
            logging.WARN,
            logging.ERROR,
            logging.FATAL,
            logging.CRITICAL
        ] else logging.ERROR

    # --------------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------------
    @staticmethod
    def is_same_array(a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> bool:
        """Verify if two arrays are th same"""
        return np.all(a == b)

    at_same_location = is_same_array

    # --------------------------------------------------------------------------------
    # I/F
    # --------------------------------------------------------------------------------
    def contains(self, position: Union[List[int], np.ndarray]) -> bool:
        """Check if the position is inside the board
        Args:
            position: coordinates (x, y) to check.
        Return:
            True if inside
        """
        decision = np.all(self._base <= position) and np.all(position <= self._tip)
        self._logger.debug("position is %s and is inside is %s", position, decision)
        return decision
