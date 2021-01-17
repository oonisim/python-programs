"""Pytest test case for the Board instance creation"""
import random
from . area import Board
from . test_00_config import *


def test_create_board_001():
    """Test Case:
        Board creation for size (m, n) where m < 0 or/and n < 0
    Expected: Creation fails.
    """
    for _ in range(MAX_TEST_TIMES):
        negative = random.randint(-MAX_BOARD_SIZE, -1)
        positive = random.randint(1, MAX_BOARD_SIZE)

        try:
            board: Board = Board(n=negative, m=0)
            assert False, f"Board creation needs to fail for (x, y)=({negative},0)"
        except Exception:
            pass

        try:
            board: Board = Board(n=0, m=negative)
            assert False, f"Board creation needs to fail for (x, y)=(0,{negative})"
        except Exception:
            pass

        try:
            board: Board = Board(n=positive, m=0)
            assert False, f"Board creation needs to fail for (x, y)=({positive},0)"
        except Exception:
            pass

        try:
            board: Board = Board(n=positive, m=0)
            assert False, f"Board creation needs to fail for (x, y)=({positive},0)"
        except Exception:
            pass

        try:
            board: Board = Board(n=negative, m=negative)
            print()
            assert False, f"Board creation needs to fail for (x, y)=({negative},{negative})"
        except Exception:
            pass


def test_create_board_002():
    """
    Test Case:
        Board creation for size (m, n) where m 0 0 or n = 0
    Expected: Creation fails.
    """

    try:
        board: Board = Board(n=0, m=0)
        assert False, "Board creation needs to fail for (x, y)=(0,0)"
    except Exception:
        pass

    for _ in range(MAX_TEST_TIMES):
        negative = random.randint(-MAX_BOARD_SIZE, -1)
        positive = random.randint(1, MAX_BOARD_SIZE)

        try:
            board: Board = Board(n=positive, m=0)
            assert False, f"Board creation needs to fail for (x, y)=({positive},0)"
        except Exception:
            pass

        try:
            board: Board = Board(n=positive, m=0)
            assert False, f"Board creation needs to fail for (x, y)=({positive},0)"
        except Exception:
            pass


def test_create_board_003():
    """
    Test Case:
        Board creation for size (m, n) where m > 0 or/and > < 0
    Expected: Creation fails.
    """
    for _ in range(MAX_TEST_TIMES):
        n = random.randint(1, MAX_BOARD_SIZE)
        m = random.randint(1, MAX_BOARD_SIZE)

        try:
            board: Board = Board(n=n, m=m)
        except Exception:
            assert False, f"Board creation needs to succeed for (x, y)=({n},{m})"
