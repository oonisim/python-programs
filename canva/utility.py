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
import functools
import time
import cProfile

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


# ================================================================================
# Generic utility functions
# ================================================================================
def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Logger instance factory method
    See https://docs.python.org/2/howto/logging.html#logging-advanced-tutorial
    The logger name should follow the package/module hierarchy

    Args:
        name: logger name following the package/module hierarchy
        level: optional log level
    Returns:
        logger instance
    """
    _logger = logging.getLogger(name=name)
    _logger.setLevel(level if level else logging.DEBUG)
    return _logger


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        print(f"\nTiming {func.__name__!r}")

        value = func(*args, **kwargs)

        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def cprofile(func):
    """Profile the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        print(f"\nprofiling {func.__name__!r}")
        profiler: cProfile.Profile = cProfile.Profile()
        profiler.enable()

        value = func(*args, **kwargs)

        profiler.disable()
        profiler.print_stats(sort="cumtime")

        print(f"\nprofiling {func.__name__!r}")
        return value
    return wrapper_timer


@timer
@cprofile
def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(10000)])


def test():
    waste_some_time(3)
