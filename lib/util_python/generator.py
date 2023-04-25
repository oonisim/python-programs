"""
Python generator utility module
"""
import functools
import logging
import threading
from typing import (
    Dict,
    Iterator,
)

from util_logging import (
    get_logger
)

_logger: logging.Logger = get_logger(name=__name__)


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
class ThreadSafeIterator:
    """Wrapper class to make an iterable thread-safe
    Generator and iterator are not thread safe. Generate cannot be thread-safe
    because if two threads call next method on a generator at the same time,
    it will raise an exception ValueError: generator already executing.

    The only way to fix it is by wrapping it in an iterator and have a lock
    that allows only one thread to call next method of the generator.

    See
        * https://anandology.com/blog/using-iterators-and-generators/
        * https://docs.python.org/3/library/functions.html#iter
        * https://anandology.com/blog/using-iterators-and-generators/
    """
    def __init__(self, iterable):
        self.lock = threading.Lock()
        self.iterable = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.iterable.__next__()


def threadsafe_iterator(func):
    """A decorator that makes an iterable thread-safe.
    """
    @functools.wraps(func)
    def _iterator(*args, **kwargs):
        return ThreadSafeIterator(func(*args, **kwargs))
    return _iterator


@threadsafe_iterator
def split(sliceable, num: int) -> Iterator:
    """Split slice-able collection into batches to stream
    Args:
        sliceable: a slice-able object e.g. list, numpy array
        num: number of batches to split
    Yields: A batch
    """
    assert num > 0
    assert (    # To be able to slice, __getitem__ method is required
        "__getitem__" in dir(sliceable)
        and (not isinstance(sliceable, Dict))
        and len(sliceable) > 0
    ), f"{type(sliceable)} not slice-able"

    _logger.debug("split(): splitting %s sliceable into %s batches.", len(sliceable), num)

    # Total rows
    total = len(sliceable)

    # Each assignment has 'quota' size which can be zero if total < number of assignments.
    quota = int(total / num)

    # Left over after each assignment takes its 'quota'
    residual = total % num

    start: int = 0
    while start < total:
        # If there is residual, each batch has (quota + 1).
        if residual > 0:
            size = quota + 1
            residual -= 1
        else:
            size = quota

        end: int = start + size
        yield sliceable[start: min(end, total)]

        start = end
        end += size
