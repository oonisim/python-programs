"""
Python generator utility module
"""
import sys
import functools
import logging
import math
import threading
from typing import (
    List,
    Dict,
    Generator,
    Callable,
    Any
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


def threadsafe_iterator(func) -> Callable[[List[Any], Dict[str, Any]], ThreadSafeIterator]:
    """A decorator that makes an iterable thread-safe.
    """
    @functools.wraps(func)
    def _iterator(*args, **kwargs):
        return ThreadSafeIterator(func(*args, **kwargs))

    return _iterator


@threadsafe_iterator
def split(sliceable, num_batches: int) -> Generator:
    """Split slice-able collection into batches to stream
    Args:
        sliceable: a slice-able object e.g. list, numpy array
        num_batches: number of batches to split
    Yields: A batch
    """
    assert len(sliceable) > 0 and num_batches > 0, \
        f"invalid data size [{len(sliceable)}] or num_batches [{num_batches}]."
    assert "__getitem__" in dir(sliceable) and (not isinstance(sliceable, Dict)), \
        f"{type(sliceable)} not slice-able."

    name: str = "split()"
    total = len(sliceable)
    _logger.info("%s: splitting [%s] records into %s batches.", name, total, num_batches)

    # Each assignment has 'quota' size which can be zero if total < number of assignments.
    quota = int(total / num_batches)

    # Left over after each assignment takes its 'quota'
    residual = total % num_batches

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

    _logger.info("%s: done", name)


def stream(
        sliceable,
        batch_size: int,
        num_batches_to_show_progress: int = sys.maxsize
) -> Generator:
    """Stream a batch at a time from a slice-able collection.
    Args:
        sliceable: a slice-able object e.g. list, numpy array, pandas dataframe
        batch_size: number of records to package into a batch to stream
        num_batches_to_show_progress: number of batches, at every consumption of which to show the progress
    Yields: A batch
    """
    assert len(sliceable) > 0 and batch_size > 0, \
        f"invalid data size [{len(sliceable)}] or batch size [{batch_size}]."
    assert "__getitem__" in dir(sliceable) and (not isinstance(sliceable, Dict)), \
        f"{type(sliceable)} not slice-able."

    name: str = "stream()"
    total_records: int = len(sliceable)
    num_batches: int = math.ceil(total_records / batch_size)
    _logger.info(
        "%s: streaming [%s] records in [%s] batches with batch size [%s].",
        name, total_records, num_batches, batch_size
    )

    position: int = 0
    while position < total_records:
        next_position: int = position + min(batch_size, total_records - position)
        yield sliceable[position:next_position]
        position = next_position

        if (position / batch_size) % num_batches_to_show_progress == 0:
            print(f"{name}: [{position}] records consumed in [{position / batch_size}] batches.")

    assert position == total_records, \
        f"expected position:[{position}] == total_records:[{total_records}]."

    _logger.info("%s: done", name)
