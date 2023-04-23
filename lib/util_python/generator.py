import logging
from typing import (
    List,
    Dict,
    Any,
    Callable,
    Generator,
)


def split(sliceable, num: int) -> Generator:
    """Split sliceable into num batches to stream
    Args:
        sliceable: a slice-able object e.g. list, numpy array
        num: number of batches to split
    Yields: A batch
    """
    assert num > 0
    assert (
        "__getitem__" in dir(sliceable)
        and (not isinstance(sliceable, Dict))
        and len(sliceable) > 0
    ), f"{type(sliceable)} not slice-able"

    logging.debug(f"split(): splitting {len(sliceable)} sliceable into {num} batches.")

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
