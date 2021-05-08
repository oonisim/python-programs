import pathlib
from itertools import islice
from typing import (
    List,
    Generator,
    Iterable
)


def take(n: int, iterable: Iterable) -> List:
    """Return next n items from the iterable as a list"""
    taken = list(islice(iterable, 0, n))
    if len(taken) > 0:
        return taken
    else:
        # https://stackoverflow.com/a/63163984/4281353
        # https://docs.python.org/3/library/exceptions.html#StopIteration
        raise StopIteration("Nothing to take")


def file_line_stream(path: str) -> Generator[str, None, None]:
    """Stream  lines from the file.
    Args:
        path: file path
    Returns: line
    Raises: RuntimeError, StopIteration

    NOTE: Generator typing
        https://stackoverflow.com/questions/57363181/
    """
    if not pathlib.Path(path).is_file():
        raise FileNotFoundError(f"file {path} does not exist or non file")
    try:
        _file = pathlib.Path(path)
        with _file.open() as f:
            for line in f:
                yield line.rstrip()
    except (IOError, FileNotFoundError) as e:
        raise RuntimeError("Read file failure") from e


def catch_stop_iteration():
    try:
        stream = file_line_stream("test.txt")
        while True:
            print(take(5, stream))
    except StopIteration as e:
        print(f"Caught StopIteration {str(e)}")
    except Exception as e:
        print("Did not catch StopIteration")


if __name__ == '__main__':
    catch_stop_iteration()
