"""
Windows-specific error code indicating an invalid pathname.

See Also
----------
https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
    Official listing of all such codes.

https://stackoverflow.com/a/34102855/4281353
"""
from typing import (
    List,
    Generator,
    Iterable,
    NoReturn
)
import errno
import os
import sys
import tempfile
import pathlib
import logging
import pickle
from itertools import islice

import function.common.base as base

# Sadly, Python fails to provide the following magic number for us.
ERROR_INVALID_NAME = 123
Logger = logging.getLogger(__name__)


class Function(base.Function):
    class GenearatorHasNoMore(Exception):
        def __init__(self):
            super().__init__("No more left to offer from the generator")

    @staticmethod
    def is_pathname_valid(pathname: str) -> bool:
        """
        `True` if the passed pathname is a valid pathname for the current OS;
        `False` otherwise.
        """
        # If this pathname is either not a string or is but is empty, this pathname
        # is invalid.
        try:
            if not isinstance(pathname, str) or not pathname:
                return False

            # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
            # if any. Since Windows prohibits path components from containing `:`
            # characters, failing to strip this `:`-suffixed prefix would
            # erroneously invalidate all valid absolute Windows pathnames.
            _, pathname = os.path.splitdrive(pathname)

            # Directory guaranteed to exist. If the current OS is Windows, this is
            # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
            # environment variable); else, the typical root directory.
            root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
                if sys.platform == 'win32' else os.path.sep
            assert os.path.isdir(root_dirname)  # ...Murphy and her ironclad Law

            # Append a path separator to this directory if needed.
            root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

            # Test whether each path component split from this pathname is valid or
            # not, ignoring non-existent and non-readable path components.
            for pathname_part in pathname.split(os.path.sep):
                try:
                    os.lstat(root_dirname + pathname_part)
                # If an OS-specific exception is raised, its error code
                # indicates whether this pathname is valid or not. Unless this
                # is the case, this exception implies an ignorable kernel or
                # filesystem complaint (e.g., path not found or inaccessible).
                #
                # Only the following exceptions indicate invalid pathnames:
                #
                # * Instances of the Windows-specific "WindowsError" class
                #   defining the "winerror" attribute whose value is
                #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
                #   fine-grained and hence useful than the generic "errno"
                #   attribute. When a too-long pathname is passed, for example,
                #   "errno" is "ENOENT" (i.e., no such file or directory) rather
                #   than "ENAMETOOLONG" (i.e., file name too long).
                # * Instances of the cross-platform "OSError" class defining the
                #   generic "errno" attribute whose value is either:
                #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
                #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
                except OSError as exc:
                    if hasattr(exc, 'winerror'):
                        if exc.winerror == ERROR_INVALID_NAME:
                            return False
                    elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                        return False
        # If a "TypeError" exception was raised, it almost certainly has the
        # error message "embedded NUL character" indicating an invalid pathname.
        except TypeError as exc:
            return False
        # If no exception was raised, all path components and hence this
        # pathname itself are valid. (Praise be to the curmudgeonly python.)
        else:
            return True
        # If any other exception was raised, this is an unrelated fatal issue
        # (e.g., a bug). Permit this exception to unwind the call stack.
        #
        # Did we mention this should be shipped with Python already?

    @staticmethod
    def is_path_creatable(pathname: str) -> bool:
        """
        `True` if the current user has sufficient permissions to create the passed
        pathname; `False` otherwise.
        """
        # Parent directory of the passed path. If empty, we substitute the current
        # working directory (CWD) instead.
        dirname = os.path.dirname(pathname) or os.getcwd()
        return os.access(dirname, os.W_OK)

    @staticmethod
    def is_path_exists_or_creatable(pathname: str) -> bool:
        """
        `True` if the passed pathname is a valid pathname for the current OS _and_
        either currently exists or is hypothetically creatable; `False` otherwise.

        This function is guaranteed to _never_ raise exceptions.
        """
        try:
            # To prevent "os" module calls from raising undesirable exceptions on
            # invalid pathnames, is_pathname_valid() is explicitly called first.
            return Function.is_pathname_valid(pathname) and (
                    os.path.exists(pathname) or Function.is_path_creatable(pathname))
        # Report failure on non-fatal filesystem complaints (e.g., connection
        # timeouts, permissions issues) implying this path to be inaccessible. All
        # other exceptions are unrelated fatal issues and should not be caught here.
        except OSError:
            return False

    @staticmethod
    def is_path_sibling_creatable(pathname: str) -> bool:
        """
        `True` if the current user has sufficient permissions to create **siblings**
        (i.e., arbitrary files in the parent directory) of the passed pathname;
        `False` otherwise.
        """
        # Parent directory of the passed path. If empty, we substitute the current
        # working directory (CWD) instead.
        dirname = os.path.dirname(pathname) or os.getcwd()

        try:
            # For safety, explicitly close and hence delete this temporary file
            # immediately after creating it in the passed path's parent directory.
            with tempfile.TemporaryFile(dir=dirname):
                pass
            return True
        # While the exact type of exception raised by the above function depends on
        # the current version of the Python interpreter, all such types subclass the
        # following exception superclass.
        except EnvironmentError:
            return False

    @staticmethod
    def is_path_exists_or_creatable_portable(pathname: str) -> bool:
        """
        `True` if the passed pathname is a valid pathname on the current OS _and_
        either currently exists or is hypothetically creatable in a cross-platform
        manner optimized for POSIX-unfriendly filesystems; `False` otherwise.

        This function is guaranteed to _never_ raise exceptions.
        """
        try:
            # To prevent "os" module calls from raising undesirable exceptions on
            # invalid pathnames, is_pathname_valid() is explicitly called first.
            return Function.is_pathname_valid(pathname) and (
                    os.path.exists(pathname) or Function.is_path_sibling_creatable(pathname))
        # Report failure on non-fatal filesystem complaints (e.g., connection
        # timeouts, permissions issues) implying this path to be inaccessible. All
        # other exceptions are unrelated fatal issues and should not be caught here.
        except OSError:
            return False

    @staticmethod
    def is_file(path_to_file) -> bool:
        return pathlib.Path(path_to_file).is_file()

    @staticmethod
    def rm_file(path_to_file):
        path = pathlib.Path(path_to_file)
        path.unlink()

    @staticmethod
    def read_file(path_to_file):
        assert Function.is_file(path_to_file), f"File {path_to_file} does not exist"
        with open(path_to_file, 'r') as _file:
            content = _file.read()

        return content

    @staticmethod
    def serialize(path: str, state: object) -> NoReturn:
        """Serialize the object to the file by overwriting.
        Args:
            path: file path
            state: object to serialize

        """
        assert Function.is_path_creatable(path), f"Cannot create {path}."
        with open(path, 'wb+') as f:
            pickle.dump(state, f)

    @staticmethod
    def deserialize(path: str):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                state = pickle.load(f)

            assert state is not None
            return state
        else:
            raise RuntimeError(f"Path {path} does not exist.")

    @staticmethod
    def take(n: int, iterator: Iterable) -> List:
        """Return n items from the iterable as a list
        Args:
            n: number of lines to take
            iterator: iterable
        Returns: List of lines
        Raises: StopIteration when there is none to take
        """

        taken = list(islice(iterator, 0, n))
        # --------------------------------------------------------------------------------
        # list(generator) returns [] instead of raising the StopIteration.
        # Need to verify if list(generator) size is > 0 or not.
        # --------------------------------------------------------------------------------
        if len(taken) > 0:
            return taken
        else:
            # --------------------------------------------------------------------------------
            # You can NOT raise StopIteration from within a Generator to notify
            # no-more-to-yield from the generator. Must raise other exception or
            # return None, [] etc.
            # --------------------------------------------------------------------------------
            # https://stackoverflow.com/a/63163984/4281353
            # https://www.python.org/dev/peps/pep-0479/
            # Raising StopIteration from within a Generator will end the generator.
            # Python implicit conversion from StopItraton to RuntimeError.
            #
            # https://stackoverflow.com/questions/67444910
            # StopIteration serves a very specific purpose
            # (to allow a next method to indicate iteration is complete), and
            # reusing it for other purposes will cause problems
            # --------------------------------------------------------------------------------
            # raise StopIteration("Nothing left to take")
            raise Function.GenearatorHasNoMore

    @staticmethod
    def take_n_from_i(n, i, iterator: Iterable):
        """Return n items from the i-th line from the iterable as a list
        Args:
            n: number of lines to take
            i: start position from 0.
            iterator: iterable
        Returns: List of lines
        Raises: StopIteration when there is none to take
        """
        if i <= 0:
            raise IndexError(f"Invalid index {i}")
        elif i == 0:
            return Function.take(n, iterator)
        else:
            _last = 0
            for _position, _line in enumerate(iterator):
                _last = _position
                if _position == (i - 1):
                    return [_line] + Function.take(n - 1, iterator)

            raise IndexError(f"i [Exceeded the max lines [{_last}]")

    @staticmethod
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
            Logger.error(e)
            raise RuntimeError("Read file failure") from e

    # ================================================================================
    # Instance
    # ================================================================================
    # --------------------------------------------------------------------------------
    # Instance properties
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            name: str,
            log_level: int = logging.ERROR
    ):
        """
        Args:
            name: ID name
        """
        super().__init__(name=name, log_level=log_level)
