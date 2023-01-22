"""
File utility
https://stackoverflow.com/a/34102855/4281353
"""
import errno
import logging
import os
import sys
import tempfile
import pathlib
from typing import (
    List,
    Dict,
    Set,
    Tuple
)

from util_logging import (
    get_logger,
)


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
ERROR_INVALID_NAME = 123
"""
Windows-specific error code indicating an invalid pathname.

See Also
----------
https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
    Official listing of all such codes.
"""


def is_file(path_to_file) -> bool:
    return pathlib.Path(path_to_file).is_file()


def is_dir(path_to_file) -> bool:
    return pathlib.Path(path_to_file).is_dir()


def rm_file(path_to_file):
    path = pathlib.Path(path_to_file)
    path.unlink()


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
                if exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
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


def is_path_creatable(pathname: str) -> bool:
    """
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    """
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)


def is_path_exists_or_creatable(pathname: str) -> bool:
    """
    `True` if the passed pathname is a valid pathname for the current OS _and_
    either currently exists or is hypothetically creatable; `False` otherwise.

    This function is guaranteed to _never_ raise exceptions.
    """
    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
                os.path.exists(pathname) or is_path_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False


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
        return is_pathname_valid(pathname) and (
                os.path.exists(pathname) or is_path_sibling_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False


def get_dir_name(path: str) -> str:
    """Get the directory from the path
    Args:
        path: path/to/file
    Returns: path/to
    """
    return str(pathlib.Path(path).parent)


def get_filename(path: str) -> str:
    """Get the filename of the file at the path including suffix
    Args:
        path: path/to/file
    Returns: filename of the file at the path without parent directories
    """
    return pathlib.Path(path).name


def get_file_basename(path: str) -> str:
    """Get the basename of the file without extension
    Args:
        path: path/to/file
    Returns: basename of the file at the path
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_file_suffix(path: str) -> str:
    """Get the suffix of the file at the path
    Args:
        path: path/to/file
    """
    return pathlib.Path(path).suffix


def list_files_in_directory(path: str, pattern: str = None) -> Set[str]:
    """List files in the directory matching glob pattern
    See
    * https://docs.python.org/3/library/glob.html
    * https://docs.python.org/3/library/fnmatch.html#module-fnmatch

    Args:
        path: path/to/dir
        pattern: glob pattern string or None
    Returns: Set of file names (not including path/to/dir) found
    """
    assert is_dir(path), f"[{path} does not exit or not a directory."
    if pattern is not None and len(pattern) > 0:
        return {
            str(p.resolve().name)
            for p in pathlib.Path(path).glob(pattern)
            if is_file(p)
        }
    else:
        return {
            str(p.resolve().name) for p
            in pathlib.Path(path).glob("**/*")
            if is_file(p)
        }


def mkdir(path: str, mode=0o777, create_parents: bool = True):
    """make directory if it does not exist and can be created. Do nothing if already exists.
    Args:
        path: path to the directory to create
        mode: permission to set
        create_parents: flag to create parents
    """
    name: str = "mkdir()"
    if is_file(path):
        msg: str = f"file [{path} exists.]"
        _logger.error("%s: %s", name, msg)
        raise RuntimeError(msg)

    elif not is_path_creatable(path):
        msg: str = f"cannot create [{path}]"
        _logger.error("%s: %s", name, msg)
        raise RuntimeError(msg)

    else:
        pathlib.Path(path).mkdir(mode=mode, parents=True, exist_ok=True)


def get_path_to_this_py_script() -> str:
    return str(pathlib.Path(__file__).resolve())
